import torch
import torch.nn.functional as F
from fused_ssim import FusedSSIMMap
from fused_ssim import fused_ssim as fast_ssim
from utils.image_utils import psnr

from gaussian_renderer import render_fastgs
from utils.loss_utils import l1_loss
from utils.frequency_calibration import split_low_detail


def fast_ssim_map(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean(0)


def sign_log1p(x):
    return x.sign() * torch.log1p(x.abs())


def normalize_features(features):
    """对特征进行标准化归一化"""
    return (features - features.mean(0, keepdim=True)) / (features.std(0, keepdim=True) + 1e-6)


def _apsr_density_enabled(opt, apsr_loss_fn, iteration=None):
    if apsr_loss_fn is None:
        return False
    if not bool(getattr(opt, "use_apsr_density_control", 0)):
        return False
    if getattr(opt, "lambda_apsr_density", 0.0) <= 0:
        return False
    if iteration is not None and iteration < getattr(opt, "apsr_density_start_iter", 0):
        return False
    return True


def _apsr_density_state_enabled(opt):
    return (
        bool(getattr(opt, "use_apsr_density_control", 0))
        and bool(getattr(opt, "apsr_density_use_state", 1))
        and getattr(opt, "lambda_apsr_density", 0.0) > 0
    )


def _sfr_aux_state_enabled(opt):
    return (
        bool(getattr(opt, "sfr_aux_enable", 0))
        and bool(getattr(opt, "sfr_aux_use_state", 1))
    )


def _sfr_aux_active(opt, iteration=None):
    if not _sfr_aux_state_enabled(opt):
        return False
    if iteration is not None and iteration < getattr(opt, "sfr_aux_start_iter", 0):
        return False
    return True


def _sfr_aux_ramp(opt, iteration=None):
    if iteration is None:
        return 1.0
    start_iter = int(getattr(opt, "sfr_aux_start_iter", 0))
    ramp_iters = int(getattr(opt, "sfr_aux_ramp_iters", 0))
    if iteration < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return min(float(iteration - start_iter + 1) / ramp_iters, 1.0)


def _normalize_aux_map(value, clamp_max=6.0, eps=1e-6):
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mean = value.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    value = value / mean
    if clamp_max and clamp_max > 0:
        value = value.clamp(max=float(clamp_max))
    return value


@torch.no_grad()
def _build_sfr_aux_metric_map(render_image, gt_image, opt, iteration=None):
    """Build a detached SFR signal for RL density control.

    The signal follows the useful part of SFR-GS: high-frequency residuals
    receive more attention at strong spatial transitions.  It is deliberately
    used only as a state feature, never added to the reconstruction loss.
    """
    if not _sfr_aux_active(opt, iteration):
        return None, None

    pred = render_image.float().unsqueeze(0) if render_image.dim() == 3 else render_image.float()
    target = gt_image.float().unsqueeze(0) if gt_image.dim() == 3 else gt_image.float()
    if pred.shape != target.shape:
        raise ValueError(f"SFR auxiliary map expects matched shapes, got {pred.shape} and {target.shape}")

    kernel_size = max(int(getattr(opt, "sfr_aux_kernel_size", 5)), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    _, pred_detail = split_low_detail(pred, kernel_size)
    _, target_detail = split_low_detail(target, kernel_size)
    detail_residual = torch.sqrt(
        (pred_detail - target_detail).square() + 1e-6
    ).mean(dim=1, keepdim=True)

    # Use the target gradient as a stable spatial prior instead of amplifying
    # transient render noise during early densification.
    luminance = (
        0.299 * target[:, 0:1]
        + 0.587 * target[:, 1:2]
        + 0.114 * target[:, 2:3]
    )
    grad_x = F.pad(luminance[:, :, :, 1:] - luminance[:, :, :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(luminance[:, :, 1:, :] - luminance[:, :, :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
    gradient = _normalize_aux_map(gradient, clamp_max=2.0)

    rho = _sfr_aux_ramp(opt, iteration)
    tau = max(float(getattr(opt, "sfr_aux_tau", 0.5)), 0.0)
    spatial_weight = 1.0 + tau * rho * gradient
    score = _normalize_aux_map(
        _normalize_aux_map(
            detail_residual,
            clamp_max=getattr(opt, "sfr_aux_map_clamp", 6.0),
        ) * spatial_weight,
        clamp_max=getattr(opt, "sfr_aux_map_clamp", 6.0),
    )
    stats = {
        "rho": torch.as_tensor(rho, device=pred.device),
        "detail_mean": detail_residual.mean().detach(),
        "gradient_mean": gradient.mean().detach(),
        "score_mean": score.mean().detach(),
        "score_max": score.max().detach(),
    }
    # A negative map keeps the rasterizer's original metric scale unchanged
    # while making accum_metric_counts a positive SFR exposure statistic.
    return -score[0, 0].reshape(-1).contiguous(), stats


def _apsr_density_weight(opt, iteration=None, base_weight=None):
    weight = float(getattr(opt, "lambda_apsr_density", 0.0) if base_weight is None else base_weight)
    if iteration is None:
        return weight

    start_iter = getattr(opt, "apsr_density_start_iter", 0)
    warmup_iters = getattr(opt, "apsr_density_warmup_iters", 0)
    if iteration < start_iter:
        return 0.0
    if warmup_iters > 0:
        warmup_step = min(iteration - start_iter + 1, warmup_iters)
        weight *= warmup_step / warmup_iters
    return weight


def _build_apsr_metric_map(render_image, gt_image, opt, apsr_loss_fn, iteration=None, state_only=False, base_weight=None):
    if not _apsr_density_enabled(opt, apsr_loss_fn, iteration):
        return None, None

    density_map, stats = apsr_loss_fn.density_map(
        render_image.unsqueeze(0),
        gt_image.unsqueeze(0),
        clamp_max=getattr(opt, "apsr_density_map_clamp", 6.0),
        return_stats=True,
    )
    weight = _apsr_density_weight(opt, iteration, base_weight=base_weight)
    metric_map = (density_map[0, 0] * weight).reshape(-1).contiguous()
    if state_only:
        metric_map = -metric_map
    stats["weight"] = torch.as_tensor(weight, device=render_image.device)
    stats["signed_weight"] = torch.as_tensor(-weight if state_only else weight, device=render_image.device)
    stats["state_only"] = torch.as_tensor(1.0 if state_only else 0.0, device=render_image.device)
    return metric_map, stats


def _log_apsr_density_stats(tb_writer, opt, iteration, prefix, stats_list, apsr_scores=None, metric_score=None, visible_mask=None):
    if tb_writer is None or iteration is None:
        return
    interval = max(int(getattr(opt, "apsr_density_log_interval", 100)), 1)
    if iteration % interval != 0:
        return

    if stats_list:
        keys = sorted(stats_list[0].keys())
        for key in keys:
            values = [float(stats[key].detach().mean().item()) for stats in stats_list if key in stats]
            if values:
                tb_writer.add_scalar(f"{prefix}/{key}", sum(values) / len(values), iteration)

    if apsr_scores is not None and apsr_scores.numel() > 0:
        tb_writer.add_scalar(f"{prefix}/gs_apsr_score_mean", apsr_scores.mean().item(), iteration)
        tb_writer.add_scalar(f"{prefix}/gs_apsr_score_std", apsr_scores.std(unbiased=False).item(), iteration)
        if visible_mask is not None and visible_mask.any():
            tb_writer.add_scalar(f"{prefix}/visible_gs_apsr_score_mean", apsr_scores[visible_mask].mean().item(), iteration)

    if metric_score is not None and metric_score.numel() > 0:
        tb_writer.add_scalar(f"{prefix}/metric_score_mean", metric_score.mean().item(), iteration)
        tb_writer.add_scalar(f"{prefix}/metric_score_std", metric_score.std(unbiased=False).item(), iteration)


def _log_sfr_aux_stats(tb_writer, opt, iteration, sfr_stats, sfr_scores=None,
                       corr=None, corr_gate=None, state_feature=None):
    if tb_writer is None or iteration is None:
        return
    interval = max(int(getattr(opt, "sfr_aux_log_interval", 100)), 1)
    if iteration % interval != 0:
        return

    if sfr_stats:
        keys = sorted(sfr_stats[0].keys())
        for key in keys:
            values = [
                float(stats[key].detach().mean().item())
                for stats in sfr_stats if key in stats
            ]
            if values:
                tb_writer.add_scalar(
                    f"sfr_aux/map_{key}", sum(values) / len(values), iteration
                )
    if sfr_scores is not None and sfr_scores.numel() > 0:
        tb_writer.add_scalar("sfr_aux/gs_score_mean", sfr_scores.mean().item(), iteration)
        tb_writer.add_scalar(
            "sfr_aux/gs_score_std",
            sfr_scores.std(unbiased=False).item(),
            iteration,
        )
    if corr is not None:
        tb_writer.add_scalar("sfr_aux/metric_corr", corr.item(), iteration)
    if corr_gate is not None:
        tb_writer.add_scalar("sfr_aux/corr_gate", corr_gate.item(), iteration)
    if state_feature is not None and state_feature.numel() > 0:
        tb_writer.add_scalar(
            "sfr_aux/state_feature_abs_mean",
            state_feature.detach().abs().mean().item(),
            iteration,
        )


def _sampled_corrcoef(x, y, mask=None, max_samples=65536):
    x = x.detach().flatten().float()
    y = y.detach().flatten().float()
    if mask is not None:
        mask = mask.detach().flatten()
        x = x[mask]
        y = y[mask]
    if x.numel() < 2:
        return None
    if x.numel() > max_samples:
        stride = max((x.numel() + max_samples - 1) // max_samples, 1)
        x = x[::stride]
        y = y[::stride]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if not torch.isfinite(denom).item() or denom.item() <= 1e-8:
        return None
    return (x * y).mean() / denom


def get_metric_score(camlist, gaussians, pipe, bg, opt, apsr_loss_fn=None, iteration=None, tb_writer=None, log_prefix="apsr_density/reward"):
    """
    计算每个Gaussian点对渲染误差的贡献度
    """
    num_points = gaussians.get_xyz.shape[0]
    metric_score = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    gs_weights = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    apsr_scores = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    apsr_stats = []

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        gt_image = my_viewpoint_cam.original_image.cuda()
        metric_map = None

        if _apsr_density_enabled(opt, apsr_loss_fn, iteration) and bool(getattr(opt, "apsr_density_use_reward", 0)):
            with torch.no_grad():
                render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult)
                render_image = render_pkg["render"].clamp(0.0, 1.0)
            gt_for_map = gt_image.clamp(0.0, 1.0)
            reward_weight = float(getattr(opt, "lambda_apsr_density", 0.0)) * float(getattr(opt, "apsr_density_reward_weight", 0.05))
            metric_map, stats = _build_apsr_metric_map(
                render_image,
                gt_for_map,
                opt,
                apsr_loss_fn,
                iteration,
                state_only=False,
                base_weight=reward_weight,
            )
            if stats is not None:
                apsr_stats.append(stats)
            del render_pkg, render_image, gt_for_map
        
        with torch.no_grad():
            render_pkg2 = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult, get_flag=True, metric_map=metric_map, gt_image=gt_image)
            accum_metric_per_gs = render_pkg2["accum_metric_per_gs"]
            accum_metric_counts = render_pkg2["accum_metric_counts"]
            accum_gs_weight = render_pkg2["accum_gs_weight"]

            metric_score += accum_metric_per_gs
            gs_weights += accum_gs_weight
            if accum_metric_counts.numel() > 0:
                apsr_scores += accum_metric_counts

    visible_mask = gs_weights > 0
    metric_score /= len(camlist)
    metric_score = sign_log1p(metric_score).clamp(min=-6.0, max=6.0)
    apsr_scores /= len(camlist)
    _log_apsr_density_stats(tb_writer, opt, iteration, log_prefix, apsr_stats, apsr_scores if apsr_stats else None, metric_score, visible_mask)
    return metric_score, visible_mask


def get_gaussians_state_for_rl(camlist, gaussians, pipe, bg, opt, apsr_loss_fn=None, iteration=None, tb_writer=None):
    """
    构建RL的状态特征
    """
    num_points = len(gaussians.get_xyz)
    
    xyz_grads = torch.zeros(num_points, 3, device="cuda", dtype=torch.float32)
    scale_grads = torch.zeros(num_points, 3, device="cuda", dtype=torch.float32)
    opacity_grads = torch.zeros(num_points, 1, device="cuda", dtype=torch.float32)
    feature_dc_grads = torch.zeros(num_points, 3, device="cuda", dtype=torch.float32)
    metric_score = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    apsr_scores = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    sfr_scores = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    gs_weights = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    apsr_stats = []
    sfr_stats = []

    n_views = len(camlist)
    
    for view in range(n_views):
        my_viewpoint_cam = camlist[view]
        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult)

        render_image = render_pkg["render"]
        gt_image = my_viewpoint_cam.original_image.cuda()

        render_image.clamp_(min=0.0, max=1.0)
        gt_image.clamp_(min=0.0, max=1.0)

        Ll1 = l1_loss(render_image, gt_image)
        ssim_value = fast_ssim(render_image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        metric_map, stats = _build_apsr_metric_map(
            render_image.detach(),
            gt_image.detach(),
            opt,
            apsr_loss_fn,
            iteration,
            state_only=True,
        )
        if stats is not None:
            apsr_stats.append(stats)

        xyz_grads += gaussians._xyz.grad
        scale_grads += gaussians._scaling.grad
        opacity_grads += gaussians._opacity.grad
        feature_dc_grads += gaussians._features_dc.grad.squeeze(1)

        gaussians.optimizer.zero_grad(set_to_none=True)
        if getattr(gaussians, "shoptimizer", None) is not None:
            gaussians.shoptimizer.zero_grad(set_to_none=True)
        gaussians.clear_grad()

        with torch.no_grad():
            render_pkg2 = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult, get_flag=True, metric_map=metric_map, gt_image=gt_image)
            accum_metric_per_gs = render_pkg2["accum_metric_per_gs"]
            accum_metric_counts = render_pkg2["accum_metric_counts"]
            accum_gs_weight = render_pkg2["accum_gs_weight"]

            metric_score += accum_metric_per_gs
            if accum_metric_counts.numel() > 0:
                apsr_scores += accum_metric_counts
            gs_weights += accum_gs_weight

        # Collect SFR exposure separately so it cannot alter the baseline
        # metric/reward signal or get entangled with the APSR map.
        if _sfr_aux_active(opt, iteration):
            sfr_metric_map, sfr_map_stats = _build_sfr_aux_metric_map(
                render_image.detach(),
                gt_image.detach(),
                opt,
                iteration,
            )
            if sfr_map_stats is not None:
                sfr_stats.append(sfr_map_stats)
            with torch.no_grad():
                sfr_render_pkg = render_fastgs(
                    my_viewpoint_cam,
                    gaussians,
                    pipe,
                    bg,
                    opt.mult,
                    get_flag=True,
                    metric_map=sfr_metric_map,
                    gt_image=gt_image,
                )
                sfr_counts = sfr_render_pkg["accum_metric_counts"]
                if sfr_counts.numel() > 0:
                    sfr_scores += sfr_counts
            del sfr_render_pkg, sfr_metric_map

        del render_pkg, render_image, gt_image, loss, render_pkg2, my_viewpoint_cam

    visible_mask = gs_weights > 0

    metric_score /= len(camlist)
    raw_metric_score = metric_score.clone().detach()
    metric_score = sign_log1p(metric_score)
    apsr_scores /= len(camlist)
    sfr_scores /= len(camlist)
    metric_score_feature = metric_score.clone().detach().unsqueeze(-1)
    metric_score_feature = normalize_features(metric_score_feature)
    
    grad_features = torch.cat([
        xyz_grads.clone().detach(),
        scale_grads.clone().detach(),
        opacity_grads.clone().detach(),
        feature_dc_grads.clone().detach(),
    ], dim=-1)
    grad_features = normalize_features(grad_features)

    states = torch.cat([
        grad_features,
        metric_score_feature,
    ], dim=-1)

    if _apsr_density_state_enabled(opt):
        if _apsr_density_enabled(opt, apsr_loss_fn, iteration):
            apsr_score_feature = sign_log1p(apsr_scores).clone().detach().unsqueeze(-1)
            apsr_score_feature = normalize_features(apsr_score_feature)
            contribution_gate = None
            corr = _sampled_corrcoef(apsr_scores, raw_metric_score, visible_mask)
            corr_gate = None
            if bool(getattr(opt, "apsr_density_use_contribution_gate", 1)):
                gate_temperature = max(float(getattr(opt, "apsr_density_gate_temperature", 1.0)), 1e-6)
                gate_floor = min(max(float(getattr(opt, "apsr_density_gate_floor", 0.0)), 0.0), 1.0)
                contribution_gate = torch.sigmoid(metric_score_feature.detach() / gate_temperature)
                if gate_floor > 0:
                    contribution_gate = gate_floor + (1.0 - gate_floor) * contribution_gate
                if bool(getattr(opt, "apsr_density_use_corr_gate", 1)) and corr is not None:
                    corr_gate_floor = min(max(float(getattr(opt, "apsr_density_corr_gate_floor", 0.1)), 0.0), 1.0)
                    corr_gate = torch.clamp((corr + 1.0) * 0.5, min=corr_gate_floor, max=1.0)
                    contribution_gate = contribution_gate * corr_gate
                apsr_score_feature = apsr_score_feature * contribution_gate
        else:
            apsr_score_feature = torch.zeros_like(metric_score_feature)
            contribution_gate = None
            corr = None
            corr_gate = None
        apsr_score_feature = apsr_score_feature * float(getattr(opt, "apsr_density_state_weight", 1.0))
        states = torch.cat([states, apsr_score_feature], dim=-1)

        if (
            tb_writer is not None
            and iteration is not None
            and iteration % max(int(getattr(opt, "apsr_density_log_interval", 100)), 1) == 0
            and contribution_gate is not None
        ):
            tb_writer.add_scalar("apsr_density/state/contribution_gate_mean", contribution_gate.mean().item(), iteration)
            if visible_mask.any():
                tb_writer.add_scalar("apsr_density/state/contribution_gate_visible_mean", contribution_gate[visible_mask].mean().item(), iteration)
            if corr is not None:
                tb_writer.add_scalar("apsr_density/state/apsr_metric_corr", corr.item(), iteration)
            if corr_gate is not None:
                tb_writer.add_scalar("apsr_density/state/corr_gate", corr_gate.item(), iteration)

    sfr_state_feature = None
    if _sfr_aux_state_enabled(opt):
        if _sfr_aux_active(opt, iteration):
            sfr_state_feature = sign_log1p(sfr_scores).clone().detach().unsqueeze(-1)
            sfr_state_feature = normalize_features(sfr_state_feature)
            sfr_corr = _sampled_corrcoef(sfr_scores, raw_metric_score, visible_mask)
            sfr_corr_gate = None
            corr_min = min(
                max(float(getattr(opt, "sfr_aux_corr_gate_min", 0.20)), 0.0),
                0.99,
            )
            if sfr_corr is not None:
                sfr_corr_gate = torch.clamp(
                    (sfr_corr - corr_min) / max(1.0 - corr_min, 1e-6),
                    min=0.0,
                    max=1.0,
                )
            if sfr_corr_gate is None:
                sfr_corr_gate = torch.zeros((), device=states.device)
            if bool(getattr(opt, "sfr_aux_use_contribution_gate", 1)):
                contribution_gate = torch.sigmoid(metric_score_feature.detach())
                sfr_state_feature = sfr_state_feature * contribution_gate
            sfr_state_feature = (
                sfr_state_feature
                * sfr_corr_gate
                * _sfr_aux_ramp(opt, iteration)
                * float(getattr(opt, "sfr_aux_state_weight", 0.25))
            )
        else:
            sfr_corr = None
            sfr_corr_gate = None
            sfr_state_feature = torch.zeros_like(metric_score_feature)
        states = torch.cat([states, sfr_state_feature], dim=-1)
    else:
        sfr_corr = None
        sfr_corr_gate = None

    _log_apsr_density_stats(
        tb_writer,
        opt,
        iteration,
        "apsr_density/state",
        apsr_stats,
        apsr_scores,
        metric_score,
        visible_mask,
    )
    _log_sfr_aux_stats(
        tb_writer,
        opt,
        iteration,
        sfr_stats,
        sfr_scores if _sfr_aux_active(opt, iteration) else None,
        sfr_corr,
        sfr_corr_gate,
        sfr_state_feature,
    )

    return states, metric_score, visible_mask
