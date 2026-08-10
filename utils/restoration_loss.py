import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff.square() + self.eps * self.eps).mean()


class LaplacianEdgeLoss(nn.Module):
    """Multi-scale Laplacian-pyramid loss for boundaries and fine texture."""

    def __init__(self, levels=3):
        super().__init__()
        kernel_1d = torch.tensor([[0.05, 0.25, 0.40, 0.25, 0.05]], dtype=torch.float32)
        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        self.register_buffer("kernel", kernel_2d.view(1, 1, 5, 5))
        self.levels = levels
        self.loss = CharbonnierLoss()

    def forward(self, pred, target):
        total = pred.new_zeros(())
        used = 0
        for level in range(self.levels):
            total = total + self.loss(self._laplacian(pred), self._laplacian(target)) / (2 ** level)
            used += 1
            if min(pred.shape[-2:]) < 32:
                break
            pred = F.avg_pool2d(pred, 2, 2)
            target = F.avg_pool2d(target, 2, 2)
        return total / sum(1.0 / (2 ** i) for i in range(used))

    def _conv_gauss(self, image):
        kernel = self.kernel.to(dtype=image.dtype, device=image.device).expand(image.shape[1], 1, 5, 5)
        image = F.pad(image, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(image, kernel, groups=image.shape[1])

    def _laplacian(self, image):
        filtered = self._conv_gauss(image)
        down = filtered[:, :, ::2, ::2]
        up = F.interpolate(down, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return image - up


class MultiScaleStructureLoss(nn.Module):
    """Differentiable multi-scale local luminance/contrast/structure loss."""

    def __init__(self, levels=3, window_size=7):
        super().__init__()
        self.levels = levels
        self.window_size = window_size

    def forward(self, pred, target):
        losses = []
        for _ in range(self.levels):
            losses.append(1.0 - self._ssim(pred, target))
            if min(pred.shape[-2:]) < 32:
                break
            pred = F.avg_pool2d(pred, 2, 2)
            target = F.avg_pool2d(target, 2, 2)
        return torch.stack(losses).mean()

    def _ssim(self, pred, target):
        k = min(self.window_size, pred.shape[-2], pred.shape[-1])
        if k % 2 == 0:
            k -= 1
        padding = k // 2
        mu_x = F.avg_pool2d(pred, k, 1, padding)
        mu_y = F.avg_pool2d(target, k, 1, padding)
        var_x = (F.avg_pool2d(pred.square(), k, 1, padding) - mu_x.square()).clamp_min(0.0)
        var_y = (F.avg_pool2d(target.square(), k, 1, padding) - mu_y.square()).clamp_min(0.0)
        cov_xy = F.avg_pool2d(pred * target, k, 1, padding) - mu_x * mu_y
        luminance = (2.0 * mu_x * mu_y + 0.01 ** 2) / (mu_x.square() + mu_y.square() + 0.01 ** 2)
        structure = (2.0 * cov_xy + 0.03 ** 2) / (var_x + var_y + 0.03 ** 2)
        return (luminance * structure).clamp(-1.0, 1.0).mean()


class PerceptualPyramidLoss(nn.Module):
    """Lightweight LPIPS-oriented proxy using contrast-normalized image pyramids."""

    def __init__(self, levels=3, eps=1e-3):
        super().__init__()
        self.levels = levels
        self.eps = eps

    def forward(self, pred, target):
        losses = []
        for _ in range(self.levels):
            pred_feat = self._features(pred)
            target_feat = self._features(target)
            losses.append(torch.sqrt((pred_feat - target_feat).square() + self.eps ** 2).mean())
            if min(pred.shape[-2:]) < 32:
                break
            pred = F.avg_pool2d(pred, 2, 2)
            target = F.avg_pool2d(target, 2, 2)
        return torch.stack(losses).mean()

    @staticmethod
    def _features(image):
        mean = F.avg_pool2d(image, 7, 1, 3)
        variance = F.avg_pool2d(image.square(), 7, 1, 3) - mean.square()
        contrast = (image - mean) * torch.rsqrt(variance.clamp_min(0.0) + 1e-4)
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        return torch.cat((image, contrast, grad_x, grad_y), dim=1)


class AdaptivePixelStructureRefinementLoss(nn.Module):
    """APSR-v2: balanced pixel, structure, frequency and perceptual refinement."""

    def __init__(self, mse_weight=1.0, hard_mse_weight=0.5, hard_gamma=1.0,
                 edge_weight=0.08, structure_weight=0.12, perceptual_weight=0.05,
                 color_weight=0.02, pyramid_levels=3, max_side=1024, eps=1e-6):
        super().__init__()
        self.mse_weight = mse_weight
        self.hard_mse_weight = hard_mse_weight
        self.hard_gamma = hard_gamma
        self.edge_weight = edge_weight
        self.structure_weight = structure_weight
        self.perceptual_weight = perceptual_weight
        self.color_weight = color_weight
        self.max_side = max_side
        self.eps = eps
        self.edge_loss = LaplacianEdgeLoss(pyramid_levels)
        self.structure_loss = MultiScaleStructureLoss(pyramid_levels)
        self.perceptual_loss = PerceptualPyramidLoss(pyramid_levels)

    def forward(self, pred, target):
        pred, target = self._prepare_pair(pred, target)
        diff = pred - target
        mse_loss = diff.square().mean()
        hard_mse_loss = self._hard_pixel_mse(diff)

        detail_pred, detail_target = self._resize_pair(pred, target)
        edge_loss = self.edge_loss(detail_pred, detail_target)
        structure_loss = self.structure_loss(detail_pred, detail_target)
        perceptual_loss = self.perceptual_loss(detail_pred, detail_target)
        color_loss = self._color_loss(pred, target)

        total = (self.mse_weight * mse_loss + self.hard_mse_weight * hard_mse_loss
                 + self.edge_weight * edge_loss + self.structure_weight * structure_loss
                 + self.perceptual_weight * perceptual_loss + self.color_weight * color_loss)
        return total, {
            "mse": mse_loss.detach(), "hard_mse": hard_mse_loss.detach(),
            "edge": edge_loss.detach(), "structure": structure_loss.detach(),
            "perceptual": perceptual_loss.detach(), "color": color_loss.detach(),
        }

    @torch.no_grad()
    def density_map(self, pred, target, clamp_max=6.0, return_stats=False):
        """Build a normalized APSR residual map for Gaussian density control.

        The map is detached and normalized to have an image-wise mean near 1.0,
        so callers can safely use a small external multiplier.
        """
        pred, target = self._prepare_pair(pred, target)
        maps = self.diagnostic_maps(pred, target)

        density = (
            self.mse_weight * self._normalize_map(maps["mse"], clamp_max)
            + self.hard_mse_weight * self._normalize_map(maps["hard_mse"], clamp_max)
            + self.edge_weight * self._normalize_map(maps["edge"], clamp_max)
            + self.structure_weight * self._normalize_map(maps["structure"], clamp_max)
            + self.perceptual_weight * self._normalize_map(maps["perceptual"], clamp_max)
            + self.color_weight * self._normalize_map(maps["color"], clamp_max)
        )
        density = self._normalize_map(density, clamp_max)

        if not return_stats:
            return density

        stats = {
            "density_mean": density.mean().detach(),
            "density_max": density.max().detach(),
        }
        for name, value in maps.items():
            stats[f"{name}_mean"] = value.mean().detach()
        return density, stats

    @torch.no_grad()
    def diagnostic_maps(self, pred, target):
        pred, target = self._prepare_pair(pred, target)
        diff = pred - target
        mse_map = diff.square().mean(dim=1, keepdim=True)
        hard_mse_map = self._hard_pixel_mse_map(diff)
        edge_map = self._multi_scale_map(pred, target, self._edge_map)
        structure_map = self._multi_scale_map(pred, target, self._structure_map)
        perceptual_map = self._multi_scale_map(pred, target, self._perceptual_map)
        color_map = self._color_map(pred, target)

        return {
            "mse": mse_map,
            "hard_mse": hard_mse_map,
            "edge": edge_map,
            "structure": structure_map,
            "perceptual": perceptual_map,
            "color": color_map,
        }

    def _prepare_pair(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        if pred.shape != target.shape:
            raise ValueError(f"APSR expects matched shapes, got {pred.shape} and {target.shape}")
        if pred.shape[1] != 3:
            raise ValueError(f"APSR expects RGB input, got {pred.shape[1]} channels")
        return pred.float(), target.float()

    def _hard_pixel_mse(self, diff):
        # Detached, saturating weights focus useful hard regions without letting
        # occlusion/specular outliers dominate the Gaussian updates.
        error = diff.detach().square().mean(dim=1, keepdim=True).sqrt()
        relative = error / error.mean().clamp_min(self.eps)
        weights = 0.5 + 1.5 * relative.pow(self.hard_gamma) / (1.0 + relative.pow(self.hard_gamma))
        weights = weights / weights.mean().clamp_min(self.eps)
        return (weights * diff.square()).mean()

    def _hard_pixel_mse_map(self, diff):
        error = diff.square().mean(dim=1, keepdim=True).sqrt()
        relative = error / error.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        weights = 0.5 + 1.5 * relative.pow(self.hard_gamma) / (1.0 + relative.pow(self.hard_gamma))
        weights = weights / weights.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return weights * diff.square().mean(dim=1, keepdim=True)

    @staticmethod
    def _color_loss(pred, target):
        # Chroma supervision reduces perceptually obvious colour shifts while
        # leaving luminance accuracy to the pixel and structure terms.
        pred_cb = pred[:, 2:3] - pred[:, 1:2]
        pred_cr = pred[:, 0:1] - pred[:, 1:2]
        target_cb = target[:, 2:3] - target[:, 1:2]
        target_cr = target[:, 0:1] - target[:, 1:2]
        return F.smooth_l1_loss(torch.cat((pred_cb, pred_cr), 1),
                                torch.cat((target_cb, target_cr), 1), beta=0.02)

    def _resize_pair(self, pred, target):
        if not self.max_side or max(pred.shape[-2:]) <= self.max_side:
            return pred, target
        scale = self.max_side / float(max(pred.shape[-2:]))
        size = (max(16, round(pred.shape[-2] * scale)), max(16, round(pred.shape[-1] * scale)))
        return (F.interpolate(pred, size=size, mode="area"),
                F.interpolate(target, size=size, mode="area"))

    def _multi_scale_map(self, pred, target, map_fn):
        base_size = pred.shape[-2:]
        total = pred.new_zeros((pred.shape[0], 1, *base_size))
        weight_sum = 0.0
        for level in range(self.edge_loss.levels):
            weight = 1.0 / (2 ** level)
            level_map = map_fn(pred, target)
            if level_map.shape[-2:] != base_size:
                level_map = F.interpolate(level_map, size=base_size, mode="bilinear", align_corners=False)
            total = total + weight * level_map
            weight_sum += weight
            if min(pred.shape[-2:]) < 32:
                break
            pred = F.avg_pool2d(pred, 2, 2)
            target = F.avg_pool2d(target, 2, 2)
        return total / max(weight_sum, self.eps)

    def _edge_map(self, pred, target):
        edge_pred = self.edge_loss._laplacian(pred)
        edge_target = self.edge_loss._laplacian(target)
        return torch.sqrt((edge_pred - edge_target).square() + self.eps ** 2).mean(dim=1, keepdim=True)

    def _structure_map(self, pred, target):
        k = min(self.structure_loss.window_size, pred.shape[-2], pred.shape[-1])
        if k % 2 == 0:
            k -= 1
        padding = k // 2
        mu_x = F.avg_pool2d(pred, k, 1, padding)
        mu_y = F.avg_pool2d(target, k, 1, padding)
        var_x = (F.avg_pool2d(pred.square(), k, 1, padding) - mu_x.square()).clamp_min(0.0)
        var_y = (F.avg_pool2d(target.square(), k, 1, padding) - mu_y.square()).clamp_min(0.0)
        cov_xy = F.avg_pool2d(pred * target, k, 1, padding) - mu_x * mu_y
        luminance = (2.0 * mu_x * mu_y + 0.01 ** 2) / (mu_x.square() + mu_y.square() + 0.01 ** 2)
        structure = (2.0 * cov_xy + 0.03 ** 2) / (var_x + var_y + 0.03 ** 2)
        return (1.0 - (luminance * structure).clamp(-1.0, 1.0)).mean(dim=1, keepdim=True).clamp_min(0.0)

    def _perceptual_map(self, pred, target):
        pred_feat = self.perceptual_loss._features(pred)
        target_feat = self.perceptual_loss._features(target)
        return torch.sqrt((pred_feat - target_feat).square() + self.eps ** 2).mean(dim=1, keepdim=True)

    @staticmethod
    def _color_map(pred, target):
        pred_chroma = torch.cat((pred[:, 2:3] - pred[:, 1:2], pred[:, 0:1] - pred[:, 1:2]), 1)
        target_chroma = torch.cat((target[:, 2:3] - target[:, 1:2], target[:, 0:1] - target[:, 1:2]), 1)
        return F.smooth_l1_loss(pred_chroma, target_chroma, beta=0.02, reduction="none").mean(dim=1, keepdim=True)

    def _normalize_map(self, residual_map, clamp_max):
        residual_map = torch.nan_to_num(residual_map, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        mean = residual_map.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        normalized = residual_map / mean
        if clamp_max and clamp_max > 0:
            normalized = normalized.clamp(max=clamp_max)
        return normalized
