"""Training-view calibrated damping of unsupported render frequencies."""

import torch
import torch.nn.functional as F


def split_low_detail(image, kernel_size=3):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    padding = int(kernel_size) // 2
    low = F.avg_pool2d(F.pad(image, (padding,) * 4, mode="reflect"), int(kernel_size), stride=1)
    return low, image - low


def apply_frequency_calibration(image, coefficient, kernel_size=3):
    squeeze = image.dim() == 3
    source = image.unsqueeze(0) if squeeze else image
    _, detail = split_low_detail(source, kernel_size)
    coefficient = coefficient.to(device=source.device, dtype=source.dtype).view(1, -1, 1, 1)
    output = (source + coefficient * detail).clamp(0.0, 1.0)
    return output[0] if squeeze else output


@torch.no_grad()
def fit_frequency_calibration(cameras, render_fn, mode="scalar", max_views=32,
                              kernel_size=3, min_coefficient=-0.75,
                              max_coefficient=0.0):
    """Solve the least-squares detail coefficient using training views only."""
    if not cameras:
        return None, None
    view_count = min(max(int(max_views), 1), len(cameras))
    indices = torch.linspace(0, len(cameras) - 1, view_count).round().long().tolist()
    channels = 1 if str(mode).lower() == "scalar" else 3
    numerator = torch.zeros(channels, dtype=torch.float64, device="cuda")
    denominator = torch.zeros_like(numerator)
    before_sse = torch.zeros((), dtype=torch.float64, device="cuda")
    after_sse = torch.zeros_like(before_sse)
    pixel_count = 0

    cached = []
    for index in indices:
        rendered = render_fn(cameras[index]).clamp(0.0, 1.0)
        target = cameras[index].original_image[:3].to(rendered.device).clamp(0.0, 1.0)
        _, detail = split_low_detail(rendered, kernel_size)
        detail = detail[0]
        residual = target - rendered
        if channels == 1:
            numerator += (detail.double() * residual.double()).sum()
            denominator += detail.double().square().sum()
        else:
            numerator += (detail.double() * residual.double()).flatten(1).sum(1)
            denominator += detail.double().square().flatten(1).sum(1)
        before_sse += residual.double().square().sum()
        pixel_count += residual.numel()
        cached.append((rendered, target))

    coefficient = (numerator / denominator.clamp_min(1e-12)).clamp(
        min=float(min_coefficient), max=float(max_coefficient)
    ).float()
    for rendered, target in cached:
        corrected = apply_frequency_calibration(rendered, coefficient, kernel_size)
        after_sse += (corrected - target).double().square().sum()

    stats = {
        "views": view_count,
        "coefficient": coefficient.tolist(),
        "before_mse": (before_sse / pixel_count).item(),
        "after_mse": (after_sse / pixel_count).item(),
        "mse_reduction": (1.0 - after_sse / before_sse.clamp_min(1e-12)).item(),
        "detail_energy": (denominator.sum() / pixel_count).item(),
    }
    return coefficient, stats
