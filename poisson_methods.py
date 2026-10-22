import numpy as np
import torch

"""
Biased asncombe transform (fails at peaks << 50)

def anscombe(x):
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + 3.0 / 8.0)

def anscombe_inv(d):
    return (d / 2.0)**2 - 3.0 / 8.0
"""

def anscombe(x):
  return 2.0 * np.sqrt(np.maximum(x, 0.0) + 3.0 / 8.0)

def anscombe_inv(d):
    return (d / 2.0)**2 - 3.0 / 8.0



def anscombe_inv_unbiased(d):
    """Inverse Anscombe (Mäkitalo & Foi)."""
    threshold = 2.0 * np.sqrt(3.0 / 8.0)
    d_safe = np.maximum(d, threshold)

    x = (
        0.25 * (d_safe**2)
        + 0.25 * np.sqrt(1.5) * (d_safe ** (-1))
        - (11.0 / 8.0) * (d_safe ** (-2))
        + (5.0 / 8.0) * np.sqrt(1.5) * (d_safe ** (-3))
        - 0.125
    )
    return np.where(d > threshold, np.maximum(x, 0.0), 0.0)




@torch.no_grad()
def denoise_anscombe(z, peak, vn, unbiased=True, sigma_ref=25.0):
    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 2: z = z[..., None]

    z_ansc = anscombe(z * peak)
    a_min = 2.0 * np.sqrt(3.0 / 8.0)
    delta = 2.0 * np.sqrt(peak + 3.0 / 8.0) - a_min
    z_input = ((z_ansc - a_min) / delta).astype(np.float32)

    parameter = next(vn.parameters())
    z_tensor = torch.from_numpy(
        np.ascontiguousarray(z_input.transpose(2, 0, 1)[None])
    ).to(device=parameter.device, dtype=parameter.dtype)

    sigma = 255.0 / delta
    scale = sigma_ref / sigma
    model_input = z_tensor * scale
    restored = vn(model_input, model_input)[-1] / scale
    x = restored[0].cpu().numpy().transpose(1, 2, 0)
    d = x * delta + a_min

    if unbiased:
        x_counts = anscombe_inv_unbiased(d)
    else:
        x_counts = anscombe_inv(d)

    return np.clip(x_counts / peak, 0.0, 1.0).astype(np.float32)


