"""PSNR de un checkpoint TDV sobre imágenes completas.

    python evaluate.py checkpoints/Oficiales/tdv3-3-25-f32-color.pth --file-list datasets/bsds500/cbsd68.txt
    python evaluate.py runs/replica/best.pth --sigma 50

Para sigma distinto del de entrenamiento se aplica el reescalado del paper (sec. 5.2): se lleva la
entrada al nivel de ruido de referencia y se devuelve la salida a su escala. Es una aproximación de
dominio, no una identidad: la red NO es escala-equivariante —bias-free no alcanza— y el esquema
completo se desvía 15-28 % de VNet(cz) = c·VNet(z). Medido el 2026-09-06, está en la bitácora.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ddr.data import ImageDataset2D, add_noise, add_poisson_noise, psnr
from ddr.data3d import UMBRAL_TEJIDO
from ddr.model import VNet2D, VNet3D
from ddr.qsm3d import nrmse_pct
from poisson_methods import denoise_anscombe
from skimage.metrics import structural_similarity


__all__ = ["load_vnet", "evaluate_denoising", "evaluate_denoising3d"]


def load_vnet(path, device="cpu", efficient: bool = False) -> tuple[VNet2D | VNet3D, dict]:
    """Carga un checkpoint (oficial o propio), 2D o 3D según el `type` de su regularizador: los
    de `train3d.py` dicen `tdv3d`. `strict=True` a propósito: si el port divergiera en nombres o
    formas, queremos que falle ruidosamente."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vnet = VNet3D if ck["config"]["R"].get("type") == "tdv3d" else VNet2D
    model = vnet(ck["config"], efficient=efficient)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), ck["config"]




def tensor_ssim(x, y, data_range=1.0):
    # El evaluador procesa una imagen NCHW por vez.
    x = x[0].detach().float().cpu().permute(1, 2, 0).numpy()
    y = y[0].detach().float().cpu().permute(1, 2, 0).numpy()

    return float(structural_similarity(
        x, y,
        data_range=data_range,
        channel_axis=-1,
    ))


@torch.no_grad()
def evaluate_denoising(model, dataset, sigma=25.0, sigma_ref=25.0, device="cpu",
                       seed=1234, data_range=1.0, noise_scale=255.0, noise_type='gaussian', poisson_peak=100.0, anscombe_mode='none') -> dict:
    """PSNR promedio sobre `dataset`, con una realización de ruido fija (gaussiana o poisson) por imagen."""

        
    scale = (sigma_ref / sigma) if noise_type == "gaussian" else 1.0 # esto hay que cambiarlo para hacer inferencia sobre la correspondencia sigma -> peak
    rows = []
    for i in range(len(dataset)):
        clean, name = dataset[i]
        clean = clean.unsqueeze(0).to(device)
        gen = torch.Generator(device="cpu").manual_seed(seed + i)

        if noise_type == 'poisson': noisy = add_poisson_noise(clean.cpu(), peak=poisson_peak, generator=gen).to(device)
        else: noisy = add_noise(clean.cpu(), sigma, gen, noise_scale).to(device)

        #denoised = model(noisy * scale, noisy * scale)[-1] / scale

        if anscombe_mode == "none":
            denoised = model(noisy * scale, noisy * scale)[-1] / scale
        else:
            z = noisy[0].permute(1, 2, 0).cpu().numpy()
            x = denoise_anscombe(z, peak=poisson_peak, vn=model, unbiased=(anscombe_mode == "unbiased"), sigma_ref=sigma_ref)

            denoised = (
                torch.from_numpy(np.ascontiguousarray(x))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=noisy.device, dtype=noisy.dtype)
            )
        
        rows.append({
            "name": name,
            "psnr_noisy": psnr(noisy, clean, data_range),
            "psnr_denoised": psnr(denoised, clean, data_range),
            "ssim_noisy": tensor_ssim(noisy, clean, data_range),
            "ssim_denoised": tensor_ssim(denoised, clean, data_range)
        })

    return {
        "noise_type": noise_type,
        "sigma": sigma if noise_type == "gaussian" else None,
        "poisson_peak": poisson_peak if noise_type == "poisson" else None,
        "anscombe": anscombe_mode,
        "num_images": len(rows),
        "psnr_noisy": float(np.mean([r["psnr_noisy"] for r in rows])),
        "psnr_denoised": float(np.mean([r["psnr_denoised"] for r in rows])),
        "ssim_noisy": float(np.mean([r["ssim_noisy"] for r in rows])),
        "ssim_denoised": float(np.mean([r["ssim_denoised"] for r in rows])),
        "per_image": rows,
    }


@torch.no_grad()
def evaluate_denoising3d(model, dataset, sigma=25.0, device="cpu", seed=1234, data_range=1.0,
                         noise_scale=255.0, amp=None, umbral_tejido=UMBRAL_TEJIDO,
                         devuelve_primero: bool = False) -> dict:
    """PSNR medio sobre los recortes fijos de un `VolumeDataset3D`, con una realización de ruido
    fija por volumen: en todo el recorte y sólo en tejido (vóxeles LIMPIOS > `umbral_tejido`).
    También el NRMSE (%) de la salida en los dos, y, si el dataset tiene `cerebro`, los PSNR en el
    cerebro (`*_cerebro`); sin él, esas claves no están.

    `amp` es el dtype de autocast del modo de la corrida (`MODES[modo]["amp"]`): se evalúa como se
    entrena. Sin reescalado de sigma: en 3D se evalúa siempre al sigma del entrenamiento. Con
    `devuelve_primero`, `primero` lleva limpia, ruidosa y salida del volumen 0, para la lámina.
    """
    tipo = torch.device(device).type
    cerebro = getattr(dataset, "cerebro", None)
    rows, primero = [], None
    for i in range(len(dataset)):
        clean, name = dataset[i]
        clean = clean.unsqueeze(0)
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        noisy = add_noise(clean, sigma, gen, noise_scale)
        entrada = noisy.to(device)
        with torch.autocast(tipo, dtype=amp, enabled=amp is not None):
            denoised = model(entrada, entrada)[-1].float().cpu()
        # La máscara sale del limpio: la del ruidoso cambiaría con cada realización.
        tejido = clean > umbral_tejido
        # En float64: la norma de 2 M de valores en float32 acumula en float32.
        y, x = clean.double().numpy(), denoised.double().numpy()
        rows.append({
            "name": name,
            "fraccion_tejido": float(tejido.float().mean()),
            "psnr_noisy": psnr(noisy, clean, data_range),
            "psnr_denoised": psnr(denoised, clean, data_range),
            "psnr_noisy_tejido": psnr(noisy[tejido], clean[tejido], data_range),
            "psnr_denoised_tejido": psnr(denoised[tejido], clean[tejido], data_range),
            "nrmse_denoised": nrmse_pct(x, y),
            "nrmse_denoised_tejido": nrmse_pct(x, y, tejido.numpy()),
        })
        if cerebro is not None:
            m = torch.from_numpy(cerebro[i])[None, None]
            rows[-1].update(psnr_noisy_cerebro=psnr(noisy[m], clean[m], data_range),
                            psnr_denoised_cerebro=psnr(denoised[m], clean[m], data_range))
        if devuelve_primero and i == 0:
            primero = (clean[0, 0].numpy(), noisy[0, 0].numpy(), denoised[0, 0].numpy())

    claves = ["psnr_noisy", "psnr_denoised", "psnr_noisy_tejido", "psnr_denoised_tejido",
              "nrmse_denoised", "nrmse_denoised_tejido"]
    if cerebro is not None:
        claves += ["psnr_noisy_cerebro", "psnr_denoised_cerebro"]
    return {
        "sigma": sigma,
        "num_volumenes": len(rows),
        **{k: float(np.mean([r[k] for r in rows])) for k in claves},
        "per_volume": rows,
        **({"primero": primero} if devuelve_primero else {}),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="PSNR de un checkpoint TDV en denoising gaussiano.")
    p.add_argument("checkpoint")
    p.add_argument("--data", default="datasets/bsds500", help="raíz del dataset")
    p.add_argument("--split", default="val")
    p.add_argument("--file-list", help="subconjunto canónico, p.ej. datasets/bsds500/cbsd68.txt")
    p.add_argument("--sigma", type=float, default=25.0)
    p.add_argument("--sigma-ref", type=float, default=25.0, help="sigma con el que se entrenó")
    p.add_argument("--gray", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", help="escribir el reporte JSON acá")
    p.add_argument("--noise-type", choices=["gaussian", "poisson"], default="gaussian",help="Tipo de ruido con el que degradar las imágenes de prueba")
    p.add_argument("--poisson-peak", type=float, default=100.0, help="Fotones peak si se usa ruido Poisson")
    p.add_argument("--anscombe", choices=["none", "biased", "unbiased"], default="none",  help="Inversa de Anscombe para evaluar ruido Poisson")
    args = p.parse_args()


    names = None
    if args.file_list:
        names = [l.strip() for l in Path(args.file_list).read_text().splitlines() if l.strip()]

    dataset = ImageDataset2D.from_root(args.data, args.split,
                                       channels=1 if args.gray else 3, names=names)
    model, config = load_vnet(args.checkpoint, args.device)
    report = evaluate_denoising(model, dataset, args.sigma, args.sigma_ref, args.device, noise_type=args.noise_type, poisson_peak=args.poisson_peak, anscombe_mode=args.anscombe)

    print(f"checkpoint : {args.checkpoint}")
    print(f"Anscombe   : {args.anscombe}")
    print(f"conjunto   : {args.split} ({report['num_images']} imágenes)")
    if args.noise_type == "poisson":
        print(f"ruido   : Poisson (peak={args.poisson_peak})")
    else:
        print(f"sigma   : {args.sigma}   S={config['S']}")
    print(f"PSNR ruido : {report['psnr_noisy']:.2f} dB")
    print(f"PSNR salida: {report['psnr_denoised']:.2f} dB")
    print(f"SSIM ruido : {report['ssim_noisy']:.4f}")
    print(f"SSIM salida: {report['ssim_denoised']:.4f}")

    if args.output:
        report["checkpoint"] = str(args.checkpoint)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"reporte    : {args.output}")


if __name__ == "__main__":
    main()
