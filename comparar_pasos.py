"""Compara S=10 y S=20 sobre evento_paso.pth, con forward y backward.

Desde Training-TDV: python comparar_pasos.py --device cuda
Busca capturas en runs/ y, si faltan, en ../tdv-tuning/checkpoints/.
Usa el modelo de entrenamiento y sus hooks para medir los tensores internos reales.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent


def medir(model, muestra, steps, acumulacion, device, micro_batch=4):
    """La loss se suma por imagen, igual que en train.py."""
    model.S = steps
    model.zero_grad(set_to_none=True)
    etapas = []
    if micro_batch < 1 or acumulacion < 1:
        raise ValueError("micro_batch y acumulacion deben ser positivos.")
    total = muestra["clean"].shape[0]
    if total == 0 or muestra["clean"].shape != muestra["noisy"].shape:
        raise ValueError("Se requieren clean/noisy del mismo tamaño y un lote no vacío.")
    combinadas = [{"etapa": s + 1, "grad_v": {"norma": 0., "max_abs": 0.},
                   "grad_x_out": {"norma": 0., "max_abs": 0.}} for s in range(steps)]
    perdida = 0.0

    def observar(nombre, tensor, backward=True):
        if nombre == "x_in":
            etapas.append({"etapa": len(etapas) + 1})
        if nombre not in ("v", "x_out"):
            return
        fila = etapas[-1]

        def guardar(grad):
            grad = grad.detach().float()
            fila["grad_" + nombre] = {
                "norma": grad.norm().item(),
                "max_abs": grad.abs().amax().item(),
            }

        if tensor.requires_grad:
            tensor.register_hook(guardar)

    model.D.observe = observar
    try:
        for inicio in range(0, total, micro_batch):
            etapas = []
            clean = muestra["clean"][inicio:inicio + micro_batch].to(device)
            noisy = muestra["noisy"][inicio:inicio + micro_batch].to(device)
            ultimo = model(noisy, noisy)[-1]
            # El denominador es el lote original, incluso para el último trozo.
            loss = (ultimo.float() - clean.float()).square().sum() / total
            (loss / acumulacion).backward()
            perdida += loss.item()
            if len(etapas) != steps or any("grad_v" not in e or "grad_x_out" not in e for e in etapas):
                raise RuntimeError("Faltan hooks DEBUG en Training-TDV/ddr/model.py.")
            for acumulada, etapa in zip(combinadas, etapas):
                for nombre in ("grad_v", "grad_x_out"):
                    valores = etapa[nombre]
                    if not all(math.isfinite(v) for v in valores.values()):
                        raise RuntimeError(f"Gradientes no finitos con S={steps}.")
                    # Norma del tensor concatenado: raíz de la suma de cuadrados.
                    acumulada[nombre]["norma"] = math.hypot(acumulada[nombre]["norma"], valores["norma"])
                    acumulada[nombre]["max_abs"] = max(acumulada[nombre]["max_abs"], valores["max_abs"])
            del ultimo, loss, clean, noisy
    finally:
        del model.D.observe
    etapas = combinadas
    gx = [e["grad_x_out"]["norma"] for e in etapas]
    if not math.isfinite(perdida) or not all(
        math.isfinite(value) for e in etapas
        for name in ("grad_v", "grad_x_out") for value in e[name].values()
    ):
        raise RuntimeError(f"Salida o gradientes no finitos con S={steps}.")
    return dict(S=steps, T=model.T.item(), lmbda=model.lmbda.item(),
                h=model.T.item()/steps, tau_D=model.lmbda.item()/steps,
                loss=perdida, amplificacion=gx[0]/gx[-1] if gx[-1] > 0 else None,
                etapas=etapas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    capturas = []
    for nombre in ("paso_pl2_peak50_s0_1k", "paso_pp_peak50_s0_1k"):
        local = ROOT / "runs" / nombre / "evento_paso.pth"
        capturas.append(local if local.is_file() else
                        ROOT.parent / "tdv-tuning/checkpoints" / nombre / "evento_paso.pth")
    parser.add_argument("capturas", nargs="*", type=Path, default=capturas)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sin-compilar", action="store_true",
                        help="usa R.grad eager; por defecto se compila en CUDA como en entrenamiento")
    parser.add_argument("--micro-batch", type=int, default=4,
                        help="imágenes simultáneas para limitar la memoria (default: 4)")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sensibilidad_s.json")
    args = parser.parse_args()
    if args.micro_batch < 1:
        parser.error("--micro-batch debe ser positivo.")
    if not all(p.is_file() for p in args.capturas):
        parser.error("Falta alguno de los archivos evento_paso.pth.")
    sys.path.insert(0, str(ROOT))
    from ddr.model import VNet2D

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    compilado = args.device == "cuda" and not args.sin_compilar
    reporte = dict(device=args.device, torch=torch.__version__, compilado=compilado,
                   micro_batch=args.micro_batch,
                   training_repo=str(ROOT),
                   medicion="norma dL/dx1 dividida por norma dL/dxS; dirección de la loss",
                   control="datos previamente aceptados, evaluados con pesos del evento",
                   resultados=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in args.capturas:
        captura = torch.load(path, map_location="cpu", weights_only=True)
        if captura["config"]["S"] != 10:
            raise ValueError(f"{path}: esta comparación requiere una captura original con S=10.")
        model = VNet2D(captura["config"], efficient=False).to(args.device)
        model.load_state_dict(captura["model"], strict=True)
        model.train()
        if compilado:
            model.R.grad = torch.compile(model.R.grad, dynamic=False)
        for tipo in ("evento", "control"):
            muestra = captura[tipo]
            if muestra is None:
                continue
            for steps in (10, 20):
                fila = medir(model, muestra, steps, captura["acumulacion"], args.device, args.micro_batch)
                fila.update(captura=str(path.resolve()), tipo=tipo,
                            it=muestra["it"], micro=muestra["micro"])
                if tipo == "evento" and steps == 10:
                    gx = [e["grad_x_out"]["norma"] for e in fila["etapas"]]
                    fila["reproduce_original"] = (
                        math.isclose(fila["loss"], muestra["loss"], rel_tol=1e-5, abs_tol=1e-7)
                        and len(gx) == len(muestra["grad_x"])
                        and all(math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-7)
                                for a, b in zip(gx, muestra["grad_x"]))
                    )
                    print(f"{path.parent.name}: reproduce_original={fila['reproduce_original']}", flush=True)
                reporte["resultados"].append(fila)
                args.output.write_text(json.dumps(reporte, indent=2, allow_nan=False), encoding="utf-8")
                print(f"{path.parent.name} {tipo} S={steps}: loss={fila['loss']:.6g} "
                      f"amplificación={fila['amplificacion']}", flush=True)
        del model
    print(f"Guardado en {args.output.resolve()}")


if __name__ == "__main__":
    main()
