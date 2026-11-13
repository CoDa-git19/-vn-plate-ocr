"""Fine-tune YOLOv8 (COCO-pretrained) for Vietnamese license plate detection (1 class).

Runs on Google Colab (T4) and on Windows. After training it automatically:
  * prints mAP50, mAP50-95, precision, recall of best.pt on the validation split
  * copies best.pt     -> models/best_<model>.pt          (e.g. models/best_yolov8n.pt)
  * copies results.png -> docs/training_curves_<model>.png
  * saves metrics + total training time -> docs/train_<model>.json

Examples:
    python scripts/train.py                                  # yolov8n, first datasets/*/data.yaml
    python scripts/train.py --model yolov8s.pt --name plate_v1_s

    # Colab: keep checkpoints on Google Drive, resume after a disconnect
    python scripts/train.py --project /content/drive/MyDrive/vn-plate-ocr/runs
    python scripts/train.py --project /content/drive/MyDrive/vn-plate-ocr/runs --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR = REPO_ROOT / "docs"

# Fixed training config; every other hyper-parameter (incl. augmentation) = Ultralytics default.
TRAIN_CFG: dict[str, Any] = {
    "imgsz": 640,
    "pretrained": True,  # transfer learning from COCO weights
    "cos_lr": True,
    "save_period": 5,    # extra checkpoint every 5 epochs (useful on Drive)
    "fliplr": 0.0,       # mirrored plates are not valid plates (see README)
    "flipud": 0.0,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained checkpoint to fine-tune")
    p.add_argument("--name", default="plate_v1_n", help="Run name (folder inside --project)")
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=12, help="Early-stopping patience (epochs)")
    p.add_argument("--data", type=Path, default=None, help="data.yaml (default: first datasets/*/data.yaml)")
    p.add_argument("--project", type=Path, default=REPO_ROOT / "runs",
                   help="Output root; on Colab pass a Google Drive path")
    p.add_argument("--device", default=None, help="'0', 'cpu', ... (default: auto)")
    p.add_argument("--resume", action="store_true", help="Resume <project>/<name>/weights/last.pt")
    p.add_argument("--wandb", action="store_true", help="Enable Ultralytics' W&B logging")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def find_data_yaml(datasets_dir: Path = DATASETS_DIR) -> Path:
    """Return the first ``datasets/*/data.yaml`` (sorted by path)."""
    found = sorted(datasets_dir.glob("*/data.yaml"))
    if not found:
        sys.exit(f"No data.yaml found in {datasets_dir}/*/ - run scripts/prepare_data.py first or pass --data")
    if len(found) > 1:
        print("[data] several datasets found, using the first one (pass --data to choose):")
        for p in found:
            print(f"         {'*' if p == found[0] else ' '} {p}")
    return found[0]


def check_data_yaml(data_yaml: Path) -> None:
    if not data_yaml.is_file():
        sys.exit(f"data.yaml not found: {data_yaml}")
    with data_yaml.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names", [])
    names = list(names.values()) if isinstance(names, dict) else list(names)
    print(f"[data] {data_yaml}  classes={names}")
    if len(names) != 1:
        print(f"[WARN] expected 1 class (plate), data.yaml has {len(names)}")


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def run_training(args: argparse.Namespace, data_yaml: Path | None, project: Path):
    """Train (or resume). Returns ``(trainer, final_val_metrics_or_None)``."""
    from ultralytics import YOLO, settings

    settings.update({"wandb": bool(args.wandb)})

    if args.resume:
        last = project / args.name / "weights" / "last.pt"
        if not last.is_file():
            sys.exit(f"--resume: checkpoint not found: {last}")
        print(f"[train] resuming from {last}")
        model = YOLO(str(last))
        overrides: dict[str, Any] = {"resume": True}
        if args.device is not None:
            overrides["device"] = args.device
        metrics = model.train(**overrides)
    else:
        model = YOLO(args.model)
        metrics = model.train(
            data=str(data_yaml),  # absolute path: Ultralytics resolves dataset root from it
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            project=str(project),
            name=args.name,
            device=args.device,
            **TRAIN_CFG,
        )
    return model.trainer, metrics


def validation_metrics(metrics, best_pt: Path, trainer) -> dict[str, float]:
    """mAP50 / mAP50-95 / P / R of best.pt on the val split.

    Uses the final validation that Ultralytics runs on best.pt after training;
    falls back to an explicit ``val()`` if it is unavailable.
    """
    if metrics is None or not hasattr(metrics, "box"):
        from ultralytics import YOLO

        save_dir = Path(trainer.save_dir)
        metrics = YOLO(str(best_pt)).val(
            data=trainer.args.data, split="val", imgsz=trainer.args.imgsz, batch=trainer.args.batch,
            device=trainer.args.device, project=str(save_dir.parent), name=f"{save_dir.name}_val",
            exist_ok=True, plots=False,
        )
    box = metrics.box
    return {
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }


def parse_results_csv(csv_path: Path) -> dict[str, Any]:
    """Epoch count, best epoch and total train time from Ultralytics' results.csv.

    The ``time`` column restarts from 0 when a run is resumed, so the total is the
    sum of the last value of each session.
    """
    if not csv_path.is_file():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v.strip() for k, v in row.items() if k} for row in csv.DictReader(f)]
    if not rows:
        return {}

    out: dict[str, Any] = {"epochs_completed": len(rows)}
    if "time" in rows[0]:
        total, prev = 0.0, 0.0
        for t in (float(r["time"]) for r in rows):
            if t < prev:  # new session after --resume
                total += prev
            prev = t
        out["train_time_sec"] = round(total + prev, 1)

    k50, k5095 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    if k50 in rows[0] and k5095 in rows[0]:
        # Ultralytics fitness = 0.1 * mAP50 + 0.9 * mAP50-95 (selects best.pt)
        best = max(rows, key=lambda r: 0.1 * float(r[k50]) + 0.9 * float(r[k5095]))
        out["best_epoch"] = int(float(best["epoch"]))
    return out


def fmt_duration(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def env_info() -> dict[str, str]:
    import torch
    import ultralytics

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({platform.processor() or 'n/a'})"
    return {
        "device": device,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project = args.project.expanduser().resolve()

    data_yaml = None
    if not args.resume:  # on resume the dataset path comes from the checkpoint
        data_yaml = (args.data or find_data_yaml()).expanduser().resolve()
        check_data_yaml(data_yaml)

    t0 = time.perf_counter()
    trainer, final_metrics = run_training(args, data_yaml, project)
    wall_sec = time.perf_counter() - t0

    save_dir = Path(trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        sys.exit(f"best.pt not found in {best_pt.parent} - training did not finish")

    metrics = validation_metrics(final_metrics, best_pt, trainer)
    csv_info = parse_results_csv(save_dir / "results.csv")
    targs = trainer.args
    train_sec = csv_info.get("train_time_sec", wall_sec)

    # ---- artifacts ----
    tag = Path(args.model).stem  # yolov8n.pt -> yolov8n
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    model_out = MODELS_DIR / f"best_{tag}.pt"
    curves_out = DOCS_DIR / f"training_curves_{tag}.png"
    json_out = DOCS_DIR / f"train_{tag}.json"

    shutil.copy2(best_pt, model_out)
    results_png = save_dir / "results.png"
    if results_png.is_file():
        shutil.copy2(results_png, curves_out)
    else:
        print(f"[WARN] {results_png} not found, training curves not copied")
        curves_out = None

    epochs_requested = int(targs.epochs)
    epochs_done = csv_info.get("epochs_completed")
    summary = {
        "model": args.model,
        "run_dir": save_dir.as_posix(),
        "data": Path(targs.data).as_posix(),
        "resumed": args.resume,
        "metrics_val": metrics,
        "epochs_requested": epochs_requested,
        "epochs_completed": epochs_done,
        "best_epoch": csv_info.get("best_epoch"),
        "early_stopped": epochs_done is not None and epochs_done < epochs_requested,
        "train_time_sec": round(train_sec, 1),
        "train_time": fmt_duration(train_sec),
        "wall_time_this_session_sec": round(wall_sec, 1),
        "hyperparams": {
            "epochs": epochs_requested,
            "batch": targs.batch,
            "patience": targs.patience,
            **{k: getattr(targs, k) for k in TRAIN_CFG},
        },
        "env": env_info(),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- report ----
    print("\n=== Validation metrics (best.pt, split=val) ===")
    for k, v in metrics.items():
        print(f"  {k:<10}: {v:.4f}")
    stop = " (early stopped)" if summary["early_stopped"] else ""
    print(f"  epochs    : {epochs_done}/{epochs_requested}, best epoch {summary['best_epoch']}{stop}")
    print(f"  train time: {summary['train_time']}  on {summary['env']['device']}")
    print("\n=== Saved ===")
    print(f"  {model_out}")
    if curves_out:
        print(f"  {curves_out}")
    print(f"  {json_out}")
    print(f"  run dir: {save_dir}")
    return 0


if __name__ == "__main__":  # guard required on Windows (DataLoader workers use spawn)
    sys.exit(main())
