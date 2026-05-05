"""
ablation.py - Structured ablation study for Task 1 CNN (Regression)

Strategy: targeted one-factor-at-a-time ablation rather than a full grid.
Runs ~25 carefully chosen configurations (~3 hrs on Kaggle GPU) and produces
a clean, report-ready table.

Background from initial full-grid run (first 64 configs):
  - Optimizer is the dominant factor: SGD MAE ~3-6 vs Adam MAE ~50-70.
  - All subsequent groups fix optimizer=SGD and vary one factor at a time.

Ablation groups
---------------
0  Optimizer            — Adam vs SGD on Model A spec (2 configs, ~14min)
1  Filter / depth       — vary filter progression & depth (5 configs, ~35min)
2  Residual connections — True vs False (2 configs, ~14min)
3  Dropout              — 0.0 / 0.2 / 0.3 / 0.5 (4 configs, ~28min)
4  Kernel size          — 3 / 5 / 7 (3 configs, ~21min)
5  Model A final        — official assignment spec, no residual (1 config)
6  Model B final        — deeper, regularised, residual (2 configs)

Usage
-----
    python ablation.py [--kaggle] [--test-run] [--groups 1,2,3]

    --groups: comma-separated list of group IDs to run (default: all 1-6).
              Group 0 is from prior data; not re-run here.
"""

import os
import csv
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from data   import get_dataloaders
from models import AblationModel


# ---------------------------------------------------------------------------
# Ablation groups
# ---------------------------------------------------------------------------

# Fixed values that are held constant once established:
#   optimizer = 'sgd'  (dominant finding from group 0)
#   lr        = 0.001  (matches SGD best configs in initial run)
#   momentum  = 0.9
#   activation = 'relu'   (relu ~= leaky_relu; keep simple)
#   weight_decay = 0.0    (small effect; keep 0 so dropout is the only reg)

ABLATION_GROUPS = {

    # ------------------------------------------------------------------
    # Group 1: Filter progression & depth
    #   Isolates: how many channels and how many blocks matter most?
    #   Fix: SGD, k=3, dr=0.0, relu, residual=False, wd=0
    # ------------------------------------------------------------------
    1: {
        "description": "Filter / depth ablation",
        "fixed": dict(kernel_size=3, dropout_rate=0.0, optimizer="sgd",
                      use_residual=False, activation="relu",
                      learning_rate=0.001, weight_decay=0.0),
        "vary": [
            {"filters": [16, 32, 64]},            # 3-block shallow
            {"filters": [32, 64, 128]},            # 3-block baseline (Model A spec)
            {"filters": [64, 128, 256]},           # 3-block wide
            {"filters": [16, 32, 64, 128]},        # 4-block slim
            {"filters": [32, 64, 128, 256]},       # 4-block standard
        ],
    },

    # ------------------------------------------------------------------
    # Group 2: Residual connections
    #   Isolates: do skip-connections help on this task?
    #   Fix: SGD, k=3, dr=0.0, relu, [32,64,128], wd=0
    # ------------------------------------------------------------------
    2: {
        "description": "Residual connections ablation",
        "fixed": dict(filters=[32, 64, 128], kernel_size=3, dropout_rate=0.0,
                      optimizer="sgd", activation="relu",
                      learning_rate=0.001, weight_decay=0.0),
        "vary": [
            {"use_residual": False},
            {"use_residual": True},
        ],
    },

    # ------------------------------------------------------------------
    # Group 3: Dropout rate
    #   Isolates: how much regularisation is needed?
    #   Fix: SGD, k=3, relu, [32,64,128], residual=True (if group 2 says yes)
    # ------------------------------------------------------------------
    3: {
        "description": "Dropout ablation",
        "fixed": dict(filters=[32, 64, 128], kernel_size=3,
                      optimizer="sgd", use_residual=True, activation="relu",
                      learning_rate=0.001, weight_decay=0.0),
        "vary": [
            {"dropout_rate": 0.0},
            {"dropout_rate": 0.2},
            {"dropout_rate": 0.3},
            {"dropout_rate": 0.5},
        ],
    },

    # ------------------------------------------------------------------
    # Group 4: Kernel size
    #   Isolates: receptive field size
    #   Fix: SGD, best dropout (0.3), relu, [32,64,128], residual=True
    # ------------------------------------------------------------------
    4: {
        "description": "Kernel size ablation",
        "fixed": dict(filters=[32, 64, 128], dropout_rate=0.3,
                      optimizer="sgd", use_residual=True, activation="relu",
                      learning_rate=0.001, weight_decay=0.0),
        "vary": [
            {"kernel_size": 3},
            {"kernel_size": 5},
            {"kernel_size": 7},
        ],
    },

    # ------------------------------------------------------------------
    # Group 5: Model A — official assignment baseline
    #   Exactly spec: 3 conv blocks [32,64,128], k=3, no residual, SGD
    #   (No dropout on Model A — keep it a true baseline)
    # ------------------------------------------------------------------
    5: {
        "description": "Model A — official baseline",
        "fixed": dict(kernel_size=3, dropout_rate=0.0, optimizer="sgd",
                      use_residual=False, activation="relu",
                      learning_rate=0.001, weight_decay=0.0),
        "vary": [
            {"filters": [32, 64, 128]},
        ],
    },

    # ------------------------------------------------------------------
    # Group 6: Model B — deeper, regularised, residual
    #   Uses best settings from groups 1-4; must beat Model A.
    #   At least 4 conv blocks + dropout (0.2-0.5) + L2 weight decay.
    # ------------------------------------------------------------------
    6: {
        "description": "Model B — deeper / regularised",
        "fixed": dict(kernel_size=3, dropout_rate=0.3, optimizer="sgd",
                      use_residual=True, activation="relu",
                      learning_rate=0.001, weight_decay=1e-4),
        "vary": [
            {"filters": [32, 64, 128, 256]},   # 4 blocks
            {"filters": [64, 128, 256, 512]},  # 4 blocks wider
        ],
    },

    # ------------------------------------------------------------------
    # Group 0 — Optimizer comparison
    #   Adam vs SGD head-to-head on the Model A spec.
    #   This is the most important single finding: SGD wins by ~20× on MAE.
    # ------------------------------------------------------------------
    0: {
        "description": "Optimizer comparison (Adam vs SGD)",
        "fixed": dict(filters=[32, 64, 128], kernel_size=3, dropout_rate=0.0,
                      use_residual=False, activation="relu",
                      learning_rate=0.001, weight_decay=0.0),
        "vary": [
            {"optimizer": "adam"},
            {"optimizer": "sgd"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_config(params: dict, train_loader, val_loader, epochs: int,
                 device: torch.device) -> dict:
    """Train one configuration and return metrics."""
    model = AblationModel(
        filters      = params["filters"],
        kernel_size  = params["kernel_size"],
        dropout_rate = params["dropout_rate"],
        use_residual = params["use_residual"],
        activation   = params["activation"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.MSELoss()
    if params["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=params["learning_rate"],
            momentum=0.9,
            weight_decay=params["weight_decay"],
        )

    best_val_mae = float("inf")
    best_epoch   = 0
    patience     = 10
    no_improve   = 0
    final_train_mae = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        train_mae_sum, train_n = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_mae_sum += torch.abs(outputs.detach() - labels).sum().item()
            train_n       += labels.size(0)
        train_mae = train_mae_sum / train_n

        # ---- validate ----
        model.eval()
        val_mae_sum, val_n = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_mae_sum += torch.abs(outputs - labels).sum().item()
                val_n       += labels.size(0)
        val_mae = val_mae_sum / val_n

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch + 1
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch + 1}")
                break

        if epoch == epochs - 1:
            final_train_mae = train_mae

    elapsed = time.time() - t0
    return {
        "best_val_mae":    round(best_val_mae, 4),
        "best_epoch":      best_epoch,
        "final_train_mae": round(final_train_mae, 4),
        "n_params":        n_params,
        "train_time_s":    round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Targeted ablation study")
    parser.add_argument("--kaggle",   action="store_true")
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument(
        "--groups",
        default="0,1,2,3,4,5,6",
        help="Comma-separated group IDs to run (e.g. '1,2,5,6'). Default: all.",
    )
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.kaggle:
        cfg["data"]["image_dir"]   = "/kaggle/input/datasets/abdullahahmedani/seeds-data/filtered"
        cfg["data"]["labels_path"] = "/kaggle/input/datasets/abdullahahmedani/seeds-data/labeled_components.pkl"

    epochs = 2 if args.test_run else cfg["training"]["epochs"]
    groups_to_run = [int(g.strip()) for g in args.groups.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {epochs} | Groups: {groups_to_run}")

    torch.manual_seed(cfg["seed"])
    train_loader, val_loader = get_dataloaders(cfg)
    if args.test_run:
        train_loader.dataset.subset.indices = train_loader.dataset.subset.indices[:16]
        val_loader.dataset.subset.indices   = val_loader.dataset.subset.indices[:16]

    os.makedirs("cnn_outputs", exist_ok=True)
    log_path = "cnn_outputs/ablation_results.csv"

    # ---- build experiment list ----
    experiments = []
    for gid in groups_to_run:
        group = ABLATION_GROUPS[gid]
        for vary_dict in group["vary"]:
            params = {**group["fixed"], **vary_dict}
            experiments.append({
                "group_id":    gid,
                "group_name":  group["description"],
                "varied_key":  list(vary_dict.keys())[0],
                "varied_value": list(vary_dict.values())[0],
                **params,
            })

    total = len(experiments)
    print(f"\nTotal experiments to run: {total}\n")

    # ---- CSV header ----
    fieldnames = [
        "run", "group_id", "group_name", "varied_key", "varied_value",
        "filters", "kernel_size", "dropout_rate", "optimizer",
        "use_residual", "activation", "learning_rate", "weight_decay",
        "best_val_mae", "best_epoch", "final_train_mae", "n_params", "train_time_s",
    ]
    # append to existing file if it exists so incremental saves work
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    writer   = csv.DictWriter(log_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    best_overall = float("inf")
    best_ckpt    = None

    for idx, exp in enumerate(experiments):
        # extract pure training params (exclude metadata keys)
        params = {k: exp[k] for k in [
            "filters", "kernel_size", "dropout_rate", "optimizer",
            "use_residual", "activation", "learning_rate", "weight_decay",
        ]}

        print(f"[{idx+1}/{total}] Group {exp['group_id']} — {exp['group_name']}")
        print(f"  vary {exp['varied_key']} = {exp['varied_value']}")
        print(f"  full config: {params}")

        metrics = train_config(params, train_loader, val_loader, epochs, device)

        row = {
            "run":           idx + 1,
            "group_id":      exp["group_id"],
            "group_name":    exp["group_name"],
            "varied_key":    exp["varied_key"],
            "varied_value":  str(exp["varied_value"]),
            **params,
            "filters":       str(params["filters"]),
            **metrics,
        }
        writer.writerow(row)
        log_file.flush()

        print(f"  → Val MAE: {metrics['best_val_mae']:.4f} at epoch {metrics['best_epoch']}"
              f" | params: {metrics['n_params']:,} | {metrics['train_time_s']}s")

        # save best model
        if metrics["best_val_mae"] < best_overall:
            best_overall = metrics["best_val_mae"]
            best_ckpt    = {"params": params, **metrics}
            # rebuild and re-save
            m = AblationModel(
                filters      = params["filters"],
                kernel_size  = params["kernel_size"],
                dropout_rate = params["dropout_rate"],
                use_residual = params["use_residual"],
                activation   = params["activation"],
            ).to(device)
            # re-train briefly just to get weights (we can't save during the closure)
            # Instead, track and save during training — done inline above
            # For now save the config; full weight saving done in train.py for Models A/B
            import json
            with open("cnn_outputs/best_ablation_config.json", "w") as jf:
                json.dump(best_ckpt, jf, indent=2)
            print(f"  ✓ New best overall (MAE {best_overall:.4f})")

        print()

    log_file.close()

    print("=" * 60)
    print(f"Ablation complete. Results saved to {log_path}")
    print(f"Best config: MAE {best_overall:.4f}")
    if best_ckpt:
        print(f"  {best_ckpt['params']}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check ablation_results.csv to confirm best settings per group.")
    print("  2. Run train.py to train the final Model A and Model B with full epochs.")


if __name__ == "__main__":
    main()
