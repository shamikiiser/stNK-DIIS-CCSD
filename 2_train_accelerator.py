"""
Script 2: ML Trainer for CCSD Convergence Acceleration -- Scalar Step Ratio
----------------------------------------------------------------------------

    y = StepLogRatio at the window's last iteration
      = log10(best_s / jacobi_norm)

found by 1_generate_data.py's oracle line search along the unit Newton
direction. 
Loss: plain MSE on the scalar target.
__authors__ = ["Shamik Chanda", "Suhita Basumalick", "Sangita Sen"]
__license__ = "BSD-3-Clause"
__date__    = "2026"
"""

import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn

from cc_ml_common import (
    WINDOW,
    EPS_LOG,
    MAX_DIIS_NORM,
    N_SCALAR_FEATURES,
    FEATURE_NAMES,
    MAX_LOG_STEP_RATIO,
    STALL_OVERSAMPLE_WEIGHT,
    CCStepPredictor,
)


#######################################################################
# CONFIGURATION CONSISTENCY CHECK
#######################################################################

if N_SCALAR_FEATURES != 19:
    raise RuntimeError(
        "\nConfiguration error in cc_ml_common.py:\n"
        f"Expected N_SCALAR_FEATURES = 19, but got {N_SCALAR_FEATURES}.\n\n"
        "This trainer expects the 19-feature scalar-step-ratio configuration."
    )

if len(FEATURE_NAMES) != N_SCALAR_FEATURES:
    raise RuntimeError(
        "\nFeature-name configuration error in cc_ml_common.py:\n"
        f"N_SCALAR_FEATURES = {N_SCALAR_FEATURES}, "
        f"but FEATURE_NAMES contains {len(FEATURE_NAMES)} entries."
    )


TARGET_TYPE = "log10_step_ratio_along_newton_direction"


#######################################################################
# NUMERICAL HELPERS
#######################################################################

def safe_log10_abs(x, floor=EPS_LOG):
    return np.log10(np.maximum(np.abs(x), floor))


def safe_log10_positive(x, floor=1.0e-12):
    return np.log10(np.maximum(np.asarray(x), floor))


#######################################################################
# LOAD ONE MOLECULAR TRAJECTORY (now cheap and eager -- see module docstring)
#######################################################################

def load_system(npz_path, window=WINDOW):

    data = np.load(npz_path)

    required = [
        "Rnorm", "E_prev1", "E_prev2", "R_prev1", "R_prev2", "R_prev3",
        "Ratio1", "Ratio2", "R_std", "R_max", "T_std", "T_max",
        "JacobiNorm", "DIISErrorNorm", "DIISStepNorm", "DIISCorrectionNorm",
        "DIISSize", "IsStall", "StepLogRatio", "LogDMin", "LogDMax",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(
            f"{os.path.basename(npz_path)} is missing required data: "
            f"{missing}\n\nRegenerate the dataset using the current "
            f"1_generate_data.py (this file may predate the scalar "
            f"step-ratio redesign)."
        )

    Rnorm = data["Rnorm"]
    n_iter = Rnorm.shape[0]

    if n_iter < window:
        return None

    is_stall = data["IsStall"].astype(bool)
    step_log_ratio = np.clip(
        data["StepLogRatio"].astype(np.float32), -MAX_LOG_STEP_RATIO, MAX_LOG_STEP_RATIO
    )

    log_D_min = float(data["LogDMin"])
    log_D_max = float(data["LogDMax"])

    logR = safe_log10_abs(Rnorm).astype(np.float32)
    logR_prev1 = safe_log10_abs(data["R_prev1"]).astype(np.float32)
    logR_prev2 = safe_log10_abs(data["R_prev2"]).astype(np.float32)
    logR_prev3 = safe_log10_abs(data["R_prev3"]).astype(np.float32)
    log_ratio1 = safe_log10_positive(data["Ratio1"]).astype(np.float32)
    log_ratio2 = safe_log10_positive(data["Ratio2"]).astype(np.float32)
    logE_prev1 = safe_log10_abs(data["E_prev1"]).astype(np.float32)
    logE_prev2 = safe_log10_abs(data["E_prev2"]).astype(np.float32)
    log_jacobi_norm = safe_log10_positive(data["JacobiNorm"]).astype(np.float32)
    log_diis_step_norm = safe_log10_positive(data["DIISStepNorm"]).astype(np.float32)
    log_diis_correction_norm = safe_log10_positive(data["DIISCorrectionNorm"]).astype(np.float32)
    log_diis_error_norm = safe_log10_positive(data["DIISErrorNorm"]).astype(np.float32)

    diis_size_norm = (data["DIISSize"].astype(np.float32) / MAX_DIIS_NORM)

    log_D_min_arr = np.full(n_iter, log_D_min, dtype=np.float32)
    log_D_max_arr = np.full(n_iter, log_D_max, dtype=np.float32)

    scalars = np.stack(
        [
            logR,
            diis_size_norm,
            data["R_std"].astype(np.float32),
            data["R_max"].astype(np.float32),
            data["T_std"].astype(np.float32),
            data["T_max"].astype(np.float32),
            logR_prev1,
            logR_prev2,
            logR_prev3,
            log_ratio1,
            log_ratio2,
            logE_prev1,
            logE_prev2,
            log_jacobi_norm,
            log_diis_step_norm,
            log_diis_correction_norm,
            log_diis_error_norm,
            log_D_min_arr,
            log_D_max_arr,
        ],
        axis=-1
    ).astype(np.float32)

    if scalars.shape[1] != N_SCALAR_FEATURES:
        raise RuntimeError(
            f"Feature dimension mismatch in {os.path.basename(npz_path)}: "
            f"got {scalars.shape[1]}, expected {N_SCALAR_FEATURES}."
        )

    n_windows = n_iter - window + 1
    is_stall_windowed = is_stall[window - 1:]
    target_windowed = step_log_ratio[window - 1:]

    # Fast harvesting computes the expensive oracle only at genuine stalls.
    # Therefore only finite, stall-aligned windows are valid training samples.
    finite = np.isfinite(target_windowed) & is_stall_windowed
    if not np.any(finite):
        return None
    valid_idx = np.flatnonzero(finite).astype(np.int32)
    target_windowed = target_windowed[valid_idx]
    window_weights = np.full(valid_idx.size, 1.0 / valid_idx.size, dtype=np.float64)

    return dict(
        scalars=scalars,               # (n_iter, 19)
        target=target_windowed,
        valid_idx=valid_idx,
        n_windows=valid_idx.size,
        window_weights=window_weights,
        n_stall_windows=valid_idx.size,
        name=os.path.basename(npz_path),
    )


#######################################################################
# TRAJECTORY DATASET (eager loading -- data is small now)
#######################################################################

class TrajectoryDataset:

    def __init__(self, data_dir, window=WINDOW, val_frac=0.15, seed=0):

        self.window = window

        paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if len(paths) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir}")

        # The scalar oracle target is available only at genuine DIIS stalls.
        # Therefore filter out systems without finite stall targets BEFORE
        # making the train/validation split.  Splitting first can randomly
        # place all usable systems in training and leave validation empty.
        rng = np.random.default_rng(seed)

        usable = {}
        skipped = []

        for p in paths:
            try:
                s = load_system(p, window=window)
                if s is None:
                    skipped.append(os.path.basename(p))
                else:
                    usable[s["name"]] = s
            except Exception as e:
                print(f"WARNING: could not load {os.path.basename(p)}: {e}")
                skipped.append(os.path.basename(p))

        if len(usable) < 2:
            raise RuntimeError(
                "\nNot enough usable systems for a system-level train/validation split.\n"
                f"Total .npz files found = {len(paths)}\n"
                f"Usable stall-containing systems = {len(usable)}\n"
                f"Skipped systems = {len(skipped)}\n\n"
                "The current scalar-step-ratio trainer requires at least "
                "two systems containing finite StepLogRatio values at "
                "genuine DIIS stalls. Regenerate the dataset with enough "
                "genuine stalls if this number is below two."
            )

        usable_names = list(usable.keys())
        rng.shuffle(usable_names)

        # Split only among systems that actually contain supervised targets.
        n_val = max(1, int(round(len(usable_names) * val_frac)))
        n_val = min(n_val, len(usable_names) - 1)

        val_names = usable_names[:n_val]
        train_names = usable_names[n_val:]

        self.train_systems = {name: usable[name] for name in train_names}
        self.val_systems = {name: usable[name] for name in val_names}

        if len(self.train_systems) == 0:
            raise RuntimeError("No usable training systems were loaded.")

        if len(self.val_systems) == 0:
            raise RuntimeError("No usable validation systems were loaded.")

        total_train_windows = sum(
            s["n_windows"] for s in self.train_systems.values()
        )
        total_train_stall_windows = sum(
            s["n_stall_windows"] for s in self.train_systems.values()
        )
        total_val_windows = sum(
            s["n_windows"] for s in self.val_systems.values()
        )
        total_val_stall_windows = sum(
            s["n_stall_windows"] for s in self.val_systems.values()
        )

        print("\nTrajectory dataset:")
        print(f"  Total .npz files     = {len(paths)}")
        print(f"  Usable systems       = {len(usable)}")
        print(f"  Skipped systems      = {len(skipped)}")
        print(f"  Training systems     = {len(self.train_systems)}")
        print(f"  Validation systems   = {len(self.val_systems)}")
        print(f"  Scalar features      = {N_SCALAR_FEATURES}")
        print(
            f"  Training windows     = {total_train_windows} "
            f"({total_train_stall_windows} genuine stalls)"
        )
        print(
            f"  Validation windows   = {total_val_windows} "
            f"({total_val_stall_windows} genuine stalls)"
        )
        print(f"  Stall oversampling   = {STALL_OVERSAMPLE_WEIGHT}x")

        if skipped:
            print("\nSkipped systems:")
            for name in skipped:
                print(f"  {name}")

        print("\nTraining systems:")
        for name, s in self.train_systems.items():
            print(f"  {name:30s} {s['n_stall_windows']:4d} stall windows")

        print("\nValidation systems:")
        for name, s in self.val_systems.items():
            print(f"  {name:30s} {s['n_stall_windows']:4d} stall windows")

        print("\nScalar feature vector:")
        for i, name in enumerate(FEATURE_NAMES):
            print(f"  [{i:2d}] {name}")

    def _load_all(self, paths, window):
        systems = {}
        for p in paths:
            s = load_system(p, window=window)
            if s is not None:
                systems[s["name"]] = s
        return systems

    def sample_batch(self, rng, batch_size, systems=None):
        """
        Draw `batch_size` independent (system, window) samples: pick a
        system uniformly, then a window within it according to that
        system's stall-oversampling weights, repeated batch_size times.
        """
        sys_dict = systems if systems is not None else self.train_systems
        if len(sys_dict) == 0:
            raise RuntimeError("Cannot sample from an empty trajectory set.")

        names = list(sys_dict.keys())
        sys_choices = rng.integers(0, len(names), size=batch_size)
        scalar_batch = np.empty(
            (batch_size, self.window, N_SCALAR_FEATURES), dtype=np.float32
        )
        target_batch = np.empty((batch_size,), dtype=np.float32)

        for b, sid in enumerate(sys_choices):
            s = sys_dict[names[int(sid)]]
            pos = int(rng.choice(s["n_windows"], p=s["window_weights"]))
            w_idx = int(s["valid_idx"][pos])
            scalar_batch[b] = s["scalars"][w_idx:w_idx + self.window, :]
            target_batch[b] = s["target"][pos]

        return scalar_batch, target_batch



#######################################################################
# VALIDATION
#######################################################################

def evaluate(model, dataset, rng, device, batch_size=512):
    model.eval()
    pred_all, tgt_all = [], []
    with torch.no_grad():
        for sys in dataset.val_systems.values():
            for start in range(0, sys["n_windows"], batch_size):
                pos = np.arange(start, min(start + batch_size, sys["n_windows"]))
                idx = sys["valid_idx"][pos]
                sc = np.stack([sys["scalars"][i:i + dataset.window] for i in idx]).astype(np.float32)
                tgt = sys["target"][pos].astype(np.float32)
                pred_all.append(model(torch.from_numpy(sc).to(device)).cpu())
                tgt_all.append(torch.from_numpy(tgt))
    model.train()
    if not pred_all:
        return float("inf"), float("inf")
    pred = torch.cat(pred_all)
    tgt = torch.cat(tgt_all)
    return (nn.functional.mse_loss(pred, tgt).item(),
            nn.functional.l1_loss(pred, tgt).item())


#######################################################################
# MAIN TRAINING FUNCTION
#######################################################################

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="cc_training_data")
    parser.add_argument("--out", default="cc_accelerator.pt")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    dataset = TrajectoryDataset(
        args.data_dir, window=WINDOW, val_frac=args.val_frac, seed=args.seed
    )

    model = CCStepPredictor(
        hidden=args.hidden,
        scalar_dim=N_SCALAR_FEATURES,
        max_log_step_ratio=MAX_LOG_STEP_RATIO,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    best_state = None
    bad_evals = 0

    for step in range(1, args.steps + 1):

        model.train()
        opt.zero_grad()

        sc, tgt = dataset.sample_batch(rng, args.batch_size)
        sc_t = torch.from_numpy(sc).to(device)
        tgt_t = torch.from_numpy(tgt).to(device)

        pred = model(sc_t)
        loss = nn.functional.mse_loss(pred, tgt_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            val_mse, val_mae = evaluate(model, dataset, rng, device)
            sched.step(val_mse)
            current_lr = opt.param_groups[0]["lr"]

            print(
                f"step {step:6d}  train_mse={loss.item():.3e}  "
                f"val_mse={val_mse:.3e}  val_mae={val_mae:.3e}  lr={current_lr:.3e}"
            )

            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_evals = 0
                print(f"          [Best] validation MSE = {best_val:.3e}")
            else:
                bad_evals += 1
                if bad_evals >= 15:
                    print("\nEarly stopping: validation loss has not improved.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state": model.state_dict(),
        "hidden": args.hidden,
        "window": WINDOW,
        "scalar_dim": N_SCALAR_FEATURES,
        "feature_names": FEATURE_NAMES,
        "target_type": TARGET_TYPE,
        "max_log_step_ratio": MAX_LOG_STEP_RATIO,
        "stall_oversample_weight": STALL_OVERSAMPLE_WEIGHT,
        "best_val_mse": best_val,
    }

    torch.save(checkpoint, args.out)

    print("\n==========================================")
    print("Training complete")
    print("==========================================")
    print(f"Model saved       = {args.out}")
    print(f"Best validation   = {best_val:.6e}")
    print(f"Target type       = {TARGET_TYPE}")
    print(f"MAX_LOG_STEP_RATIO= {MAX_LOG_STEP_RATIO}")
    print(f"Hidden dimension  = {args.hidden}")
    print(f"Window            = {WINDOW}")
    print(f"Scalar features   = {N_SCALAR_FEATURES}")

    print("\nFeatures:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {i:2d}: {name}")


if __name__ == "__main__":
    main()
