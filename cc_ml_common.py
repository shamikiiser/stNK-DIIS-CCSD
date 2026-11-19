"""
Shared model architecture, feature definitions, and utilities for
ML-accelerated Coupled-Cluster Singles and Doubles (stNKML-DIIS).

Imported by:
    - 1_generate_data.py
    - 2_train_accelerator.py
    - ccsd_master.py (method: st_nkml_diis)

Model overview:
    CCStepPredictor takes a sliding window of 19 system- and basis-agnostic
    scalar convergence diagnostics (shape: [batch, WINDOW, 19]) through a
    lightweight GRU + MLP head and predicts a bounded scalar log step ratio:

        y = log10(step_length / jacobi_step_norm)

    At runtime, the physical amplitude step is applied along the unit
    GMRES Newton-Krylov direction v_dom:

        dT_ml = (jacobi_norm * 10**y) * v_dom
"""

__authors__ = ["Shamik Chanda", "Suhita Basumalick", "Sangita Sen"]
__license__ = "BSD-3-Clause"
__date__    = "2026"

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

WINDOW = 5

EPS_D = 1e-3

EPS_LOG = 1e-12

MAX_DIIS_NORM = 8.0

N_SCALAR_FEATURES = 19

TARGET_TYPE = "log10_step_ratio_along_newton_direction"

# Training target and model output are both clipped/bounded to this
# range: log10(step_length / jacobi_step_norm) in [-MAX_LOG_STEP_RATIO,
# +MAX_LOG_STEP_RATIO], i.e. a step between 1e-3x and 1e3x the natural
# Jacobi step scale. Generous enough to cover real stall-rescue steps
# (live_nk_rescue's old grid went up to 2x jacobi_norm; oracle line
# searches during harvesting go further to find the true optimum) while
# still bounding the model's output to something physically sane.
MAX_LOG_STEP_RATIO = 3.0


# ============================================================================
# STALL DETECTION (shared by harvester and live driver)
# ============================================================================

STALL_COUNT = 3
STALL_RATIO = 0.98

# Relative sampling weight given to a training window whose target
# iteration was a genuine stall, versus an ordinary window (weight 1.0).
STALL_OVERSAMPLE_WEIGHT = 6.0
ORACLE_COARSE_POINTS = 11


def is_stalled(recent_Rnorm, stall_count=STALL_COUNT, stall_ratio=STALL_RATIO):
    """
    True if the most recent `stall_count` consecutive residual-norm ratios
    are all > stall_ratio (i.e. the residual is not meaningfully improving).

    recent_Rnorm: residual norms in chronological order, ending with the
    CURRENT iteration's residual norm as the last element.
    """
    recent = list(recent_Rnorm)
    if len(recent) < stall_count + 1:
        return False
    recent = recent[-(stall_count + 1):]
    ratios = [
        recent[i + 1] / max(recent[i], 1.0e-300)
        for i in range(len(recent) - 1)
    ]
    return all(r > stall_ratio for r in ratios)


# ============================================================================
# SCALAR FEATURE NAMES
# ============================================================================
#
# IMPORTANT: order here MUST remain identical to the order used in
# 1_generate_data.py, 2_train_accelerator.py, and 3_run_ml_diis_nk.py.
#
#   0  logR
#   1  diis_size (normalized by MAX_DIIS_NORM)
#   2  R_std        (std of R/D, computed once at harvest/inference time)
#   3  R_max        (max|R/D|)
#   4  T_std        (std of T)
#   5  T_max        (max|T|)
#   6  logR_prev1
#   7  logR_prev2
#   8  logR_prev3
#   9  log_ratio1
#  10  log_ratio2
#  11  logE_prev1
#  12  logE_prev2
#  13  log_jacobi_norm
#  14  log_diis_step_norm
#  15  log_diis_correction_norm
#  16  log_diis_error_norm
#  17  log_D_min   (NEW -- per-system constant, see module docstring)
#  18  log_D_max   (NEW -- per-system constant, see module docstring)
#
# ============================================================================

FEATURE_NAMES = [
    "logR",
    "diis_size",
    "R_std",
    "R_max",
    "T_std",
    "T_max",
    "logR_prev1",
    "logR_prev2",
    "logR_prev3",
    "log_ratio1",
    "log_ratio2",
    "logE_prev1",
    "logE_prev2",
    "log_jacobi_norm",
    "log_diis_step_norm",
    "log_diis_correction_norm",
    "log_diis_error_norm",
    "log_D_min",
    "log_D_max",
]


if len(FEATURE_NAMES) != N_SCALAR_FEATURES:
    raise RuntimeError(
        "Internal feature configuration error: "
        f"N_SCALAR_FEATURES={N_SCALAR_FEATURES}, "
        f"but FEATURE_NAMES contains {len(FEATURE_NAMES)} features."
    )


# ============================================================================
# SAFE DENOMINATORS / D-SPREAD FEATURES
# ============================================================================

def safe_D(D):
    """Floor |D| away from zero without changing its sign."""
    D = np.asarray(D)
    sign = np.where(D >= 0, 1.0, -1.0)
    return sign * np.maximum(np.abs(D), EPS_D)


def packed_residual(cc, rhs1, rhs2):
    """Pack the nonlinear CCSD residual from already-computed RHS arrays."""
    R1 = rhs1 - cc.Dia * cc.t1
    R2 = rhs2 - cc.Dijab * cc.t2
    R = np.concatenate([R1.ravel(), R2.ravel()])
    return R, float(np.linalg.norm(R))


def d_spread_features(D, floor=1.0e-6):
    """
    Per-system, per-CONSTANT features summarizing how much the Fock
    denominators span in magnitude -- this is what determines how much
    a naive per-amplitude reconstruction would be amplified (see module
    docstring). Returns (log_D_min, log_D_max) as plain floats.
    """
    absD = np.maximum(np.abs(np.asarray(D)), floor)
    return float(np.log10(absD.min())), float(np.log10(absD.max()))


# ============================================================================
# MODEL
# ============================================================================

class CCStepPredictor(nn.Module):
    """
    Predicts a single scalar: log10(step_length / jacobi_step_norm), for
    stepping along the unit Newton direction v_dom = cc.newton_direction().

    Input
    -----
    scalar_w : (B, WINDOW, scalar_dim)
        Temporal window of the 19 global scalar features (see
        FEATURE_NAMES). B is a batch of independent (system, window)
        samples -- unlike the old per-amplitude model, B has no
        required relationship to n_amp; ordinary minibatching applies.

    Output
    ------
    (B,) tensor, each entry in [-max_log_step_ratio, +max_log_step_ratio]
    by construction (tanh-bounded), representing the predicted
    log10(step / jacobi_norm) for that sample.
    """

    def __init__(
        self,
        hidden=32,
        scalar_dim=N_SCALAR_FEATURES,
        n_layers=1,
        max_log_step_ratio=MAX_LOG_STEP_RATIO,
    ):
        super().__init__()
        self.hidden = hidden
        self.scalar_dim = scalar_dim
        self.n_layers = n_layers
        self.max_log_step_ratio = float(max_log_step_ratio)

        if scalar_dim <= 0:
            raise ValueError(f"scalar_dim must be positive, got {scalar_dim}")
        if self.max_log_step_ratio <= 0.0:
            raise ValueError(
                f"max_log_step_ratio must be positive, got {self.max_log_step_ratio}"
            )

        self.scalar_gru = nn.GRU(
            input_size=scalar_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, scalar_w):
        if scalar_w.ndim != 3:
            raise ValueError(
                f"scalar_w must have shape (B, WINDOW, scalar_dim), "
                f"got {tuple(scalar_w.shape)}"
            )
        if scalar_w.shape[1] != WINDOW:
            raise ValueError(
                f"scalar_w has window dimension {scalar_w.shape[1]}, "
                f"but WINDOW={WINDOW}"
            )
        if scalar_w.shape[2] != self.scalar_dim:
            raise ValueError(
                f"Expected {self.scalar_dim} scalar features, "
                f"got {scalar_w.shape[2]}"
            )

        _, h_n = self.scalar_gru(scalar_w)
        h = h_n[-1]                                   # (B, hidden)
        raw = self.head(h).squeeze(-1)                 # (B,) -- UNBOUNDED

        # Hard, architectural bound -- guaranteed for any input.
        return self.max_log_step_ratio * torch.tanh(raw)


# ============================================================================
# CHECKPOINT LOADER
# ============================================================================

def load_checkpoint(path, device):
    """
    Load a trained CCStepPredictor checkpoint, validating that its
    architecture/feature contract matches this file's current
    definitions, and reconstructing the model with the SAME
    max_log_step_ratio bound it was trained under.
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint '{path}' did not contain a dictionary.")

    if "model_state" not in ckpt:
        raise RuntimeError(f"Checkpoint '{path}' does not contain 'model_state'.")

    hidden = int(ckpt.get("hidden", 32))

    checkpoint_window = ckpt.get("window", WINDOW)
    if checkpoint_window != WINDOW:
        raise RuntimeError(
            "\nCheckpoint/model window mismatch.\n"
            f"Checkpoint window = {checkpoint_window}\n"
            f"Current WINDOW     = {WINDOW}\n\n"
            "The training and inference code must use the same temporal window."
        )

    scalar_dim = int(ckpt.get("scalar_dim", N_SCALAR_FEATURES))
    if scalar_dim != N_SCALAR_FEATURES:
        raise RuntimeError(
            "\nCheckpoint/model scalar-feature mismatch.\n"
            f"Checkpoint scalar_dim = {scalar_dim}\n"
            f"Current scalar_dim    = {N_SCALAR_FEATURES}\n\n"
            "Regenerate the checkpoint using the current 2_train_accelerator.py "
            "(this checkpoint may predate the scalar step-ratio redesign)."
        )

    checkpoint_feature_names = ckpt.get("feature_names", None)
    if checkpoint_feature_names is not None:
        checkpoint_feature_names = list(checkpoint_feature_names)
        if checkpoint_feature_names != FEATURE_NAMES:
            raise RuntimeError(
                "\nCheckpoint feature ordering does not match the current model.\n\n"
                f"Checkpoint features:\n{checkpoint_feature_names}\n\n"
                f"Current features:\n{FEATURE_NAMES}\n\n"
                "Retrain the accelerator."
            )

    max_log_step_ratio = float(ckpt.get("max_log_step_ratio", MAX_LOG_STEP_RATIO))

    model = CCStepPredictor(
        hidden=hidden,
        scalar_dim=scalar_dim,
        max_log_step_ratio=max_log_step_ratio,
    )

    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            "\nCould not load accelerator checkpoint -- this usually means "
            "the checkpoint predates the scalar-step-ratio redesign (it was "
            "trained with the old per-amplitude CCGlobalMessagePassing "
            "architecture). Retrain with the current 2_train_accelerator.py.\n\n"
            f"Checkpoint: {path}\n\n"
            f"Original PyTorch error:\n{exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model, ckpt

