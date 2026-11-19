"""
Unified CCSD Master Solver (ML-Free + ML-Accelerated)
=====================================================

Supports generalized RHF/UHF references and 9 solver methods controllable
via 'job.inp' or command-line flags:

    1. jacobi       : Standard fixed-point Jacobi iterations.
    2. diis         : Standard Pulay DIIS (error = T_new - T_old).
    3. rdiis        : Residual Pulay DIIS (error = exact residual R, 1 RHS/iter).
    4. nk           : Pure Jacobian-Free Newton-Krylov (GMRES).
    5. nk_diis      : Unconditional JFNK steps accelerated by DIIS.
    6. st_nk_diis   : Stall-Triggered Hybrid (exact parity with ccsd_diis_nk_2.py,
                      optimized to 1 RHS/iter + cached post-NK line-search RHS).
    7. st_nkml_diis : ML-Accelerated Stall-Triggered Hybrid (GRU scalar step-size
                      predictor along shared GMRES direction, verified + cached).
    8. broyden      : Limited-memory Type-II Broyden solver (zero extra RHS cost).
    9. broyden_diis : Limited-memory Broyden steps accelerated by DIIS.

The solver uses the adapted Psi4NumPy helper_CC.py backend.   

Usage:
    python3 ccsd_master.py --job job.inp
    python3 ccsd_master.py --job job.inp --method st_nkml_diis --model cc_accelerator.pt
"""

__authors__ = [
    "Shamik Chanda",
    "Suhita Basumalick",
    "Sangita Sen",
]
__license__ = "BSD-3-Clause"
__date__    = "2026"

import time
import argparse
from collections import deque
import numpy as np
import psi4

from helper_CC import helper_CCSD

np.set_printoptions(precision=5, linewidth=200, suppress=True)


# ============================================================================
# Reference & Electron-Count Validation
# ============================================================================

def validate_and_resolve_reference(mol, requested_ref=None):
    """
    Validates total electron count, charge, and spin multiplicity, and resolves
    whether RHF or UHF can be used.
      - Closed-shell (even N_elec and multiplicity == 1): RHF or UHF allowed (default RHF).
      - Open-shell (odd N_elec or multiplicity > 1): Only UHF allowed.
    """
    z_total = sum(int(round(mol.Z(a))) for a in range(mol.natom()))
    charge = int(round(mol.molecular_charge()))
    mult = int(round(mol.multiplicity()))
    n_elec = z_total - charge
    n_unpaired = mult - 1

    if n_elec < 1:
        raise ValueError(f"Invalid electron count ({n_elec}) for charge={charge}.")

    if (n_elec - n_unpaired) < 0 or (n_elec - n_unpaired) % 2 != 0:
        raise ValueError(
            f"Impossible charge/multiplicity combination: charge={charge} gives "
            f"N_elec={n_elec}, incompatible with multiplicity={mult} (N_unpaired={n_unpaired})."
        )

    is_open_shell = (n_elec % 2 != 0) or (mult > 1)
    req = None if requested_ref is None else requested_ref.strip().upper()
    if req in ("", "AUTO", "NONE"):
        req = None

    if is_open_shell:
        if req in ("RHF", "ROHF"):
            raise ValueError(
                f"Open-shell system detected (N_elec={n_elec}, multiplicity={mult}). "
                f"Requested reference '{req}' is not supported because ROHF is not "
                f"implemented and RHF requires a closed-shell singlet. Use 'UHF'."
            )
        resolved = "UHF"
    else:
        if req is None:
            resolved = "RHF"
        elif req in ("RHF", "UHF"):
            resolved = req
        else:
            raise ValueError(f"Unsupported reference '{req}'. Choose 'RHF' or 'UHF'.")

    return resolved, n_elec, mult, is_open_shell


def is_finite_array(x):
    return x is not None and np.all(np.isfinite(x))


def safe_norm(x):
    if x is None or not is_finite_array(x):
        return np.inf
    return float(np.linalg.norm(x))


def build_residual(cc, rhs1, rhs2):
    R1 = rhs1 - cc.Dia * cc.t1
    R2 = rhs2 - cc.Dijab * cc.t2
    return np.concatenate((R1.ravel(), R2.ravel()))


# ============================================================================
# Gatekeepers (ML-Free st_nk_diis & ML st_nkml_diis)
# ============================================================================

class Gatekeeper:
    """Exact state machine for ML-free st_nk_diis (parity with ccsd_diis_nk_2.py)."""

    def __init__(self, stall_count=3, stall_ratio=0.98, verify_every=3, force_after=None):
        if stall_count < 1:
            raise ValueError("stall_count must be >= 1")
        if not 0.0 < stall_ratio <= 1.0:
            raise ValueError("stall_ratio must satisfy 0 < stall_ratio <= 1")
        if verify_every < 1:
            raise ValueError("verify_every must be >= 1")

        self.stall_count = stall_count
        self.stall_ratio = stall_ratio
        self.verify_every = verify_every
        self.force_after = force_after

        self.recent_Rnorm = deque(maxlen=stall_count + 1)
        self.recovery_streak = 0
        self.iter_count = 0
        self.force_diis = False

    def is_stalled(self):
        if len(self.recent_Rnorm) < (self.stall_count + 1):
            return False
        values = list(self.recent_Rnorm)
        ratios = []
        for old, new in zip(values[:-1], values[1:]):
            if not np.isfinite(old) or not np.isfinite(new):
                return True
            ratios.append(new / max(old, 1e-300))
        return all(ratio > self.stall_ratio for ratio in ratios)

    def want_recovery(self):
        if self.force_diis:
            return False
        if self.force_after is not None and self.iter_count >= self.force_after:
            return True
        return self.is_stalled()

    def record(self, Rnorm, took_recovery):
        self.recent_Rnorm.append(float(Rnorm))
        if took_recovery:
            self.recovery_streak += 1
            if self.recovery_streak >= self.verify_every:
                self.force_diis = True
        else:
            self.recovery_streak = 0
            self.force_diis = False
        self.iter_count += 1


class MLGatekeeper:
    """Exact state machine for st_nkml_diis (parity with 3_run_ml_diis_nk.py)."""

    def __init__(self, stall_count=3, stall_ratio=0.98, verify_every=2,
                 force_after=None, min_rnorm_for_ml=None):
        self.stall_count = stall_count
        self.stall_ratio = stall_ratio
        self.verify_every = verify_every
        self.force_after = force_after
        self.min_rnorm_for_ml = min_rnorm_for_ml

        self.recent_Rnorm = deque(maxlen=stall_count + 1)
        self.ml_streak = 0
        self.iter_count = 0

    def is_stalled(self, current_Rnorm):
        recent = list(self.recent_Rnorm) + [float(current_Rnorm)]
        if len(recent) < self.stall_count + 1:
            return False
        recent = recent[-(self.stall_count + 1):]
        ratios = [
            recent[i + 1] / max(recent[i], 1.0e-300)
            for i in range(len(recent) - 1)
        ]
        return all(r > self.stall_ratio for r in ratios)

    def want_ml(self, current_Rnorm):
        if self.min_rnorm_for_ml is not None and current_Rnorm < self.min_rnorm_for_ml:
            return False
        if self.ml_streak >= self.verify_every:
            return False
        if self.force_after is not None and self.iter_count >= self.force_after:
            return True
        return self.is_stalled(current_Rnorm)

    def record(self, Rnorm, took_ml):
        self.recent_Rnorm.append(float(Rnorm))
        self.ml_streak = self.ml_streak + 1 if took_ml else 0
        self.iter_count += 1


# ============================================================================
# Rolling 19-Feature Buffer for st_nkml_diis
# ============================================================================

class RollingFeatures:
    """Tracks the 5-step window of 19 scalar features for GRU inference."""

    def __init__(self, window, eps_log, max_diis_norm):
        self.window = window
        self.eps_log = eps_log
        self.max_diis_norm = max_diis_norm
        self.buf = deque(maxlen=window)

    def _log_abs(self, x):
        return float(np.log10(max(abs(float(x)), self.eps_log)))

    @staticmethod
    def _log_pos(x):
        return float(np.log10(max(float(x), 1.0e-12)))

    def push(self, Rnorm, diis_size, R_prev1, R_prev2, R_prev3,
             ratio1, ratio2, E_prev1, E_prev2, R_std, R_max, T_std, T_max,
             jacobi_norm, diis_step_norm, diis_correction_norm,
             diis_error_norm, log_D_min, log_D_max):
        scalars = np.array([
            self._log_abs(Rnorm),
            float(diis_size) / self.max_diis_norm,
            float(R_std),
            float(R_max),
            float(T_std),
            float(T_max),
            self._log_abs(R_prev1),
            self._log_abs(R_prev2),
            self._log_abs(R_prev3),
            self._log_pos(ratio1),
            self._log_pos(ratio2),
            self._log_abs(E_prev1),
            self._log_abs(E_prev2),
            self._log_pos(jacobi_norm),
            self._log_pos(diis_step_norm),
            self._log_pos(diis_correction_norm),
            self._log_pos(diis_error_norm),
            float(log_D_min),
            float(log_D_max),
        ], dtype=np.float32)
        self.buf.append(scalars)

    def ready(self):
        return len(self.buf) == self.window

    def as_array(self):
        if not self.ready():
            raise RuntimeError("ML feature window is not full.")
        return np.stack(self.buf, axis=0)


# ============================================================================
# Jacobi & DIIS Extrapolation Helpers
# ============================================================================

def compute_jacobi_step(cc, rhs1, rhs2):
    if not is_finite_array(cc.Dia) or not is_finite_array(cc.Dijab):
        raise FloatingPointError("Non-finite Fock denominators.")
    if np.any(np.abs(cc.Dia) < 1e-14) or np.any(np.abs(cc.Dijab) < 1e-14):
        raise FloatingPointError("Near-zero denominator encountered.")

    t1_jacobi = rhs1 / cc.Dia
    t2_jacobi = rhs2 / cc.Dijab
    if not is_finite_array(t1_jacobi) or not is_finite_array(t2_jacobi):
        raise FloatingPointError("Non-finite Jacobi amplitudes.")

    T_jacobi = np.concatenate((t1_jacobi.ravel(), t2_jacobi.ravel()))
    return t1_jacobi, t2_jacobi, T_jacobi


def push_and_apply_diis(cc, diis_vals_t1, diis_vals_t2, diis_errors, error_vector, max_diis):
    """
    Stores the current (cc.t1, cc.t2) trial state and its corresponding error_vector
    in 1-to-1 lockstep and performs Pulay DIIS extrapolation.
    """
    if not is_finite_array(error_vector):
        print("          [DIIS] non-finite error vector.")
        return False, len(diis_errors)

    diis_vals_t1.append(cc.t1.copy())
    diis_vals_t2.append(cc.t2.copy())
    diis_errors.append(error_vector.copy())

    while len(diis_errors) > max_diis:
        del diis_vals_t1[0]
        del diis_vals_t2[0]
        del diis_errors[0]

    n_errors = len(diis_errors)
    if n_errors < 2:
        return False, n_errors

    B = np.full((n_errors + 1, n_errors + 1), -1.0, dtype=np.float64)
    B[-1, -1] = 0.0

    for i in range(n_errors):
        ei = diis_errors[i]
        if not is_finite_array(ei):
            print("          [DIIS] non-finite error vector.")
            return False, n_errors
        for j in range(i, n_errors):
            val = float(np.dot(ei, diis_errors[j]))
            if not np.isfinite(val):
                print("          [DIIS] non-finite B-matrix element.")
                return False, n_errors
            B[i, j] = val
            B[j, i] = val

    scale = np.max(np.abs(B[:-1, :-1]))
    if not np.isfinite(scale) or scale < 1e-30:
        print("          [DIIS] degenerate error subspace; retaining Jacobi state.")
        return False, n_errors
    B[:-1, :-1] /= scale

    rhs = np.zeros(n_errors + 1, dtype=np.float64)
    rhs[-1] = -1.0

    try:
        ci = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        print("          [DIIS] singular B matrix; retaining Jacobi state.")
        return False, n_errors

    if not is_finite_array(ci):
        print("          [DIIS] non-finite coefficients; retaining Jacobi state.")
        return False, n_errors

    if np.max(np.abs(ci[:-1])) > 1e6:
        print("          [DIIS] unstable coefficients; retaining Jacobi state.")
        return False, n_errors

    t1_new = np.zeros_like(cc.t1)
    t2_new = np.zeros_like(cc.t2)
    for i in range(n_errors):
        t1_new += ci[i] * diis_vals_t1[i]
        t2_new += ci[i] * diis_vals_t2[i]

    if not is_finite_array(t1_new) or not is_finite_array(t2_new):
        print("          [DIIS] non-finite extrapolated amplitudes.")
        return False, n_errors

    cc.t1[:] = t1_new
    cc.t2[:] = t2_new
    return True, n_errors


def preview_diis_step_lockstep(diis_vals_t1, diis_vals_t2, diis_errors):
    """
    Non-mutating Pulay DIIS preview matching 3_run_ml_diis_nk.py (including n_errors == 1).
    Returns (T_diis_flat, ok).
    """
    n_errors = len(diis_errors)
    if n_errors <= 0:
        return None, False

    B = np.full((n_errors + 1, n_errors + 1), -1.0, dtype=np.float64)
    B[-1, -1] = 0.0
    for i in range(n_errors):
        ei = diis_errors[i]
        for j in range(i, n_errors):
            val = float(np.dot(ei, diis_errors[j]))
            B[i, j] = val
            B[j, i] = val

    scale = float(np.max(np.abs(B[:-1, :-1])))
    if not np.isfinite(scale) or scale < 1.0e-300:
        return None, False
    B[:-1, :-1] /= scale

    rhs = np.zeros(n_errors + 1, dtype=np.float64)
    rhs[-1] = -1.0
    try:
        ci = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        return None, False

    if not is_finite_array(ci):
        return None, False

    t1_new = np.zeros_like(diis_vals_t1[0])
    t2_new = np.zeros_like(diis_vals_t2[0])
    for k in range(n_errors):
        t1_new += ci[k] * diis_vals_t1[k]
        t2_new += ci[k] * diis_vals_t2[k]

    T_diis = np.concatenate((t1_new.ravel(), t2_new.ravel())).astype(np.float64)
    if not is_finite_array(T_diis):
        return None, False
    return T_diis, True


# ============================================================================
# Main Unified Driver
# ============================================================================

def run_solver(
    job_path="job.inp",
    method_override=None,
    reference_override=None,
    model_override=None,
    maxiter=100,
    R_conv=1e-7,
    E_conv=1e-8,
    max_diis=8,
    memory=12.0,
    stall_count=3,
    stall_ratio=0.98,
    verify_every=None,
    nk_restart=None,
    nk_maxiter=None,
    nk_tol=None,
    nk_step_fractions=(0.125, 0.25, 0.5, 1.0, 1.5, 2.0),
    nk_min_reduction=1e-3,
    nk_failed_gmres_max_factor=1.0,
    ml_fallback_fractions=(0.25, 0.5, 1.0, 1.5, 2.0),
    min_rnorm_for_ml=None,
    force_after=None,
    compare_psi4=True,
):
    if maxiter < 1 or max_diis < 1:
        raise ValueError("maxiter and max_diis must be >= 1")
    if R_conv <= 0 or E_conv <= 0:
        raise ValueError("R_conv and E_conv must be positive")

    # ------------------------------------------------------------------
    # Read job file
    # ------------------------------------------------------------------
    basis = "cc-pvdz"
    method = "diis"
    freeze_core = False
    mem = memory
    file_ref = None
    file_model = None
    geom_lines = []

    with open(job_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            low = s.lower()
            if low.startswith("basis"):
                basis = s.split()[1]
            elif low.startswith("method"):
                method = s.split()[1].lower()
            elif low.startswith("reference"):
                file_ref = s.split()[1].upper()
            elif low.startswith("model"):
                file_model = s.split(None, 1)[1].strip()
            elif low.startswith("freeze_core"):
                freeze_core = (s.split()[1].lower() == "true")
            elif low.startswith("memory"):
                mem = float(s.split()[1])
            else:
                geom_lines.append(line)

    if method_override is not None:
        method = method_override.lower()
    requested_ref = reference_override if reference_override is not None else file_ref
    model_path = model_override or file_model or "cc_accelerator.pt"

    valid_methods = [
        "jacobi", "diis", "rdiis", "nk", "nk_diis",
        "st_nk_diis", "st_nkml_diis", "broyden", "broyden_diis"
    ]
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {valid_methods}")

    # Resolve method-exact defaults unless explicitly overridden on CLI
    if verify_every is None:
        verify_every = 2 if method == "st_nkml_diis" else 3
    if nk_restart is None:
        nk_restart = 5 if method == "st_nkml_diis" else 10
    if nk_maxiter is None:
        nk_maxiter = 5 if method == "st_nkml_diis" else 10
    if nk_tol is None:
        nk_tol = 1e-7 if method == "st_nkml_diis" else 1e-8

    if nk_restart < 1 or nk_maxiter < 1 or nk_tol <= 0:
        raise ValueError("Invalid NK GMRES settings")

    # ------------------------------------------------------------------
    # Lazy-load ML stack ONLY for st_nkml_diis
    # ------------------------------------------------------------------
    ml_model = None
    ml_device = None
    torch_mod = None
    packed_residual_fn = None

    if method == "st_nkml_diis":
        import torch as torch_mod
        from cc_ml_common import (
            WINDOW, EPS_LOG, MAX_DIIS_NORM,
            d_spread_features, safe_D, load_checkpoint, packed_residual,
        )
        packed_residual_fn = packed_residual
        ml_device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
        ml_model, ckpt = load_checkpoint(model_path, ml_device)
        ml_model.eval()
        print(f"Using ML device: {ml_device}")
        print(f"Loaded ML model: {model_path}")
        print(f"Checkpoint val MSE: {ckpt.get('best_val_mse', 'n/a')}")
        print(f"Model log-step-ratio bound: +/-{ml_model.max_log_step_ratio}\n")

    # ------------------------------------------------------------------
    # Initialize Psi4 & Spin-Orbital CCSD
    # ------------------------------------------------------------------
    psi4.set_memory(f"{mem} GB")
    psi4.core.set_output_file("output.dat", False)
    mol = psi4.geometry("".join(geom_lines))

    resolved_ref, n_elec, mult, is_open_shell = validate_and_resolve_reference(
        mol, requested_ref=requested_ref
    )
    psi4.set_options({"basis": basis, "reference": resolved_ref})

    print("==================================================")
    print(" Unified CCSD Master Solver")
    print("==================================================")
    print(f" System        : N_elec={n_elec}, Multiplicity={mult} ({'open-shell' if is_open_shell else 'closed-shell'})")
    print(f" Reference     : {resolved_ref}")
    print(f" Method        : {method}")
    print(f" Basis         : {basis}")
    if method == "st_nkml_diis":
        print(f" ML Model      : {model_path}")
    print("==================================================\n")

    cc = helper_CCSD(mol, memory=mem, freeze_core=freeze_core, reference=resolved_ref)

    gate = Gatekeeper(
        stall_count=stall_count,
        stall_ratio=stall_ratio,
        verify_every=verify_every,
        force_after=force_after,
    )

    ml_gate = None
    feats = None
    Dsafe = None
    log_D_min = log_D_max = 0.0
    history_Rnorm = []
    history_E = []

    if method == "st_nkml_diis":
        D = cc.pack_diag().astype(np.float64)
        Dsafe = safe_D(D)
        log_D_min, log_D_max = d_spread_features(D)
        print(
            f"Fock denominator spread: log|D| in [{log_D_min:.2f}, {log_D_max:.2f}] "
            f"({10 ** (log_D_max - log_D_min):.1f}x range)"
        )
        n1 = cc.t1.size
        if n1 + n1 * n1 != D.size:
            raise RuntimeError("Packed T1/T2 size is inconsistent.")

        feats = RollingFeatures(WINDOW, EPS_LOG, MAX_DIIS_NORM)
        ml_gate = MLGatekeeper(
            stall_count=stall_count,
            stall_ratio=stall_ratio,
            verify_every=verify_every,
            force_after=force_after,
            min_rnorm_for_ml=min_rnorm_for_ml,
        )

    if force_after is not None and method in ("st_nk_diis", "st_nkml_diis"):
        print(f"*** DEBUG MODE: forcing recovery from iteration {force_after} onward. ***")
        print("*** Results are NOT a valid efficiency benchmark. ***")

    diis_vals_t1 = []
    diis_vals_t2 = []
    diis_errors = []
    diis_size = 0

    ccsd_tstart = time.time()
    CCSDcorr_E_old = cc.compute_corr_energy()
    if not np.isfinite(CCSDcorr_E_old):
        raise FloatingPointError("Initial CCSD correlation energy is non-finite.")

    print(f"Iter   0: Ecorr = {CCSDcorr_E_old:.12f}  MP2")

    n_rhs_start = cc.nrhs
    nrhs_old = cc.nrhs
    nres_old = cc.nresidual

    n_diis_steps = 0
    n_diis_fail = 0
    n_nk_attempts = 0
    n_nk_accepted = 0
    n_nk_rejected = 0
    n_nk_gmres_failures = 0
    n_ml_attempted = 0
    n_ml_accepted = 0
    n_ml_rejected = 0
    n_nk_fallback = 0
    n_nk_fallback_accepted = 0

    converged = False
    stopped_early = False
    last_it = 0
    Rnorm = np.inf
    Rnorm_before = np.inf
    took_recovery_prev = False

    for it in range(1, maxiter + 1):
        last_it = it
        T_before = cc.pack().astype(np.float64)
        oldt1, oldt2 = cc.t1.copy(), cc.t2.copy()
        diis_size_for_features = diis_size

        # ------------------------------------------------------------------
        # Single RHS Evaluation per Iteration (or reuse cached trial RHS)
        # ------------------------------------------------------------------
        if cc._cached_rhs is not None:
            rhs1, rhs2 = cc._cached_rhs
            cc._cached_rhs = None
        else:
            cc.nresidual += 1
            rhs1, rhs2 = cc.compute_rhs()

        R_before = build_residual(cc, rhs1, rhs2)
        Rnorm_before = safe_norm(R_before)
        cc.current_R = R_before
        cc.current_Rnorm = Rnorm_before

        if not np.isfinite(Rnorm_before):
            print(f"Iter {it:3d}: non-finite residual -> stopping.")
            cc.unpack(T_before)
            stopped_early = True
            break

        Rnorm = Rnorm_before

        # Gatekeeper update for ML-free st_nk_diis (exact parity with ccsd_diis_nk_2.py)
        if it > 1 and method == "st_nk_diis":
            gate.record(Rnorm_before, took_recovery_prev)

        # Print iteration header
        if method == "st_nk_diis":
            mode_str = "DIIS*" if gate.force_diis else ("NK" if gate.want_recovery() else "DIIS")
            print(
                f"Iter {it:3d}: ||R||={Rnorm_before:.3e}  DIIS={diis_size}  "
                f"mode={mode_str}  NKstreak={gate.recovery_streak}"
            )
        elif method == "st_nkml_diis":
            print(
                f"Iter {it:3d}: ||R||={Rnorm_before:.3e}  DIIS={diis_size}  "
                f"ML_streak={ml_gate.ml_streak}"
            )
        else:
            print(f"Iter {it:3d}: ||R||={Rnorm_before:.3e}  DIIS={diis_size}  mode={method.upper()}")

        # ------------------------------------------------------------------
        # Convergence Check (Rnorm < R_conv)
        # ------------------------------------------------------------------
        if Rnorm_before < R_conv:
            print(f"\nConverged in {it - 1} iterations.")
            converged = True
            break

        # ------------------------------------------------------------------
        # Amplitude Update Dispatcher
        # ------------------------------------------------------------------
        took_recovery_prev = False
        took_ml = False

        if method == "jacobi":
            try:
                t1_jacobi, t2_jacobi, _ = compute_jacobi_step(cc, rhs1, rhs2)
            except FloatingPointError as exc:
                print(f"          [Jacobi] ERROR: {exc}")
                cc.unpack(T_before)
                stopped_early = True
                break
            cc.t1[:], cc.t2[:] = t1_jacobi, t2_jacobi

        elif method == "diis":
            try:
                t1_jacobi, t2_jacobi, _ = compute_jacobi_step(cc, rhs1, rhs2)
            except FloatingPointError as exc:
                print(f"          [Jacobi] ERROR: {exc}")
                cc.unpack(T_before)
                stopped_early = True
                break
            cc.t1[:], cc.t2[:] = t1_jacobi, t2_jacobi
            error_vec = np.concatenate(((cc.t1 - oldt1).ravel(), (cc.t2 - oldt2).ravel()))
            _, diis_size = push_and_apply_diis(
                cc, diis_vals_t1, diis_vals_t2, diis_errors, error_vec, max_diis
            )
            n_diis_steps += 1

        elif method == "rdiis":
            try:
                t1_jacobi, t2_jacobi, _ = compute_jacobi_step(cc, rhs1, rhs2)
            except FloatingPointError as exc:
                print(f"          [Jacobi] ERROR: {exc}")
                cc.unpack(T_before)
                stopped_early = True
                break
            cc.t1[:], cc.t2[:] = t1_jacobi, t2_jacobi
            _, diis_size = push_and_apply_diis(
                cc, diis_vals_t1, diis_vals_t2, diis_errors, R_before, max_diis
            )
            n_diis_steps += 1

        elif method == "nk":
            n_nk_attempts += 1
            _, dT_raw, info, _, _ = cc.newton_direction(
                R=R_before, restart=nk_restart, tol=nk_tol, maxiter=nk_maxiter, return_raw=True
            )
            if info != 0:
                n_nk_gmres_failures += 1
            if dT_raw is not None:
                cc.unpack(T_before + dT_raw)
                n_nk_accepted += 1
            else:
                n_nk_rejected += 1
                t1_jacobi, t2_jacobi, _ = compute_jacobi_step(cc, rhs1, rhs2)
                cc.t1[:], cc.t2[:] = t1_jacobi, t2_jacobi

        elif method == "nk_diis":
            n_nk_attempts += 1
            _, dT_raw, info, _, _ = cc.newton_direction(
                R=R_before, restart=nk_restart, tol=nk_tol, maxiter=nk_maxiter, return_raw=True
            )
            if info != 0:
                n_nk_gmres_failures += 1
            if dT_raw is not None:
                cc.unpack(T_before + dT_raw)
                n_nk_accepted += 1
            else:
                n_nk_rejected += 1
                t1_jacobi, t2_jacobi, _ = compute_jacobi_step(cc, rhs1, rhs2)
                cc.t1[:], cc.t2[:] = t1_jacobi, t2_jacobi

            error_vec = cc.pack() - T_before
            _, diis_size = push_and_apply_diis(
                cc, diis_vals_t1, diis_vals_t2, diis_errors, error_vec, max_diis
            )
            n_diis_steps += 1

        elif method == "st_nk_diis":
            try:
                t1_jacobi, t2_jacobi, T_jacobi = compute_jacobi_step(cc, rhs1, rhs2)
            except FloatingPointError as exc:
                print(f"          [Jacobi] ERROR: {exc}")
                cc.unpack(T_before)
                stopped_early = True
                break

            dT_jacobi = T_jacobi - T_before
            jacobi_scale = safe_norm(dT_jacobi)
            if not np.isfinite(jacobi_scale) or jacobi_scale < 1e-14:
                jacobi_scale = max(1e-14, safe_norm(T_before) * 1e-8)

            if gate.want_recovery():
                n_nk_attempts += 1
                try:
                    v_unit, gmres_info, _, gmres_iters = cc.newton_direction(
                        R=R_before, restart=nk_restart, tol=nk_tol, maxiter=nk_maxiter, return_raw=False
                    )
                except Exception as exc:
                    print(f"          [NK] Newton solve failed: {exc}")
                    v_unit, gmres_info, gmres_iters = None, -1, 0

                if gmres_info != 0:
                    n_nk_gmres_failures += 1

                direction_norm = safe_norm(v_unit)
                if v_unit is None or not np.isfinite(direction_norm) or direction_norm < 1e-14:
                    print("          [NK] invalid Newton direction -> fallback to DIIS")
                    cc.unpack(T_before)
                    n_nk_rejected += 1
                else:
                    trial_scales = tuple(jacobi_scale * f for f in nk_step_fractions)
                    if gmres_info != 0:
                        max_failed_step = jacobi_scale * nk_failed_gmres_max_factor
                        trial_scales = tuple(min(s, max_failed_step) for s in trial_scales)
                    trial_scales = tuple(
                        sorted(set(s for s in trial_scales if s > 0.0 and np.isfinite(s)))
                    )

                    print(f"          [NK] GMRES iters={gmres_iters}  info={gmres_info}")

                    try:
                        best_s, best_norm, accepted = cc.line_search_along(
                            v_unit,
                            R_before=R_before,
                            trial_scales=trial_scales,
                            min_relative_reduction=nk_min_reduction,
                        )
                    except Exception as exc:
                        print(f"          [NK] line search failed: {exc}")
                        best_s, best_norm, accepted = 0.0, np.inf, False

                    relative_reduction = (Rnorm_before - best_norm) / max(Rnorm_before, 1e-300)
                    print(
                        f"          [NK] ||R||before={Rnorm_before:.3e}  "
                        f"||R||trial={best_norm:.3e}  reduction={relative_reduction:.3e}"
                    )

                    valid_accept = (
                        accepted
                        and np.isfinite(best_norm)
                        and best_norm < Rnorm_before
                        and relative_reduction >= nk_min_reduction
                    )

                    if valid_accept:
                        took_recovery_prev = True
                        n_nk_accepted += 1
                    else:
                        cc.unpack(T_before)
                        cc._cached_rhs = None
                        n_nk_rejected += 1

            if not took_recovery_prev:
                cc.unpack(T_before)
                cc.t1[:], cc.t2[:] = t1_jacobi, t2_jacobi
                error_vec = np.concatenate(((cc.t1 - oldt1).ravel(), (cc.t2 - oldt2).ravel()))
                _, diis_size = push_and_apply_diis(
                    cc, diis_vals_t1, diis_vals_t2, diis_errors, error_vec, max_diis
                )
                n_diis_steps += 1

        elif method == "st_nkml_diis":
            try:
                t1_jacobi, t2_jacobi, T_jacobi = compute_jacobi_step(cc, rhs1, rhs2)
            except FloatingPointError as exc:
                print(f"          [Jacobi] ERROR: {exc}")
                cc.unpack(T_before)
                stopped_early = True
                break

            dT_jacobi = T_jacobi - T_before
            jacobi_norm = float(np.linalg.norm(dT_jacobi))

            error_vec = np.concatenate((
                (t1_jacobi - oldt1).ravel(),
                (t2_jacobi - oldt2).ravel(),
            )).astype(np.float64)

            diis_vals_t1.append(t1_jacobi.copy())
            diis_vals_t2.append(t2_jacobi.copy())
            diis_errors.append(error_vec.copy())
            while len(diis_errors) > max_diis:
                del diis_vals_t1[0]
                del diis_vals_t2[0]
                del diis_errors[0]
            diis_size = len(diis_errors)

            T_diis_preview, diis_ok = preview_diis_step_lockstep(
                diis_vals_t1, diis_vals_t2, diis_errors
            )
            dT_diis = (T_diis_preview - T_before) if diis_ok else np.zeros_like(T_before)

            diis_step_norm = float(np.linalg.norm(dT_diis))
            diis_correction_norm = float(np.linalg.norm(dT_diis - dT_jacobi))
            error_norm = float(np.linalg.norm(error_vec))

            R_prev1 = history_Rnorm[-1] if len(history_Rnorm) >= 1 else 0.0
            R_prev2 = history_Rnorm[-2] if len(history_Rnorm) >= 2 else 0.0
            R_prev3 = history_Rnorm[-3] if len(history_Rnorm) >= 3 else 0.0
            E_prev1 = history_E[-1] if len(history_E) >= 1 else 0.0
            E_prev2 = history_E[-2] if len(history_E) >= 2 else 0.0
            ratio1 = Rnorm_before / max(R_prev1, 1.0e-15) if R_prev1 > 1.0e-15 else 1.0
            ratio2 = R_prev1 / max(R_prev2, 1.0e-15) if R_prev2 > 1.0e-15 else 1.0

            Rn_now = R_before / Dsafe
            feats.push(
                Rnorm_before, diis_size_for_features, R_prev1, R_prev2, R_prev3,
                ratio1, ratio2, E_prev1, E_prev2,
                float(np.std(Rn_now)), float(np.max(np.abs(Rn_now))),
                float(np.std(T_before)), float(np.max(np.abs(T_before))),
                jacobi_norm, diis_step_norm, diis_correction_norm, error_norm,
                log_D_min, log_D_max,
            )

            ml_failed = False
            v_dom = None
            diis_rhs_cache = None

            if diis_ok and feats.ready() and ml_gate.want_ml(Rnorm_before):
                n_ml_attempted += 1
                print("          [ML] DIIS stall detected; predicting step ratio.")
                try:
                    # 1. Exact ordinary-DIIS baseline at this state (cached if later committed)
                    cc.unpack(T_diis_preview)
                    cc.nresidual += 1
                    diis_rhs1, diis_rhs2 = cc.compute_rhs()
                    _, diis_trial_norm = packed_residual_fn(cc, diis_rhs1, diis_rhs2)
                    diis_rhs_cache = (diis_rhs1, diis_rhs2)
                    cc.unpack(T_before)

                    # 2. Single GMRES solve reusing R_before (shared by ML and fallback)
                    v_dom, gmres_info, _, _ = cc.newton_direction(
                        R=R_before, restart=nk_restart, tol=nk_tol, maxiter=nk_maxiter, return_raw=False
                    )
                    if gmres_info != 0:
                        n_nk_gmres_failures += 1
                    if v_dom is None:
                        raise RuntimeError("GMRES returned invalid Newton direction.")

                    # 3. GRU scalar step-size prediction
                    sc_t = torch_mod.from_numpy(feats.as_array()[None, :, :]).to(ml_device)
                    with torch_mod.no_grad():
                        pred = ml_model(sc_t)
                    log_ratio_pred = float(pred.detach().cpu().numpy()[0])
                    if not np.isfinite(log_ratio_pred):
                        raise RuntimeError("ML model returned non-finite step ratio.")

                    s_pred = jacobi_norm * (10.0 ** log_ratio_pred)
                    cc.unpack(T_before + s_pred * v_dom)
                    cc.nresidual += 1
                    trial_rhs1, trial_rhs2 = cc.compute_rhs()
                    R_trial, trial_norm = packed_residual_fn(cc, trial_rhs1, trial_rhs2)

                    print(
                        f"          [ML verify] before={Rnorm_before:.3e} "
                        f"DIIS_trial={diis_trial_norm:.3e} "
                        f"ML_trial={trial_norm:.3e} "
                        f"ML/current={trial_norm / max(Rnorm_before, 1e-300):.4f} "
                        f"ML/DIIS={trial_norm / max(diis_trial_norm, 1e-300):.4f} "
                        f"gmres_info={gmres_info}"
                    )

                    if (
                        np.isfinite(trial_norm)
                        and np.isfinite(diis_trial_norm)
                        and trial_norm < Rnorm_before
                        and trial_norm < diis_trial_norm
                    ):
                        cc.current_R = R_trial
                        cc.current_Rnorm = trial_norm
                        cc._cached_rhs = (trial_rhs1, trial_rhs2)
                        took_ml = True
                        n_ml_accepted += 1
                        print("          [ML accept] beats current state and ordinary DIIS trial.")
                    else:
                        cc.unpack(T_before)
                        ml_failed = True
                        n_ml_rejected += 1
                        print("          [ML reject] reusing Newton direction for NK fallback.")
                except Exception as exc:
                    cc.unpack(T_before)
                    ml_failed = True
                    n_ml_rejected += 1
                    print(f"          [ML error] {exc}")

            if not took_ml:
                if ml_failed:
                    n_nk_fallback += 1
                    if v_dom is None:
                        v_dom, _, _, _ = cc.newton_direction(
                            R=R_before, restart=nk_restart, tol=nk_tol, maxiter=nk_maxiter, return_raw=False
                        )
                    accepted_nk = False
                    if v_dom is not None:
                        scales = tuple(jacobi_norm * f for f in ml_fallback_fractions)
                        best_s, best_norm, accepted_nk = cc.line_search_along(
                            v_dom, R_before=R_before, trial_scales=scales, min_relative_reduction=0.0
                        )
                        print(
                            f"          [NK fallback] s={best_s:.3e} before={Rnorm_before:.3e} "
                            f"after={best_norm:.3e} accepted={accepted_nk}"
                        )
                    if accepted_nk:
                        n_nk_fallback_accepted += 1
                    else:
                        if diis_ok:
                            cc.unpack(T_diis_preview)
                            cc._cached_rhs = diis_rhs_cache
                            n_diis_steps += 1
                        else:
                            cc.unpack(T_before)
                            cc.t1[:], cc.t2[:] = t1_jacobi, t2_jacobi
                            n_diis_fail += 1
                else:
                    if diis_ok:
                        cc.unpack(T_diis_preview)
                        n_diis_steps += 1
                    else:
                        cc.t1[:], cc.t2[:] = t1_jacobi, t2_jacobi
                        n_diis_fail += 1

        elif method == "broyden":
            try:
                _, _, T_jacobi = compute_jacobi_step(cc, rhs1, rhs2)
            except FloatingPointError as exc:
                print(f"          [Jacobi] ERROR: {exc}")
                cc.unpack(T_before)
                stopped_early = True
                break
            dT_jacobi = T_jacobi - T_before
            cc.update_broyden(R_before=R_before, dT_jacobi=dT_jacobi)

        elif method == "broyden_diis":
            try:
                _, _, T_jacobi = compute_jacobi_step(cc, rhs1, rhs2)
            except FloatingPointError as exc:
                print(f"          [Jacobi] ERROR: {exc}")
                cc.unpack(T_before)
                stopped_early = True
                break
            dT_jacobi = T_jacobi - T_before
            cc.update_broyden(R_before=R_before, dT_jacobi=dT_jacobi)
            error_vec = cc.pack() - T_before
            _, diis_size = push_and_apply_diis(
                cc, diis_vals_t1, diis_vals_t2, diis_errors, error_vec, max_diis
            )
            n_diis_steps += 1

        # ------------------------------------------------------------------
        # Energy Evaluation & Safety Reversion
        # ------------------------------------------------------------------
        try:
            CCSDcorr_E = cc.compute_corr_energy()
        except Exception as exc:
            print(f"          [Energy] evaluation failed: {exc}")
            cc.unpack(T_before)
            cc._cached_rhs = None
            CCSDcorr_E = CCSDcorr_E_old
            took_recovery_prev = False
            took_ml = False

        if not np.isfinite(CCSDcorr_E):
            print("          [Energy] non-finite energy -> restoring T_before.")
            cc.unpack(T_before)
            cc._cached_rhs = None
            CCSDcorr_E = CCSDcorr_E_old
            took_recovery_prev = False
            took_ml = False

        dE = CCSDcorr_E - CCSDcorr_E_old
        CCSDcorr_E_old = CCSDcorr_E

        if method == "st_nkml_diis":
            history_Rnorm.append(Rnorm_before)
            history_E.append(CCSDcorr_E)
            ml_gate.record(Rnorm_before, took_ml)

        nrhs_iter = cc.nrhs - nrhs_old
        nrhs_old = cc.nrhs
        nres_iter = cc.nresidual - nres_old
        nres_old = cc.nresidual

        print(
            f"          Ecorr={CCSDcorr_E:.12f}  dE={dE:+.3e}  "
            f"Residuals={nres_iter}  RHS={nrhs_iter}"
        )

    # ----------------------------------------------------------------------
    # Final Residual Check on Iteration == maxiter
    # ----------------------------------------------------------------------
    if not converged and not stopped_early:
        if cc._cached_rhs is not None:
            rhs1_f, rhs2_f = cc._cached_rhs
            cc._cached_rhs = None
        else:
            cc.nresidual += 1
            rhs1_f, rhs2_f = cc.compute_rhs()
        R_final = build_residual(cc, rhs1_f, rhs2_f)
        Rnorm = safe_norm(R_final)

        if Rnorm < R_conv:
            print(f"\nConverged in {maxiter} iterations.")
            converged = True
        else:
            print(f"\nWARNING: reached maxiter={maxiter} without convergence.")

    wall = time.time() - ccsd_tstart
    CCSD_E = cc.rhf_e + CCSDcorr_E_old
    iters_completed = (last_it - 1) if (converged and last_it <= maxiter and Rnorm_before < R_conv) else last_it

    print("\n==============================")
    print("Solver Statistics")
    print("==============================")
    print(f"Method                      = {method}")
    print(f"Reference                   = {resolved_ref}")
    print(f"Converged                   = {converged}")
    print(f"Iterations                  = {iters_completed}")
    print(f"DIIS steps                  = {n_diis_steps}")
    if method == "st_nkml_diis":
        print(f"DIIS failures               = {n_diis_fail}")
        print(f"ML attempts                 = {n_ml_attempted}")
        print(f"ML accepted                 = {n_ml_accepted}")
        print(f"ML rejected                 = {n_ml_rejected}")
        print(f"NK fallbacks                = {n_nk_fallback}")
        print(f"NK fallbacks accepted       = {n_nk_fallback_accepted}")
    else:
        print(f"NK attempts                 = {n_nk_attempts}")
        print(f"NK accepted                 = {n_nk_accepted}")
        print(f"NK rejected                 = {n_nk_rejected}")
    print(f"NK GMRES incomplete         = {n_nk_gmres_failures}")
    print(f"compute_rhs() calls         = {cc.nrhs - n_rhs_start}")
    print(f"residual() calls            = {cc.nresidual}")
    print(f"Final residual norm         = {Rnorm:.3e}")
    print(f"Final correlation energy    = {CCSDcorr_E_old:.12f}")
    print(f"Total CCSD energy           = {CCSD_E:.12f}")
    print(f"Wall time (s)               = {wall:.2f}")

   #if compare_psi4:
   #    try:
   #        psi4.compare_values(psi4.energy("CCSD", ref_wfn=cc.wfn), CCSD_E, 6, "CCSD Energy")
   #    except Exception as exc:
   #        print("\nPsi4 native CCSD comparison failed or did not converge independently:")
   #        print(exc)


    if compare_psi4:
        try:
            psi4.set_options({"ccenergy__maxiter": 200, "ccenergy__r_convergence": 1e-8})
            psi4.compare_values(psi4.energy("CCSD", ref_wfn=cc.wfn), CCSD_E, 6, "CCSD Energy")
        except Exception as exc:
            print("\nPsi4 native CCSD comparison failed or did not converge independently:")
            print(exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Master CCSD Solver (ML-Free + ML).")
    parser.add_argument("--job", default="job.inp")
    parser.add_argument("--method", default=None, help="Override method in job.inp")
    parser.add_argument("--reference", default=None, help="Override reference (RHF or UHF)")
    parser.add_argument("--model", default=None, help="Path to GRU checkpoint for st_nkml_diis")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--stall_count", type=int, default=3)
    parser.add_argument("--stall_ratio", type=float, default=0.98)
    parser.add_argument("--verify_every", type=int, default=None, help="Default: 3 (st_nk_diis), 2 (st_nkml_diis)")
    parser.add_argument("--nk_restart", type=int, default=None, help="Default: 10 (st_nk_diis), 5 (st_nkml_diis)")
    parser.add_argument("--nk_maxiter", type=int, default=None, help="Default: 10 (st_nk_diis), 5 (st_nkml_diis)")
    parser.add_argument("--nk_tol", type=float, default=None, help="Default: 1e-8 (st_nk_diis), 1e-7 (st_nkml_diis)")
    parser.add_argument("--nk_min_reduction", type=float, default=1e-3)
    parser.add_argument("--nk_failed_gmres_max_factor", type=float, default=1.0)
    parser.add_argument("--min_rnorm_for_ml", type=float, default=None)
    parser.add_argument("--force_recovery_test", type=int, default=None)
    args = parser.parse_args()

    run_solver(
        job_path=args.job,
        method_override=args.method,
        reference_override=args.reference,
        model_override=args.model,
        maxiter=args.maxiter,
        stall_count=args.stall_count,
        stall_ratio=args.stall_ratio,
        verify_every=args.verify_every,
        nk_restart=args.nk_restart,
        nk_maxiter=args.nk_maxiter,
        nk_tol=args.nk_tol,
        nk_min_reduction=args.nk_min_reduction,
        nk_failed_gmres_max_factor=args.nk_failed_gmres_max_factor,
        min_rnorm_for_ml=args.min_rnorm_for_ml,
        force_after=args.force_recovery_test,
    )
