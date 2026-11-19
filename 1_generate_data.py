"""
Script 1: Harvest DIIS Trajectories + Scalar Newton-Step-Ratio Targets
======================================================================

Harvests closed-shell CCSD/DIIS convergence trajectories and computes the
scalar oracle target at genuine DIIS stalls:

    step_log_ratio = log10(best_s / jacobi_norm)

where v_dom is the unit GMRES Newton direction from cc.newton_direction()
and best_s minimizes the nonlinear CCSD residual along a logarithmic grid
of trial step lengths via cc.line_search_along().

Compatible with the unified RHF/UHF helper_CC.py backend.
__authors__ = ["Shamik Chanda", "Suhita Basumalick", "Sangita Sen"]
__license__ = "BSD-3-Clause"
__date__    = "2026"
"""

import os

# Limit BLAS/OpenMP threading inside each independent worker. The workers
# themselves provide the parallelism. Set these BEFORE importing NumPy/Psi4.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psi4

from helper_CC import helper_CCSD
from cc_ml_common import (
    is_stalled,
    d_spread_features,
    MAX_LOG_STEP_RATIO,
    safe_D,
    ORACLE_COARSE_POINTS,
)


def harvest_trajectory(
    geom_string,
    basis,
    filename_prefix,
    maxiter=100,
    memory_gb=12.0,
    output_dir="cc_training_data",
    nk_restart=5,
    nk_maxiter=5,
    nk_tol=1e-7,
):
    os.makedirs(output_dir, exist_ok=True)

    # Every worker has its own Psi4 state and its own output files.
    psi4_output = os.path.join(output_dir, f"{filename_prefix}.dat")
    npz_output = os.path.join(output_dir, f"{filename_prefix}.npz")

    psi4.core.set_output_file(psi4_output, False)
    psi4.set_memory(f"{memory_gb} GB")

    mol = psi4.geometry(geom_string)
    psi4.set_options({"basis": basis})

    cc = helper_CCSD(mol, freeze_core=False, memory=memory_gb)

    max_diis = 8
    R_conv = 1e-7

    D = np.concatenate([cc.Dia.ravel(), cc.Dijab.ravel()])

    assert np.all(D < 0), (
        "Fock denominators must all be negative. "
        "Check that Dia = eps_i - eps_a and "
        "Dijab = eps_i + eps_j - eps_a - eps_b."
    )

    log_D_min, log_D_max = d_spread_features(D)
    Dsafe = safe_D(D)

    diis_vals_t1 = [cc.t1.copy()]
    diis_vals_t2 = [cc.t2.copy()]
    diis_errors = []
    diis_size = 0

    dataset = []
    history_Rnorm = []
    history_E = []
    n_stall_points = 0

    print(f"Harvesting DIIS trajectory (+ scalar NK-ratio oracle) for {filename_prefix}")

    cc.compute_corr_energy()

    for ccsd_iter in range(1, maxiter + 1):

        T_before = cc.pack()

        rhs1, rhs2 = cc.compute_rhs()
        R1 = rhs1 - cc.Dia * cc.t1
        R2 = rhs2 - cc.Dijab * cc.t2
        R_before = np.concatenate([R1.ravel(), R2.ravel()])
        Rnorm = float(np.linalg.norm(R_before))

        E_before = cc.compute_corr_energy()

        R_prev1 = history_Rnorm[-1] if len(history_Rnorm) >= 1 else 0.0
        R_prev2 = history_Rnorm[-2] if len(history_Rnorm) >= 2 else 0.0
        R_prev3 = history_Rnorm[-3] if len(history_Rnorm) >= 3 else 0.0
        E_prev1 = history_E[-1] if len(history_E) >= 1 else 0.0
        E_prev2 = history_E[-2] if len(history_E) >= 2 else 0.0

        ratio1 = Rnorm / R_prev1 if R_prev1 > 1.0e-15 else 1.0
        ratio2 = R_prev1 / R_prev2 if R_prev2 > 1.0e-15 else 1.0

        is_stall_flag = is_stalled(history_Rnorm + [Rnorm])
        if is_stall_flag:
            n_stall_points += 1

        # Scalar statistics of T/R -- computed now from in-memory arrays,
        # without ever persisting high-dimensional T/R tensors to disk.
        Rn_now = (R_before / Dsafe).astype(np.float64)
        R_std = float(np.std(Rn_now))
        R_max = float(np.max(np.abs(Rn_now)))
        T_std = float(np.std(T_before))
        T_max = float(np.max(np.abs(T_before)))

        print(
            f"Iter {ccsd_iter:3d}: ||R|| = {Rnorm:.3e}  DIIS = {diis_size}  "
            f"stall = {is_stall_flag}"
        )

        if Rnorm < R_conv:
            print(f"Converged in {ccsd_iter - 1} iterations.")
            break

        # ----------------------------------------------------------------
        # JACOBI CANDIDATE (needed both as the real step and as the
        # natural length scale for the oracle target)
        # ----------------------------------------------------------------
        t1_jacobi = rhs1 / cc.Dia
        t2_jacobi = rhs2 / cc.Dijab
        T_jacobi = np.concatenate([t1_jacobi.ravel(), t2_jacobi.ravel()])
        dT_jacobi = T_jacobi - T_before
        JacobiNorm = float(np.linalg.norm(dT_jacobi))

        # ----------------------------------------------------------------
        # SCALAR NK-RATIO ORACLE -- ONLY AT A GENUINE STALL
        # ----------------------------------------------------------------
        step_log_ratio = np.nan
        if is_stall_flag and JacobiNorm > 1.0e-300:
            v_dom, gmres_info, _, _ = cc.newton_direction(
                R=R_before,
                restart=nk_restart,
                tol=nk_tol,
                maxiter=nk_maxiter,
                return_raw=False,
            )

            if v_dom is not None:
                ratios_grid = np.geomspace(1.0e-3, 1.0e3, num=ORACLE_COARSE_POINTS)
                trial_scales = JacobiNorm * ratios_grid
                best_s, best_norm, accepted = cc.line_search_along(
                    v_dom,
                    R_before=R_before,
                    trial_scales=trial_scales,
                    min_relative_reduction=0.0,
                )
                cc.unpack(T_before)
                cc._cached_rhs = None
                cc.current_R = R_before
                cc.current_Rnorm = Rnorm

                if accepted and best_s > 0.0 and np.isfinite(best_norm):
                    raw_log_ratio = np.log10(best_s / JacobiNorm)
                else:
                    raw_log_ratio = -MAX_LOG_STEP_RATIO

                step_log_ratio = float(
                    np.clip(raw_log_ratio, -MAX_LOG_STEP_RATIO, MAX_LOG_STEP_RATIO)
                )
                print(
                    f"          oracle(stall): before={Rnorm:.3e} after={best_norm:.3e} "
                    f"accepted={accepted} best_s/jacobi={best_s / max(JacobiNorm, 1e-300):.3e} "
                    f"step_log_ratio={step_log_ratio:+.3f} gmres_info={gmres_info}"
                )
            else:
                cc.unpack(T_before)
                cc._cached_rhs = None
                print(
                    f"          oracle(stall): GMRES returned invalid direction "
                    f"(info={gmres_info}); leaving target as NaN."
                )
        elif is_stall_flag:
            print("          oracle: skipped because Jacobi step is numerically zero.")

        # ----------------------------------------------------------------
        # DIIS ERROR VECTOR (needs the pre-commit Jacobi state)
        # ----------------------------------------------------------------
        oldt1, oldt2 = cc.t1.copy(), cc.t2.copy()
        cc.t1[:] = t1_jacobi
        cc.t2[:] = t2_jacobi

        error = np.concatenate([(cc.t1 - oldt1).ravel(), (cc.t2 - oldt2).ravel()])
        DIISErrorNorm = float(np.linalg.norm(error))

        diis_vals_t1.append(cc.t1.copy())
        diis_vals_t2.append(cc.t2.copy())
        diis_errors.append(error.copy())

        if len(diis_vals_t1) > max_diis:
            del diis_vals_t1[0]
            del diis_vals_t2[0]
            del diis_errors[0]

        diis_size = len(diis_vals_t1) - 1

        # ----------------------------------------------------------------
        # DIIS EXTRAPOLATION
        # ----------------------------------------------------------------
        B = np.ones((diis_size + 1, diis_size + 1)) * (-1)
        B[-1, -1] = 0

        for n1, e1 in enumerate(diis_errors):
            for n2, e2 in enumerate(diis_errors):
                B[n1, n2] = np.dot(e1, e2)

        B_scale = np.abs(B[:-1, :-1]).max()
        if B_scale < 1.0e-300 or not np.isfinite(B_scale):
            print("          [DIIS] singular/degenerate B matrix; keeping plain Jacobi.")
            ci = None
        else:
            B[:-1, :-1] /= B_scale
            resid = np.zeros(diis_size + 1)
            resid[-1] = -1
            try:
                ci = np.linalg.solve(B, resid)
                if not np.all(np.isfinite(ci)):
                    ci = None
            except np.linalg.LinAlgError:
                print("          [DIIS] singular B matrix; keeping plain Jacobi.")
                ci = None

        if ci is not None:
            cc.t1[:] = 0.0
            cc.t2[:] = 0.0
            for num in range(diis_size):
                cc.t1 += ci[num] * diis_vals_t1[num + 1]
                cc.t2 += ci[num] * diis_vals_t2[num + 1]

        T_after = cc.pack()
        dT_diis = T_after - T_before
        dT_correction = dT_diis - dT_jacobi

        DIISStepNorm = float(np.linalg.norm(dT_diis))
        DIISCorrectionNorm = float(np.linalg.norm(dT_correction))

        dataset.append(dict(
            Rnorm=Rnorm,
            E=E_before,
            E_prev1=E_prev1,
            E_prev2=E_prev2,
            R_prev1=R_prev1,
            R_prev2=R_prev2,
            R_prev3=R_prev3,
            ratio1=ratio1,
            ratio2=ratio2,
            R_std=R_std,
            R_max=R_max,
            T_std=T_std,
            T_max=T_max,
            JacobiNorm=JacobiNorm,
            DIISErrorNorm=DIISErrorNorm,
            DIISStepNorm=DIISStepNorm,
            DIISCorrectionNorm=DIISCorrectionNorm,
            diis_size=diis_size,
            iter=ccsd_iter,
            is_stall=is_stall_flag,
            step_log_ratio=step_log_ratio,
        ))

        history_Rnorm.append(Rnorm)
        history_E.append(E_before)

    if len(dataset) == 0:
        print(f"No trajectory points generated for {filename_prefix}; nothing to save.")
        psi4.core.clean()
        return 0, 0

    print(
        f"{filename_prefix}: {n_stall_points}/{len(dataset)} "
        f"harvested points ({100.0 * n_stall_points / len(dataset):.1f}%) "
        f"tagged as genuine DIIS stalls."
    )

    def col(key):
        return np.array([x[key] for x in dataset])

    np.savez_compressed(
        npz_output,

        # Per-system constants
        LogDMin=np.float32(log_D_min),
        LogDMax=np.float32(log_D_max),

        # Per-iteration scalars only -- no (n_iter, n_amp) arrays.
        Rnorm=col("Rnorm").astype(np.float32),
        E=col("E").astype(np.float64),
        E_prev1=col("E_prev1").astype(np.float64),
        E_prev2=col("E_prev2").astype(np.float64),
        R_prev1=col("R_prev1").astype(np.float32),
        R_prev2=col("R_prev2").astype(np.float32),
        R_prev3=col("R_prev3").astype(np.float32),
        Ratio1=col("ratio1").astype(np.float32),
        Ratio2=col("ratio2").astype(np.float32),
        R_std=col("R_std").astype(np.float32),
        R_max=col("R_max").astype(np.float32),
        T_std=col("T_std").astype(np.float32),
        T_max=col("T_max").astype(np.float32),
        JacobiNorm=col("JacobiNorm").astype(np.float32),
        DIISErrorNorm=col("DIISErrorNorm").astype(np.float32),
        DIISStepNorm=col("DIISStepNorm").astype(np.float32),
        DIISCorrectionNorm=col("DIISCorrectionNorm").astype(np.float32),
        DIISSize=col("diis_size").astype(np.float32),
        Iter=col("iter").astype(np.int32),
        IsStall=col("is_stall").astype(bool),
        StepLogRatio=col("step_log_ratio").astype(np.float32),
    )

    print(f"Saved {len(dataset)} trajectory points to {npz_output}")

    psi4.core.clean()

    return n_stall_points, len(dataset)


def _worker(task):
    """Top-level worker so ProcessPoolExecutor can pickle it."""
    geom, tag, basis, maxiter, memory_gb, output_dir = task

    pid = os.getpid()
    print(f"[PID {pid}] STARTING {tag}", flush=True)

    npz_output = os.path.join(output_dir, f"{tag}.npz")
    if os.path.exists(npz_output):
        print(f"[PID {pid}] SKIPPING {tag} (already exists)", flush=True)
        return {
            "tag": tag,
            "status": "skipped",
            "stalls": 0,
            "points": 0,
            "message": "output .npz already exists",
        }

    try:
        stalls, points = harvest_trajectory(
            geom,
            basis,
            tag,
            maxiter=maxiter,
            memory_gb=memory_gb,
            output_dir=output_dir,
        )
        print(
            f"[PID {pid}] FINISHED {tag}: {points} points, {stalls} stalls",
            flush=True,
        )
        return {
            "tag": tag,
            "status": "completed",
            "stalls": stalls,
            "points": points,
            "message": "",
        }
    except Exception as exc:
        print(f"[PID {pid}] FAILED {tag}: {exc}", flush=True)
        try:
            psi4.core.clean()
        except Exception:
            pass
        return {
            "tag": tag,
            "status": "failed",
            "stalls": 0,
            "points": 0,
            "message": repr(exc),
        }


def build_systems():
    """Return independent molecule/geometry jobs."""
    systems = []

    for r in np.linspace(0.8, 3.2, 25):
        systems.append((f"H\nF 1 {r}\nsymmetry c1", f"hf_{r:.2f}"))

    for r in np.linspace(0.8, 3.0, 23):
        systems.append((
            f"O\nH 1 {r}\nH 1 {r} 2 104.5\nsymmetry c1",
            f"h2o_{r:.2f}",
        ))

    for r in [0.90, 1.00, 1.10, 1.20, 1.30, 1.45, 1.60, 1.80, 2.00,
              2.20, 2.50, 2.80, 3.20]:
        systems.append((f"N\nN 1 {r}\nsymmetry c1", f"n2_{r:.2f}"))

    for r in [0.95, 1.05, 1.13, 1.25, 1.40, 1.60, 1.80, 2.00, 2.30,
              2.60, 3.00]:
        systems.append((f"C\nO 1 {r}\nsymmetry c1", f"co_{r:.2f}"))

    return systems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Parallel independent CCSD/DIIS data generation with the "
            "scalar Newton-step-ratio oracle."
        )
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of independent CCSD processes (default: 2).",
    )
    parser.add_argument(
        "--memory", type=float, default=12.0,
        help="Psi4 memory in GB PER WORKER (default: 12).",
    )
    parser.add_argument(
        "--maxiter", type=int, default=100,
        help="Maximum CCSD iterations per geometry (default: 100).",
    )
    parser.add_argument(
        "--output-dir", default="cc_training_data",
        help="Directory for .npz and Psi4 .dat files.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.memory <= 0:
        parser.error("--memory must be > 0")
    if args.maxiter < 1:
        parser.error("--maxiter must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)
    systems = build_systems()

    print("\n" + "=" * 72)
    print("PARALLEL CCSD TRAINING-DATA GENERATION")
    print("=" * 72)
    print(f"Total independent systems : {len(systems)}")
    print(f"Workers                   : {args.workers}")
    print(f"Memory per worker         : {args.memory:.1f} GB")
    print(f"Maximum iterations        : {args.maxiter}")
    print(f"Output directory          : {args.output_dir}")
    print("\nEach molecule/geometry is a separate process.")
    print("Memory is PER WORKER; start with --workers 2 on a ~12 GB machine.\n")

    tasks = [
        (geom, tag, "cc-pvdz", args.maxiter, args.memory, args.output_dir)
        for geom, tag in systems
    ]

    total_stalls = 0
    total_points = 0
    completed = 0
    skipped = 0
    failed = []
    results = []

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        future_to_tag = {
            executor.submit(_worker, task): task[1]
            for task in tasks
        }

        for future in as_completed(future_to_tag):
            tag = future_to_tag[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "tag": tag,
                    "status": "failed",
                    "stalls": 0,
                    "points": 0,
                    "message": repr(exc),
                }

            results.append(result)

            if result["status"] == "completed":
                completed += 1
                total_stalls += result["stalls"]
                total_points += result["points"]
                print(
                    f"[{completed + skipped + len(failed):2d}/{len(tasks)}] "
                    f"DONE {tag}: {result['stalls']}/{result['points']} stalls",
                    flush=True,
                )
            elif result["status"] == "skipped":
                skipped += 1
                print(f"SKIP {tag}: {result['message']}", flush=True)
            else:
                failed.append(result)
                print(f"FAILED {tag}: {result['message']}", flush=True)

    print("\n" + "=" * 72)
    print("PARALLEL RUN-WIDE STALL SUMMARY")
    print("=" * 72)
    print(f"Systems attempted : {len(systems)}")
    print(f"Systems completed : {completed}")
    print(f"Systems skipped   : {skipped}")
    print(f"Systems failed    : {len(failed)}")
    print(f"Total points      : {total_points}")
    print(
        f"Total stall points: {total_stalls} "
        f"({100.0 * total_stalls / max(total_points, 1):.2f}%)"
    )

    if total_stalls == 0:
        print(
            "\nWARNING: zero genuine DIIS stalls were harvested. "
            "Consider stretching bonds further or adding harder systems."
        )
    else:
        print("\nSystems with at least one genuine stalled point:")
        for result in sorted(results, key=lambda x: x["tag"]):
            if result["status"] == "completed" and result["stalls"] > 0:
                print(
                    f"  {result['tag']:12s} "
                    f"{result['stalls']:3d}/{result['points']:3d} "
                    f"({100.0 * result['stalls'] / result['points']:.1f}%)"
                )

    if failed:
        print("\nFailed systems:")
        for result in sorted(failed, key=lambda x: x["tag"]):
            print(f"  {result['tag']:12s} {result['message']}")
