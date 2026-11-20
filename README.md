# Unified Spin-Orbital CCSD Solver (ML-Free & ML-Accelerated)

A unified spin-orbital Coupled-Cluster Singles and Doubles (CCSD) solver built on Psi4 and NumPy. It supports both **Restricted (RHF)** and **Unrestricted (UHF)** references—covering closed-shell singlets, stretched bonds, open-shell radicals, and ions—and integrates nine iterative convergence schemes under a single `job.inp` interface, including stall-triggered Newton–Krylov recovery (`st_nk_diis`) and its machine-learning-accelerated extension (`st_nkml_diis`).

---

## Repository Structure

```text
.
├── cc_training_data/         # Harvested scalar trajectory features & stall oracle targets
├── example_inputs/           # Ready-to-run sample molecular input files
├── 1_generate_data.py        # Offline CCSD trajectory harvester & oracle line-search generator
├── 2_train_accelerator.py    # Offline training script for the GRU step-size predictor
├── cc_accelerator.pt         # Pre-trained PyTorch GRU checkpoint for st_nkml_diis
├── cc_ml_common.py           # Shared 19-feature contract, CCStepPredictor model & loader
├── ccsd_master.py            # Unified CLI solver driver (all 9 methods, RHF/UHF)
├── helper_CC.py              # Spin-orbital CCSD engine, integrals, GMRES, line search & Broyden
├── job.inp                   # Default molecular geometry and solver configuration file
├── LICENSE                   # BSD-3-Clause license with Psi4NumPy and extension copyrights
├── .gitignore                # Excludes Psi4 runtime logs and Python bytecode caches
└── README.md                 # Documentation
```

| Path | Description |
| :--- | :--- |
| `ccsd_master.py` | Unified CLI driver, `job.inp` parser, RHF/UHF reference validator, DIIS/RDIIS manager, `st_nk_diis` stall-trigger state machine, and lazy-loaded `st_nkml_diis` inference runtime. |
| `helper_CC.py` | Spin-orbital CCSD engine (`helper_CCSD`), RHF and explicit UHF integral assembly, staged SCF convergence recovery (`SOSCF` $\to$ level-shifting), $\mathcal{O}(N^6)$ RHS contractions, preconditioned GMRES JFNK, line search with RHS caching, and limited-memory Type-II Broyden updates. |
| `cc_ml_common.py` | Shared 19-feature configuration (`WINDOW = 5`, `MAX_LOG_STEP_RATIO = 3.0`), `CCStepPredictor` GRU architecture, and validated checkpoint loader (`load_checkpoint`). Required only for `st_nkml_diis` and offline training. |
| `cc_accelerator.pt` | Pre-trained PyTorch GRU checkpoint for scalar logarithmic step-size prediction. Required only for `st_nkml_diis`. |
| `1_generate_data.py` | Offline data-harvesting script that runs CCSD across training geometries, detects stalls, executes oracle line searches along the GMRES Newton direction, and writes datasets to `cc_training_data/`. |
| `2_train_accelerator.py` | Offline training script that trains `CCStepPredictor` on `cc_training_data/` and exports `cc_accelerator.pt`. |
| `cc_training_data/` | Directory storing harvested scalar diagnostics and stall-associated oracle targets used to train `cc_accelerator.pt`. |
| `example_inputs/` | Collection of ready-to-run sample `.inp` files for benchmarking different molecules and solver methods. |
| `job.inp` | Default input file read when running `python3 ccsd_master.py` without `--job`. |
| `LICENSE` | BSD-3-Clause license preserving the original Psi4NumPy copyright notice alongside this project's copyright. |
| `.gitignore` | Git ignore rules excluding temporary runtime artifacts (`output.dat`, `timer.dat`, `__pycache__/`). |

> **Runtime-generated files:** Executing the solver creates Psi4 log files (`output.dat`, `timer.dat`) and Python bytecode caches (`__pycache__/`). These are automatically ignored by `.gitignore`.

---

## Prerequisites & Installation

* **Psi4 must be installed locally on your machine** with its Python API (`import psi4`) accessible in your active Python environment. `helper_CC.py` calls Psi4 directly for geometry parsing, AO integrals (`MintsHelper`), RHF/UHF SCF wavefunctions, and native CCSD energy verification.
* You do **not** need to install `psi4numpy` separately—the adapted and extended spin-orbital CCSD kernel is self-contained inside `helper_CC.py`.
* **Core dependencies (all 8 ML-free methods):** Python 3.8+, `psi4`, `numpy`, `scipy`.
* **Optional dependency (`st_nkml_diis` and offline ML training):** `torch` (PyTorch; automatically detects a CUDA GPU if available and falls back to CPU). PyTorch and `cc_ml_common` are imported lazily so all ML-free runs execute without PyTorch installed.

### Recommended Environment Setup (Conda)

```bash
conda create -n p4env psi4 numpy scipy -c conda-forge
conda activate p4env

# Optional: only required for st_nkml_diis, 1_generate_data.py, and 2_train_accelerator.py
pip install torch
```

Verify that Psi4 and the core numerical stack are accessible before running the solver:

```bash
python3 -c "import psi4, numpy, scipy; print('Psi4 version:', psi4.__version__)"
```

*(If Psi4 was compiled from source rather than installed via Conda, export its `PATH` and `PYTHONPATH` using the output of `psi4 --psiapi-path`.)*

---

## Supported Solver Methods

Select any of the nine methods via `method <name>` in `job.inp` (or override on the command line with `--method <name>`):

| Method | Description | Cost per Standard Iteration |
| :--- | :--- | :--- |
| `jacobi` | Standard fixed-point Jacobi amplitude iteration. | 1 RHS evaluation |
| `diis` | Pulay DIIS using amplitude differences ($\Delta T = T_{\mathrm{J}} - T_{\mathrm{old}}$) as error vectors. | 1 RHS evaluation |
| `rdiis` | Residual Pulay DIIS using the exact CCSD residual ($R$) directly as the error vector. | 1 RHS evaluation |
| `nk` | Pure Jacobian-Free Newton–Krylov (preconditioned GMRES) at every step. | 1 RHS + GMRES $Jv$ products |
| `nk_diis` | Unconditional JFNK steps accelerated by Pulay DIIS extrapolation. | 1 RHS + GMRES $Jv$ products |
| `st_nk_diis` | **Stall-Triggered NK–DIIS:** defaults to Pulay DIIS and triggers GMRES + verified line-search recovery only upon sustained residual stagnation; caches accepted post-NK RHS tensors. | 1 RHS (DIIS); GMRES + line search only at stalls |
| `st_nkml_diis` | **ML-Accelerated Stall-Triggered NK–DIIS:** invokes the trained GRU (`CCStepPredictor`) at stalls to predict a scalar logarithmic step-size ratio $\hat{q} = \log_{10}(s^* / s_{\mathrm{J}})$ along a shared GMRES Newton direction. Verifies against both the current state and the ordinary DIIS trial before acceptance, with NK line-search fallback and RHS caching. | 1 RHS (DIIS); shared GMRES + 1 verified trial at stalls |
| `broyden` | Limited-memory Type-II Broyden quasi-Newton update initialized with diagonal Fock denominators. | 1 RHS evaluation |
| `broyden_diis` | Limited-memory Broyden steps accelerated by Pulay DIIS extrapolation. | 1 RHS evaluation |

---

## Input File (`job.inp`)

Keywords are **case-insensitive** and placed at the top of the input file before the Psi4 geometry block. Blank lines and lines beginning with `#` are ignored. Additional templates are provided inside `example_inputs/`.

### Input Keywords

| Keyword | Default | Format / Options |
| :--- | :--- | :--- |
| `method` | `diis` | `jacobi`, `diis`, `rdiis`, `nk`, `nk_diis`, `st_nk_diis`, `st_nkml_diis`, `broyden`, `broyden_diis` |
| `model` | `cc_accelerator.pt` | Path to the trained GRU checkpoint (used only when `method` is `st_nkml_diis`; spaces in path supported) |
| `basis` | `cc-pvdz` | Any valid Psi4 basis set (e.g., `cc-pvdz`, `aug-cc-pvdz`, `6-31g*`) |
| `reference` | Auto | `RHF` or `UHF`. If omitted, closed-shell singlets default to `RHF` and open-shell systems (`mult > 1` or odd electrons) automatically select `UHF`. |
| `freeze_core` | `false` | `true` or `false` (implemented for atoms with $Z \le 18$) |
| `memory` | `12.0` | Numeric value in GB **without units** (e.g., `memory 12`, not `memory 12 GB`) |

### Example Input Files

#### 1. Closed-Shell Stretched Bond with ML Acceleration (`st_nkml_diis`)
```text
memory 12
method st_nkml_diis
model cc_accelerator.pt
reference RHF
basis cc-pvdz
freeze_core false
0 1
C
N 1 3.2
H 1 1.09 2 122.0
H 1 1.09 2 122.0 3 180.0
H 2 1.01 1 110.0 3 0.0
symmetry c1
```

#### 2. Closed-Shell Stretched Bond without ML (`st_nk_diis`)
```text
memory 12
method st_nk_diis
reference RHF
basis cc-pvdz
freeze_core false
0 1
C
N 1 3.2
H 1 1.09 2 122.0
H 1 1.09 2 122.0 3 180.0
H 2 1.01 1 110.0 3 0.0
symmetry c1
```

#### 3. Open-Shell Doublet Radical (`UHF`, `st_nk_diis`)
```text
memory 12
method st_nk_diis
reference UHF
basis cc-pvdz
freeze_core false
0 2
C
N 1 1.17
symmetry c1
```

---

## Usage

### Basic Execution (Configured Entirely via `job.inp`)
```bash
python3 ccsd_master.py
```

### Running a Sample Input from `example_inputs/`
```bash
python3 ccsd_master.py --job example_inputs/<sample_file>.inp
```

### Overriding Methods, References, and Models from the Command Line
You can benchmark all nine methods against the same input file without editing it:

```bash
python3 ccsd_master.py --job job.inp --method diis
python3 ccsd_master.py --job job.inp --method rdiis
python3 ccsd_master.py --job job.inp --method st_nk_diis
python3 ccsd_master.py --job job.inp --method st_nkml_diis --model cc_accelerator.pt
python3 ccsd_master.py --job job.inp --method broyden
python3 ccsd_master.py --job job.inp --method broyden_diis
```

To reproduce legacy standalone `nk` / `nk_diis` GMRES budgets (`restart=5`, `maxiter=5`, `tol=1e-7`):
```bash
python3 ccsd_master.py --job job.inp --method nk --nk_restart 5 --nk_maxiter 5 --nk_tol 1e-7
python3 ccsd_master.py --job job.inp --method nk_diis --nk_restart 5 --nk_maxiter 5 --nk_tol 1e-7
```

### Command-Line Flags (`ccsd_master.py`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--job` | `job.inp` | Path to the input file. |
| `--method` | `None` | Overrides the `method` keyword in `job.inp`. |
| `--reference` | `None` | Overrides `reference` (`RHF` or `UHF`). |
| `--model` | `None` | Overrides `model` in `job.inp` (path to `.pt` checkpoint for `st_nkml_diis`). |
| `--maxiter` | `100` | Maximum CCSD macro-iterations. |
| `--stall_count` | `3` | Consecutive stagnant iterations required to trigger recovery. |
| `--stall_ratio` | `0.98` | Residual ratio threshold ($r_k = \Vert R^{(k)} \Vert_2 / \Vert R^{(k-1)} \Vert_2 > \rho_{\mathrm{stall}}$). |
| `--verify_every` | Auto (`3` / `2`) | Consecutive accepted recoveries before forcing DIIS recalibration. Defaults to `3` for `st_nk_diis` and `2` for `st_nkml_diis`. |
| `--nk_restart` | Auto (`10` / `5`) | GMRES Krylov subspace restart size. Defaults to `10` (`st_nk_diis`/others) and `5` (`st_nkml_diis`). |
| `--nk_maxiter` | Auto (`10` / `5`) | Maximum GMRES iteration/restart budget. Defaults to `10` (`st_nk_diis`/others) and `5` (`st_nkml_diis`). |
| `--nk_tol` | Auto (`1e-8` / `1e-7`) | Relative GMRES linear-solver tolerance. Defaults to `1e-8` (`st_nk_diis`/others) and `1e-7` (`st_nkml_diis`). |
| `--nk_min_reduction` | `1e-3` | Minimum relative residual reduction required to accept an NK line-search step in `st_nk_diis`. |
| `--nk_failed_gmres_max_factor` | `1.0` | Maximum trial step (multiple of Jacobi step norm) when GMRES terminates incomplete (`info != 0`) in `st_nk_diis`. |
| `--min_rnorm_for_ml` | `None` | Optional residual-norm floor below which ML attempts are skipped in `st_nkml_diis`. |
| `--force_recovery_test` | `None` | **Debug only:** forces recovery attempts from the given iteration onward (not a valid timing benchmark). |

---

## Offline Data Generation & Model Training

A pre-trained checkpoint (`cc_accelerator.pt`) and its harvested dataset (`cc_training_data/`) are already included, so `st_nkml_diis` can be run immediately without retraining. To regenerate data or retrain the GRU model from scratch:

1. **Harvest stall-focused trajectory data** (writes compact 19-feature scalar histories and oracle step-size targets into `cc_training_data/`):
   ```bash
   python3 1_generate_data.py
   ```
2. **Train the scalar GRU predictor** (reads `cc_training_data/`, validates feature/window contracts from `cc_ml_common.py`, and writes `cc_accelerator.pt`):
   ```bash
   python3 2_train_accelerator.py
   ```
3. **Run live inference** using the newly trained checkpoint:
   ```bash
   python3 ccsd_master.py --job job.inp --method st_nkml_diis --model cc_accelerator.pt
   ```

---

## Output, Verification, and Reference Benchmarks

* **Convergence Criterion:** Convergence is declared when the Euclidean residual norm satisfies $\|R\|_2 < 10^{-7}$.
* **Efficiency Accounting:** Every run reports total macro-iterations, DIIS steps, NK/ML attempts and acceptances, exact total $\mathcal{O}(N^6)$ `compute_rhs()` evaluations (outer loop + GMRES + line search), final residual norm, correlation/total energy, and wall time.
* **Automatic SCF Recovery:** If standard Psi4 SCF fails to converge (common for stretched radicals), `helper_CC.py` automatically retries with `SOSCF + 20% damping`, followed by `level_shift = 0.5 + 30% damping`.
* **Psi4 Cross-Check:** Upon completion, the total CCSD energy is compared to 6 decimal places against Psi4's native CCSD (`psi4.energy("CCSD", ref_wfn=cc.wfn)` with `ccenergy__maxiter = 200` and `ccenergy__r_convergence = 1e-8`). Full Psi4 logs are written to `output.dat`.

### Reference Benchmark (Stretched $\mathrm{CH_2NH}$, $R_{\mathrm{CN}} = 3.2\text{ \AA}$, RHF/cc-pVDZ)

| Method | Converged | Iterations | Recoveries Accepted | Total `compute_rhs()` Calls | Final $\|R\|_2$ | Total CCSD Energy ($E_h$) | Wall Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `st_nk_diis` | Yes | 45 | 3 NK | 91 | $9.121 \times 10^{-8}$ | `-94.027722449851` | 60.76 |
| `st_nkml_diis` | Yes | 40 | 1 ML | 49 | $9.239 \times 10^{-8}$ | `-94.027722452379` | 33.40 |

---

## Important Notes

* Always include `symmetry c1` in the geometry block so Psi4 builds integrals without point-group symmetry blocking.
* Open-shell systems (`mult > 1` or odd electron counts) require `reference UHF`; requesting `RHF` or `ROHF` raises a validation error.
* Memory usage is checked against $5 \times n_{\mathrm{mo}}^4 \times 128 \times 10^{-9}\text{ GB}$ before integral transformation. Increase `memory` in `job.inp` for larger molecules or augmented basis sets.

---

## Attribution, Citations & License

The spin-orbital CCSD tensor-contraction kernel in `helper_CC.py` is adapted from the [Psi4NumPy](https://github.com/psi4/psi4numpy) project (Daniel G. A. Smith and Lori A. Burns) under the **BSD-3-Clause** license, and extended by Shamik Chanda, Suhita Basumallick, and Sangita Sen with generalized UHF spin-orbital support, staged SCF recovery, JFNK GMRES, RDIIS, Stall-Triggered NK–DIIS (`st_nk_diis`), ML-accelerated scalar step-size control (`st_nkml_diis`), and limited-memory Type-II Broyden solvers.

If you use this codebase in published work, please cite:

1. **Psi4NumPy:** D. G. A. Smith, L. A. Burns, D. A. Sirianni, D. R. Nascimento, A. Kumar, A. M. James, J. B. Schriber, T. Zhang, B. Zhang, A. S. Abbott, E. J. Berquist, M. H. Lechner, L. A. Cunha, A. G. Heide, J. M. Waldrop, T. Y. Takeshita, A. Alenaizan, D. Neuhauser, R. A. King, A. C. Simmonett, J. M. Turney, H. F. Schaefer, F. A. Evangelista, A. E. DePrince, T. D. Crawford, K. Patkowski, and C. D. Sherrill, *"Psi4NumPy: An Interactive Quantum Chemistry Programming Environment for Reference Implementations and Rapid Development,"* *J. Chem. Theory Comput.* **2018**, *14* (7), 3504–3511. DOI: `10.1021/acs.jctc.8b00286`.
2. **Spin-Orbital CCSD Formulation:** J. F. Stanton, J. Gauss, J. D. Watts, and R. J. Bartlett, *"A direct product decomposition approach for symmetry exploitation in many-body methods. I. Energy calculations,"* *J. Chem. Phys.* **1991**, *94*, 4334–4345. DOI: `10.1063/1.460620`.

Distributed under the **BSD-3-Clause** License (see `LICENSE`).
