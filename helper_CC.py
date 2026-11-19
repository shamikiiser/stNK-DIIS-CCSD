"""
Helper classes and functions for spin-orbital Coupled-Cluster (CCSD) module.
Supports both Restricted (RHF) and Unrestricted (UHF) references for
closed-shell singlets, open-shell doublets/triplets, and radical anions.

References:
- DPD formulation of CCSD equations: [Stanton:1991:4334]
- Psi4NumPy project: http://github.com/CrawfordGroup/ProgrammingProjects
"""

__authors__   = ["Daniel G. A. Smith", "Lori A. Burns", "Shamik Chanda", "Suhita Basumalick", "Sangita Sen"]
__credits__   = ["The Psi4NumPy Developers"]
__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers; extended 2025-2026"
__license__   = "BSD-3-Clause"
__date__    = "2026"

import time
import numpy as np
import psi4
from scipy.sparse.linalg import gmres, LinearOperator

try:
    from psi4.driver.p4util.exceptions import SCFConvergenceError
except ImportError:
    SCFConvergenceError = getattr(psi4, "SCFConvergenceError", Exception)


def ndot(input_string, op1, op2, prefactor=None):
    """
    Fast N-dimensional tensor contraction helper (mini-DPD library).
    Example: ndot('abcd,cdef->abef', arr1, arr2)
    """
    inp, output_ind = input_string.split('->')
    input_left, input_right = inp.split(',')

    size_dict = {}
    for s, size in zip(input_left, op1.shape):
        size_dict[s] = size
    for s, size in zip(input_right, op2.shape):
        size_dict[s] = size

    set_left = set(input_left)
    set_right = set(input_right)
    set_out = set(output_ind)

    idx_removed = (set_left | set_right) - set_out
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed

    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s), )
        right_pos += (input_right.find(s), )
    tdot_axes = (left_pos, right_pos)

    tdot_result = input_left + input_right
    for s in idx_removed:
        tdot_result = tdot_result.replace(s, '')

    rs = len(idx_removed)
    dim_left, dim_right, dim_removed = 1, 1, 1
    for key, size in size_dict.items():
        if key in keep_left:
            dim_left *= size
        if key in keep_right:
            dim_right *= size
        if key in idx_removed:
            dim_removed *= size

    shape_result = tuple(size_dict[x] for x in tdot_result)
    used_einsum = False

    if input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed), op2.reshape(dim_removed, dim_right))
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T, op2.reshape(dim_right, dim_removed).T)
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed), op2.reshape(dim_right, dim_removed).T)
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T, op2.reshape(dim_removed, dim_right))
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        new_view = np.einsum(input_string, op1, op2, optimize=True)
        used_einsum = True
    else:
        new_view = np.tensordot(op1, op2, axes=tdot_axes)

    if (new_view.shape != shape_result) and not used_einsum:
        if len(shape_result) > 0:
            new_view = new_view.reshape(shape_result)
        else:
            new_view = np.squeeze(new_view)

    if prefactor is not None:
        new_view *= prefactor

    if used_einsum or tdot_result == output_ind:
        return new_view
    return np.einsum(tdot_result + '->' + output_ind, new_view)


def _run_gmres(A, b, M, restart, tol, maxiter, callback):
    """Wrapper supporting both newer SciPy (rtol/atol) and older SciPy (tol)."""
    try:
        return gmres(A, b, M=M, restart=restart, rtol=tol, atol=0.0, maxiter=maxiter, callback=callback)
    except TypeError:
        return gmres(A, b, M=M, restart=restart, tol=tol, maxiter=maxiter, callback=callback)


class helper_CCSD(object):
    def __init__(self, mol, freeze_core=False, memory=2, reference=None):
        """
        Initializes the helper_CCSD object for RHF or UHF references.
        Includes staged SCF convergence fallbacks (SOSCF -> level-shift + damping).
        """
        print("\nInitializing CCSD object...\n")
        time_init = time.time()

        if reference is None:
            self.reference = "UHF" if mol.multiplicity() > 1 else "RHF"
        else:
            self.reference = reference.strip().upper()

        print(f"Computing {self.reference} reference.")
        psi4.core.set_active_molecule(mol)
        psi4.set_options({
            'scf__scf_type': 'pk',
            'scf__reference': self.reference,
            'scf__e_convergence': 1e-10,
            'scf__d_convergence': 1e-10,
            'scf__maxiter': 150,
            'scf__soscf': False,
            'scf__level_shift': 0.0,
            'scf__damping_percentage': 0.0,
            'ccenergy__freeze_core': 'true' if freeze_core else 'false',
        })

        try:
            self.rhf_e, self.wfn = psi4.energy('SCF', return_wfn=True)
        except SCFConvergenceError:
            print("Standard SCF failed to converge; retrying with SOSCF + damping...")
            psi4.set_options({
                'scf__soscf': True,
                'scf__damping_percentage': 20.0,
                'scf__maxiter': 300,
            })
            try:
                self.rhf_e, self.wfn = psi4.energy('SCF', return_wfn=True)
            except SCFConvergenceError:
                print("SOSCF failed to converge; retrying with level_shift=0.5 + damping...")
                psi4.set_options({
                    'scf__soscf': False,
                    'scf__level_shift': 0.5,
                    'scf__damping_percentage': 30.0,
                    'scf__maxiter': 500,
                })
                self.rhf_e, self.wfn = psi4.energy('SCF', return_wfn=True)

        print(f"{self.reference} Final Energy                          {self.rhf_e: 16.10f}\n")

        self.ccsd_corr_e = 0.0
        self.ccsd_e = 0.0
        self.memory = memory
        self.nresidual = 0
        self.nrhs = 0
        self._cached_rhs = None

        mints = psi4.core.MintsHelper(self.wfn.basisset())
        self.nmo = self.wfn.nmo()
        self.nalpha = self.wfn.nalpha()
        self.nbeta = self.wfn.nbeta()

        if freeze_core:
            Zlist = np.array([mol.Z(x) for x in range(mol.natom())])
            self.nfzc = int(np.sum(Zlist > 2) + np.sum(Zlist > 10) * 4)
            if np.any(Zlist > 18):
                raise ValueError("Frozen core for Z > 18 not yet implemented.")
            if self.nfzc > min(self.nalpha, self.nbeta):
                raise ValueError(
                    f"Frozen core count ({self.nfzc}) exceeds occupied orbitals "
                    f"(nalpha={self.nalpha}, nbeta={self.nbeta})."
                )
            print(f"Freezing {self.nfzc} spatial core orbitals.")
        else:
            self.nfzc = 0

        # Check memory footprint BEFORE allocating 4D spin-orbital integrals
        print('Starting AO -> spin-orbital MO transformation...')
        ERI_Size = (self.nmo ** 4) * 128.0e-9
        memory_footprint = ERI_Size * 5.0
        if memory_footprint > self.memory:
            raise MemoryError(
                f"Estimated memory utilization ({memory_footprint:4.2f} GB) exceeds "
                f"allotted limit of {self.memory:4.2f} GB."
            )
        print(f"Size of the ERI tensor is {ERI_Size:4.2f} GB, {self.nmo} basis functions.")

        H_core = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        self.nso = 2 * self.nmo
        self.nfzc_so = 2 * self.nfzc
        nocc_a = self.nalpha - self.nfzc
        nocc_b = self.nbeta - self.nfzc
        nvirt_a = self.nmo - self.nalpha
        nvirt_b = self.nmo - self.nbeta

        self.nocc = nocc_a + nocc_b
        self.nvirt = nvirt_a + nvirt_b

        if self.reference == 'RHF':
            # Interleaved ordering already groups [fzc | active_occ | vir] contiguously
            self.C = self.wfn.Ca()
            npC = np.asarray(self.C)
            eps_spatial = np.asarray(self.wfn.epsilon_a())

            H_spatial = npC.T @ H_core @ npC
            H_so = np.repeat(np.repeat(H_spatial, 2, axis=0), 2, axis=1)
            spin_ind = np.arange(self.nso, dtype=int) % 2
            H_so *= (spin_ind.reshape(-1, 1) == spin_ind)

            self.eps = np.repeat(eps_spatial, 2)
            self.MO = np.asarray(mints.mo_spin_eri(self.C, self.C))
        else:
            # Direct UHF spin-orbital assembly in [fzc_a, fzc_b | occ_a, occ_b | vir_a, vir_b]
            Ca = self.wfn.Ca()
            Cb = self.wfn.Cb()
            npCa = np.asarray(Ca)
            npCb = np.asarray(Cb)

            eps_a = np.asarray(self.wfn.epsilon_a())
            eps_b = np.asarray(self.wfn.epsilon_b())

            n_all_occ = self.nfzc_so + self.nocc
            idx_a = np.concatenate([
                np.arange(0, self.nfzc),
                np.arange(self.nfzc_so, self.nfzc_so + nocc_a),
                np.arange(n_all_occ, n_all_occ + nvirt_a),
            ])
            idx_b = np.concatenate([
                np.arange(self.nfzc, self.nfzc_so),
                np.arange(self.nfzc_so + nocc_a, n_all_occ),
                np.arange(n_all_occ + nvirt_a, self.nso),
            ])

            self.eps = np.empty(self.nso, dtype=np.float64)
            self.eps[idx_a] = eps_a
            self.eps[idx_b] = eps_b

            H_so = np.zeros((self.nso, self.nso), dtype=np.float64)
            H_so[np.ix_(idx_a, idx_a)] = npCa.T @ H_core @ npCa
            H_so[np.ix_(idx_b, idx_b)] = npCb.T @ H_core @ npCb

            # Antisymmetrized physicists' notation: <pq||rs> = (pr|qs) - (ps|qr)
            self.MO = np.zeros((self.nso, self.nso, self.nso, self.nso), dtype=np.float64)

            I_aaaa = np.asarray(mints.mo_eri(Ca, Ca, Ca, Ca)).transpose(0, 2, 1, 3)
            self.MO[np.ix_(idx_a, idx_a, idx_a, idx_a)] = I_aaaa - I_aaaa.transpose(0, 1, 3, 2)
            del I_aaaa

            I_bbbb = np.asarray(mints.mo_eri(Cb, Cb, Cb, Cb)).transpose(0, 2, 1, 3)
            self.MO[np.ix_(idx_b, idx_b, idx_b, idx_b)] = I_bbbb - I_bbbb.transpose(0, 1, 3, 2)
            del I_bbbb

            I_abab = np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb)).transpose(0, 2, 1, 3)
            self.MO[np.ix_(idx_a, idx_b, idx_a, idx_b)] = I_abab
            self.MO[np.ix_(idx_a, idx_b, idx_b, idx_a)] = -I_abab.transpose(0, 1, 3, 2)
            self.MO[np.ix_(idx_b, idx_a, idx_b, idx_a)] = I_abab.transpose(1, 0, 3, 2)
            self.MO[np.ix_(idx_b, idx_a, idx_a, idx_b)] = -I_abab.transpose(1, 0, 2, 3)
            del I_abab

        self.slice_nfzc = slice(0, self.nfzc_so)
        self.slice_o = slice(self.nfzc_so, self.nfzc_so + self.nocc)
        self.slice_v = slice(self.nfzc_so + self.nocc, self.nso)
        self.slice_a = slice(0, self.nso)
        self.slice_dict = {
            'f': self.slice_nfzc,
            'o': self.slice_o,
            'v': self.slice_v,
            'a': self.slice_a,
        }

        # Fock matrix includes sum over all occupied spin-orbitals (frozen + active)
        slice_all_occ = slice(0, self.nfzc_so + self.nocc)
        self.F = H_so + np.einsum('pmqm->pq', self.MO[:, slice_all_occ, :, slice_all_occ], optimize=True)

        print('\nBuilding denominator arrays...')
        Focc = np.diag(self.F)[self.slice_o]
        Fvir = np.diag(self.F)[self.slice_v]

        self.Dia = Focc.reshape(-1, 1) - Fvir
        self.Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvir.reshape(-1, 1) - Fvir

        print('Building initial MP2 guess...')
        self.t1 = np.zeros((self.nocc, self.nvirt), dtype=np.float64)
        self.t2 = self.MO[self.slice_o, self.slice_o, self.slice_v, self.slice_v] / self.Dijab

        self.current_R = None
        self.current_Rnorm = np.inf

        # Limited-memory Broyden (Type-II) state
        self.broyden_max = 8
        self._broyden_u = []
        self._broyden_w = []
        self._broyden_prev_T = None
        self._broyden_prev_R = None

        print(f"\n..initialized CCSD in {time.time() - time_init:.3f} seconds.\n")

    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise ValueError(f"get_MO: string {string} must have 4 elements.")
        return self.MO[
            self.slice_dict[string[0]],
            self.slice_dict[string[1]],
            self.slice_dict[string[2]],
            self.slice_dict[string[3]],
        ]

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise ValueError(f"get_F: string {string} must have 2 elements.")
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def build_tilde_tau(self):
        ttau = self.t2.copy()
        tmp = 0.5 * np.einsum('ia,jb->ijab', self.t1, self.t1, optimize=True)
        ttau += tmp
        ttau -= tmp.swapaxes(2, 3)
        return ttau

    def build_tau(self):
        ttau = self.t2.copy()
        tmp = np.einsum('ia,jb->ijab', self.t1, self.t1, optimize=True)
        ttau += tmp
        ttau -= tmp.swapaxes(2, 3)
        return ttau

    def build_Fae(self):
        Fae = self.get_F('vv').copy()
        Fae[np.diag_indices_from(Fae)] = 0.0
        Fae -= ndot('me,ma->ae', self.get_F('ov'), self.t1, prefactor=0.5)
        Fae += ndot('mf,mafe->ae', self.t1, self.get_MO('ovvv'))
        Fae -= ndot('mnaf,mnef->ae', self.build_tilde_tau(), self.get_MO('oovv'), prefactor=0.5)
        return Fae

    def build_Fmi(self):
        Fmi = self.get_F('oo').copy()
        Fmi[np.diag_indices_from(Fmi)] = 0.0
        Fmi += ndot('ie,me->mi', self.t1, self.get_F('ov'), prefactor=0.5)
        Fmi += ndot('ne,mnie->mi', self.t1, self.get_MO('ooov'))
        Fmi += ndot('inef,mnef->mi', self.build_tilde_tau(), self.get_MO('oovv'), prefactor=0.5)
        return Fmi

    def build_Fme(self):
        Fme = self.get_F('ov').copy()
        Fme += ndot('nf,mnef->me', self.t1, self.get_MO('oovv'))
        return Fme

    def build_Wmnij(self):
        Wmnij = self.get_MO('oooo').copy()
        Pij = ndot('je,mnie->mnij', self.t1, self.get_MO('ooov'))
        Wmnij += Pij
        Wmnij -= Pij.swapaxes(2, 3)
        Wmnij += ndot('ijef,mnef->mnij', self.build_tau(), self.get_MO('oovv'), prefactor=0.25)
        return Wmnij

    def build_Wabef(self):
        Wabef = self.get_MO('vvvv').copy()
        Pab = ndot('mb,amef->abef', self.t1, self.get_MO('vovv'))
        Wabef -= Pab
        Wabef += Pab.swapaxes(0, 1)
        Wabef += ndot('mnab,mnef->abef', self.build_tau(), self.get_MO('oovv'), prefactor=0.25)
        return Wabef

    def build_Wmbej(self):
        Wmbej = self.get_MO('ovvo').copy()
        Wmbej += ndot('jf,mbef->mbej', self.t1, self.get_MO('ovvv'))
        Wmbej -= ndot('nb,mnej->mbej', self.t1, self.get_MO('oovo'))
        tmp = 0.5 * self.t2 + np.einsum('jf,nb->jnfb', self.t1, self.t1, optimize=True)
        Wmbej -= ndot('jnfb,mnef->mbej', tmp, self.get_MO('oovv'))
        return Wmbej

    def compute_rhs(self):
        """Computes CCSD T1 and T2 right-hand sides (O(N^6) contraction)."""
        self.nrhs += 1

        Fae = self.build_Fae()
        Fmi = self.build_Fmi()
        Fme = self.build_Fme()

        rhs_T1 = self.get_F('ov').copy()
        rhs_T1 += ndot('ie,ae->ia', self.t1, Fae)
        rhs_T1 -= ndot('ma,mi->ia', self.t1, Fmi)
        rhs_T1 += ndot('imae,me->ia', self.t2, Fme)
        rhs_T1 -= ndot('nf,naif->ia', self.t1, self.get_MO('ovov'))
        rhs_T1 -= ndot('imef,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=0.5)
        rhs_T1 -= ndot('mnae,nmei->ia', self.t2, self.get_MO('oovo'), prefactor=0.5)

        rhs_T2 = self.get_MO('oovv').copy()

        tmp = Fae - 0.5 * ndot('mb,me->be', self.t1, Fme)
        Pab = ndot('ijae,be->ijab', self.t2, tmp)
        rhs_T2 += Pab
        rhs_T2 -= Pab.swapaxes(2, 3)

        tmp = Fmi + 0.5 * ndot('je,me->mj', self.t1, Fme)
        Pij = ndot('imab,mj->ijab', self.t2, tmp)
        rhs_T2 -= Pij
        rhs_T2 += Pij.swapaxes(0, 1)

        tmp_tau = self.build_tau()
        Wmnij = self.build_Wmnij()
        Wabef = self.build_Wabef()
        rhs_T2 += ndot('mnab,mnij->ijab', tmp_tau, Wmnij, prefactor=0.5)
        rhs_T2 += ndot('ijef,abef->ijab', tmp_tau, Wabef, prefactor=0.5)

        tmp = ndot('ie,mbej->mbij', self.t1, self.get_MO('ovvo'))
        tmp = ndot('ma,mbij->ijab', self.t1, tmp)
        Wmbej = self.build_Wmbej()
        Pijab = ndot('imae,mbej->ijab', self.t2, Wmbej) - tmp

        rhs_T2 += Pijab
        rhs_T2 -= Pijab.swapaxes(2, 3)
        rhs_T2 -= Pijab.swapaxes(0, 1)
        rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)

        Pij = ndot('ie,abej->ijab', self.t1, self.get_MO('vvvo'))
        rhs_T2 += Pij
        rhs_T2 -= Pij.swapaxes(0, 1)

        Pab = ndot('ma,mbij->ijab', self.t1, self.get_MO('ovoo'))
        rhs_T2 -= Pab
        rhs_T2 += Pab.swapaxes(2, 3)

        return rhs_T1, rhs_T2

    def residual_from_rhs(self, rhs1, rhs2):
        R1 = rhs1 - self.Dia * self.t1
        R2 = rhs2 - self.Dijab * self.t2
        R = np.concatenate((R1.ravel(), R2.ravel()))
        self.current_R = R
        self.current_Rnorm = float(np.linalg.norm(R))
        return R

    def update(self):
        rhs_T1, rhs_T2 = self.compute_rhs()
        self.t1 = rhs_T1 / self.Dia
        self.t2 = rhs_T2 / self.Dijab

    def residual(self):
        self.nresidual += 1
        rhs1, rhs2 = self.compute_rhs()
        R1 = rhs1 - self.Dia * self.t1
        R2 = rhs2 - self.Dijab * self.t2
        return R1, R2

    def update_jacobi(self):
        R1, R2 = self.residual()
        R = np.concatenate((R1.ravel(), R2.ravel()))
        self.current_R = R
        self.current_Rnorm = float(np.linalg.norm(R))
        self.t1 += R1 / self.Dia
        self.t2 += R2 / self.Dijab

    def pack(self):
        return np.concatenate((self.t1.ravel(), self.t2.ravel()))

    def unpack(self, x):
        nt1 = self.t1.size
        self.t1[:] = x[:nt1].reshape(self.t1.shape)
        self.t2[:] = x[nt1:].reshape(self.t2.shape)

    def pack_residual(self):
        R1, R2 = self.residual()
        return np.concatenate((R1.ravel(), R2.ravel()))

    def pack_diag(self):
        return np.concatenate((self.Dia.ravel(), self.Dijab.ravel()))

    def Jv(self, v):
        T0 = self.Tref
        vnorm = np.linalg.norm(v)
        if vnorm < 1e-14:
            return np.zeros_like(v)

        Tnorm = np.linalg.norm(T0)
        h = 1e-7 * (1.0 + Tnorm) / vnorm

        self.unpack(T0 + h * v)
        try:
            Rp = self.pack_residual()
        finally:
            self.unpack(T0)
        return (Rp - self.current_R) / h

    def newton_direction(self, R=None, restart=10, tol=1e-8, maxiter=10, sigma=0.2, return_raw=False):
        """
        Computes the Jacobian-Free Newton-Krylov step using preconditioned GMRES.
        Reuses R if provided to avoid an extra O(N^6) evaluation.

        Returns:
          - If return_raw=False: (v_unit, info, raw_newton_norm, gmres_iters)
          - If return_raw=True : (v_unit, dT_raw, info, raw_newton_norm, gmres_iters)
        """
        if R is None:
            R = self.pack_residual()

        self.current_R = R
        self.current_Rnorm = float(np.linalg.norm(R))
        self.Tref = self.pack()

        n = R.size
        A = LinearOperator((n, n), matvec=self.Jv)
        gmres_count = [0]

        def cb(_):
            gmres_count[0] += 1

        diag = self.pack_diag() - sigma
        diag = np.where(np.abs(diag) < 1e-6, np.where(diag >= 0.0, 1e-6, -1e-6), diag)
        Minv = LinearOperator((n, n), matvec=lambda x: x / diag)

        try:
            dT_raw, info = _run_gmres(
                A, -R, M=Minv, restart=restart, tol=tol, maxiter=maxiter, callback=cb
            )
        except Exception as exc:
            print(f"          [NK] GMRES exception: {exc}")
            dT_raw, info = None, -1
        finally:
            self.unpack(self.Tref)
            del self.Tref

        raw_norm = float(np.linalg.norm(dT_raw)) if dT_raw is not None else np.inf
        if dT_raw is None or not np.isfinite(raw_norm) or raw_norm < 1e-14:
            if return_raw:
                return None, None, info, raw_norm, gmres_count[0]
            return None, info, raw_norm, gmres_count[0]

        v_unit = dT_raw / raw_norm
        if return_raw:
            return v_unit, dT_raw, info, raw_norm, gmres_count[0]
        return v_unit, info, raw_norm, gmres_count[0]

    def line_search_along(
        self,
        v_nk,
        R_before=None,
        trial_scales=(0.125, 0.25, 0.5, 1.0, 1.5, 2.0),
        min_relative_reduction=1e-3,
    ):
        """
        Evaluates trial steps T_before + s * v_nk along unit Newton direction v_nk.
        Caches the accepted (rhs1, rhs2) so the next solver iteration costs zero extra RHS calls.
        """
        T_before = self.pack()
        if R_before is None:
            R_before = self.current_R if self.current_R is not None else self.pack_residual()
        Rnorm_before = float(np.linalg.norm(R_before))

        best_s = 0.0
        best_norm = np.inf
        best_T = None
        best_rhs = None
        best_R = None

        for s in trial_scales:
            T_trial = T_before + s * v_nk
            self.unpack(T_trial)
            self.nresidual += 1
            rhs1_t, rhs2_t = self.compute_rhs()
            R1_t = rhs1_t - self.Dia * self.t1
            R2_t = rhs2_t - self.Dijab * self.t2
            R_t = np.concatenate((R1_t.ravel(), R2_t.ravel()))
            trial_norm = float(np.linalg.norm(R_t))

            if np.isfinite(trial_norm) and trial_norm < best_norm:
                best_norm = trial_norm
                best_s = float(s)
                best_T = T_trial.copy()
                best_rhs = (rhs1_t, rhs2_t)
                best_R = R_t

        rel_reduction = (Rnorm_before - best_norm) / max(Rnorm_before, 1e-300)
        accepted = (
            best_T is not None
            and np.isfinite(best_norm)
            and best_norm < Rnorm_before
            and rel_reduction >= min_relative_reduction
        )

        if accepted:
            self.unpack(best_T)
            self.current_R = best_R
            self.current_Rnorm = best_norm
            self._cached_rhs = best_rhs
        else:
            self.unpack(T_before)
            self._cached_rhs = None

        return best_s, best_norm, accepted

    def update_nk(self, R_before=None, restart=5, tol=1e-7, maxiter=5, sigma=0.2):
        """Legacy pure-NK update helper (reuses R_before when provided)."""
        _, dT_raw, info, raw_norm, gmres_iters = self.newton_direction(
            R=R_before, restart=restart, tol=tol, maxiter=maxiter, sigma=sigma, return_raw=True
        )
        if dT_raw is not None:
            self.unpack(self.pack() + dT_raw)
        return dT_raw, info, raw_norm, gmres_iters

    def update_broyden(self, R_before=None, dT_jacobi=None):
        """
        Limited-memory Broyden (Type-II) quasi-Newton update with diagonal
        initial inverse-Jacobian H0 = -diag^{-1}. Zero extra RHS cost when
        R_before and dT_jacobi are passed from ccsd_master.py.
        """
        T_current = self.pack()
        diag = self.pack_diag()

        if R_before is None:
            R_before = self.pack_residual()
        if dT_jacobi is None:
            dT_jacobi = R_before / diag

        self.current_R = R_before
        self.current_Rnorm = float(np.linalg.norm(R_before))

        # Update secant memory from previous iteration: H_k y_{k-1} = s_{k-1}
        if self._broyden_prev_T is not None and self._broyden_prev_R is not None:
            s_prev = T_current - self._broyden_prev_T
            y_prev = R_before - self._broyden_prev_R
            yy = float(np.dot(y_prev, y_prev))

            if np.isfinite(yy) and yy > 1e-24:
                # Apply H_{k-1} to y_prev (H0 = -1 / diag)
                Hy = -y_prev / diag
                for u_j, w_j in zip(self._broyden_u, self._broyden_w):
                    Hy += u_j * float(np.dot(w_j, y_prev))

                u_new = s_prev - Hy
                w_new = y_prev / yy

                if np.all(np.isfinite(u_new)) and np.all(np.isfinite(w_new)):
                    self._broyden_u.append(u_new)
                    self._broyden_w.append(w_new)
                    if len(self._broyden_u) > self.broyden_max:
                        self._broyden_u.pop(0)
                        self._broyden_w.pop(0)

        # Compute Broyden step: dT = -H_k R_k = dT_jacobi - sum_j u_j (w_j^T R_k)
        dT_broyden = dT_jacobi.copy()
        for u_j, w_j in zip(self._broyden_u, self._broyden_w):
            dT_broyden -= u_j * float(np.dot(w_j, R_before))

        if not np.all(np.isfinite(dT_broyden)):
            dT_broyden = dT_jacobi

        self._broyden_prev_T = T_current.copy()
        self._broyden_prev_R = R_before.copy()
        self.unpack(T_current + dT_broyden)

    def compute_corr_energy(self):
        """Computes CCSD correlation energy from current T1 and T2 amplitudes."""
        CCSDcorr_E = np.einsum('ia,ia->', self.get_F('ov'), self.t1, optimize=True)
        CCSDcorr_E += 0.25 * np.einsum('ijab,ijab->', self.get_MO('oovv'), self.t2, optimize=True)
        CCSDcorr_E += 0.5 * np.einsum('ijab,ia,jb->', self.get_MO('oovv'), self.t1, self.t1, optimize=True)

        self.ccsd_corr_e = float(CCSDcorr_E)
        self.ccsd_e = self.rhf_e + self.ccsd_corr_e
        return self.ccsd_corr_e
