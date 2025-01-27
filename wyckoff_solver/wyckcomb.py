r"""
# Wyckoff Combinations

Let $\bm{p}=(\bm p_i)$ be repeatabilities of sites not repeatable,
and $\bm{q}=(\bm q_j)$ be repeatabilities of sites repeatable,
and $\bm{u}=(\bm u_k)$ be number of each element.

Let $\hat{\bm{p}}$ and $\hat{\bm{q}}$ are solves.

To reduce number of variable, sites with same multiplicity in $\bm{b}$ can be merged together.

Equations:
$$
\bm{p}_i\hat{\bm{p}}_{ki} + \bm{q}_j\hat{\bm{q}}_{kj} = \bm u_k, \quad k=0,1,... \\
0 \le \hat{\bm{p}}_{ki} \le 1, \quad k,i=0,1,... \\
\sum_{k}\hat{\bm{p}}_{ki} \le 1, \quad j=0,1,... \\
\hat{\bm{q}}_{kj} \ge 0 \quad k,j=0,1,... \\
$$

-----------

Discuss $\bm{p}$ first. Let $\beta=\text{gcd}(\bm{q})$,

$$
\beta | \bm{u}_k - \bm{p}_i\hat{\bm{p}}_{ki}, \quad k=0,1,... \\
0 \le \hat{\bm{p}}_{ki} \le 1, \quad k,i=0,1,... \\
\sum_{k}\hat{\bm{p}}_{ki} \le 1, \quad j=0,1,... \\
$$

calling exclusive binary divisible linear programming problem,

we can easily solve all $\hat{\bm{p}}$. We note $\bm{u}^p=(\bm{p}_i\hat{\bm{p}}_{ki})$, and $\bm{u}^q=\bm{u} - \bm{u}^p$

----------

Then discuss $\bm{q}$, the constrains becomes

$$
\bm{q}_j\hat{\bm{q}}_{kj} = \bm{u}^b_k, \quad k=0,1,...\\
\hat{\bm{q}}_{kj} \ge 0, \quad k,j=0,1,... \\
$$

For simplarity, we can collect all $\bm{q}_j$ which have the same value,
then reduce $\bm{q}_j$ and $\bm{u}^q_k$ by $\beta$, resulting in $\bm{q}^{*}_j$ and $\bm{u}^{*q}_k$.

This operation will not change the solutions and the form of the equations.

The constrains become

$$
\bm{q}^{*}_j\hat{\bm{q}}^{*}_{kj} = \bm{u}^{*q}_k, \quad k=0,1,...\\
\hat{\bm{q}}^{*}_{kj} \ge 0, \quad k,j=0,1,... \\
\bm{q}^{*}_0 > \bm{q}^{*}_1 > ... \\
$$

these are $k$ independent non-negative linear diophantine problems.

The generation function is

$$
G(x) = \prod_{j} (1 + x^{\bm{q}^{*}_j} + x^{2\bm{q}^{*}_j} + ...) \
$$

the number of sulutions for $\bm{u}^{*q}_{k}$ equals to the coefficients of term $x^{\bm{u}^{*q}_{k}}$ in $G(x)$.

Using dynamic programming method to count the number of solutions, and backtracing method to find all solutions.

Note that reversed-sorted $\bm{q}^{*}$ can improve performance.

---------
Last, for each solutions, unpacking the collected same values by each solution term falls back to a series of integer partition problems.

Assume the original $\bm{q}_j$ has $v$ repeated times, the target constant term is $\hat{\bm{q}}^{*}_{kj}$.

The equation is

$$
\sum_{l=1}^v{\hat{\bm{q}}_{kjl}} = \hat{\bm{q}}^{*}_{kj} \\
\hat{\bm{q}}_{kjl} \ge 0 \\
$$

It also is a non-negative linear diophantine equation with all coeficients are 1. The generation function

$$
G(x) = (1 + x + x^2 + ... + x^{\hat{\bm{q}}^{*}_{kj}})^v
$$

the number of solutions is $\text{C}_{v - 1 + \hat{\bm{q}}^{*}_{kj}}^{v-1}$.

Also use backtracking method to get all solutions.
"""

import math
import threading
import warnings
from inspect import signature
from itertools import accumulate, chain, compress, groupby, islice, product, repeat
from queue import Queue
from typing import Any
from pathlib import Path

try:
    from itertools import batched  # type: ignore
except ImportError:

    def batched(iterable, n):  # type: ignore
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError('n must be at least one')
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


import numpy as np
import numpy.typing as npt
import scipy
import scipy.sparse
from scipy.sparse import csr_array

from calypso.math.symmetry.group import Group

from .cache import PROJECT_CACHEDIR
from .linprog import ExBinDivLinSolver, NonNegLinDiophSolver
from .utils.hash import hash_uints


def argsort(arr, /, *, key=None, reverse=False):
    def get_sort_key(item):
        return item[1] if key is None else key(item[1])

    indexed_arr = list(enumerate(arr))
    sorted_pairs = sorted(indexed_arr, key=get_sort_key, reverse=reverse)
    sorted_indices, sorted_arr = list(map(list, zip(*sorted_pairs)))
    return sorted_indices, sorted_arr


class WyckCombSolver:
    """Wyckoff Combination Solver

    Parameters
    ----------
    group: int
        Space group number from [1,230].
    num_atoms : list[int] | npt.NDArray[np.int_]
        List of number of atoms of each elements.

    Examples
    --------
    >>> from calypso.math.symmetry.wyckcomb import WyckCombSolver
    >>> group = 23
    >>> num_atoms = [8, 4, 16]
    >>> solver = WyckCombSolver(group, num_atoms)
    >>> solver.A
    (8, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2)
    >>> solver.p
    [2, 2, 2, 2]
    >>> solver.q
    [8, 4, 4, 4, 4, 4, 4]
    >>> solver.u
    [8, 4, 16]
    >>> solver.num_solutions
    150840
    >>> solver.get_solution(0)
    array([[0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0]], dtype=uint16)
    >>> solver.self_verify()
    True
    >>> solver.self_verify(strict=True)
    # This may take for a long time if total number of solutions is too large
    >>> solver.verify(solver.get_solution(0))
    True
    """

    def __init__(self, group: int, num_atoms: list[int] | npt.NDArray[np.int_], dim=3):
        sorted_indices, sorted_arr = argsort(num_atoms, reverse=True)
        # map from sorted indices back to original indices
        self.inverse_indices = sorted(
            range(len(sorted_indices)), key=lambda i: sorted_indices[i]
        )
        self.dim = dim
        self.group = Group(group, dim=dim)
        self.descending_solver = DescWyckCombSolver(group, num_atoms, dim=dim)
        self.is_sorted = np.all(np.subtract(sorted_arr, num_atoms) == 0)

        self.u = num_atoms
        self.A = self.descending_solver.A
        self.p = self.descending_solver.p  # irrepeatable
        self.q = self.descending_solver.q  # repeatable
        self.pindices = self.descending_solver.pindices
        self.qindices = self.descending_solver.qindices

        self.np = len(self.p)
        self.nq = len(self.q)
        self.nu = len(self.u)

    @property
    def num_solutions(self):
        return self.descending_solver.num_solutions

    @property
    def has_solutions(self) -> bool:
        return self.descending_solver.has_solutions

    def get_solution(self, index: int) -> np.ndarray:
        sol = self.descending_solver.get_solution(index)
        sol = sol if self.is_sorted else np.take(sol, self.inverse_indices, axis=-2)
        return sol

    def self_verify(self, strict=False) -> bool:
        return self.descending_solver.self_verify(strict=strict)

    def verify(self, solution):
        if np.any(solution < 0):
            warnings.warn(f"Some solution is negative!", stacklevel=2)
            return False
        if np.any(solution[..., self.pindices] > 1):
            warnings.warn(f"Irrepeatable sites are repeated occupied!", stacklevel=2)
            return False
        if np.any(solution @ self.coef - np.array(self.u)) != 0:
            warnings.warn(f"Some are not a feasible solution!", stacklevel=2)
            return False
        return True


class DescWyckCombSolver:
    """Wyckoff Combination Solver by descending constant vector term

    A singleton class by parameters, and cache solutions.

    Parameters
    ----------
    group: int
        Space group number from [1,230].
    num_atoms : list[int] | npt.NDArray[np.int_]
        List of number of atoms of each elements.

    Notes
    -----
    Input `num_atoms` will be treated by them in descending order, and the solutions are
    also related to the descending input order.

    Warnings
    --------
    This parameterized-singleton class uses a WeakValueDictionary. Watchout it release all instances.

    Warnings
    --------
    Calling `solve()` will always solve the equation and won't cache the results.
    """

    _instance_queue: Queue = Queue(230 * 30)
    _instance: dict = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        sig = signature(cls.__init__)
        bound_args = sig.bind_partial(cls, *args, **kwargs)
        bound_args.apply_defaults()

        eqcode = cls.encode_equations(
            bound_args.arguments["group"],
            bound_args.arguments["num_atoms"],
        )

        key = (cls, eqcode, frozenset(bound_args.kwargs.items()))
        if key not in cls._instance:
            with cls._lock:
                if key not in cls._instance:
                    instance = super().__new__(cls)
                    instance.__eqcode = eqcode
                    if cls._instance_queue.full():
                        drop_key = cls._instance_queue.get_nowait()
                        cls._instance.pop(drop_key)
                    cls._instance_queue.put_nowait(key)
                    cls._instance[key] = instance
                    return instance
        return cls._instance[key]

    @staticmethod
    def encode_equations(group, num_atoms):
        group = Group(group)
        u = sorted(num_atoms, reverse=True)
        return hash_uints(chain((group.number, group.dim), u))

    @property
    def __eqcode(self) -> str:
        return self.___eqcode

    @__eqcode.setter  # hide it!
    def __eqcode(self, eqcode: str):
        self.___eqcode = eqcode

    def __init__(
        self,
        group: int,
        num_atoms: list[int] | tuple[list] | npt.NDArray[np.int_],
        dim: int = 3,
    ):
        if not hasattr(self, "_initialized"):
            self._initialized = True

            self.dim = dim
            self.prec = np.uint16
            self.group = Group(group, dim=dim)

            # Important! Sorted constants u!
            self.u = sorted(num_atoms, reverse=True)

            self.A = mul = self.group.multiplicity
            rep = self.group.repeatability
            self.p = list(map(int, compress(mul, map(lambda x: not x, rep))))
            self.q = list(map(int, compress(mul, rep)))
            if np.max(num_atoms) > (np.min(self.q) * np.iinfo(self.prec).max):
                raise ValueError("Number of atoms exceed precision of this solver.")
            self.q_compressed = sorted(set(self.q), reverse=True)
            self.gcd_q = int(np.gcd.reduce(self.q_compressed))
            self.q_star = list(map(lambda i: i // self.gcd_q, self.q_compressed))
            self.v_list = [len(list(vg)) for k, vg in groupby(self.q)]
            self.Ac = self.p + self.q_compressed  # A with compressed q

            self.nA = len(self.A)
            self.np = len(self.p)
            self.nq = len(self.q)
            self.nu = len(self.u)
            self.pindices = [i for i, r in enumerate(rep) if not r]  # irrepeatable
            self.qindices = [i for i, r in enumerate(rep) if r]  # repeatable
            self.n_qc = len(self.q_compressed)
            self.n_Ac = len(self.Ac)

            self.cachedir = PROJECT_CACHEDIR / f"{self.__class__.__name__}"
            try:
                self.cachedir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.warn(
                    f"Fail to create cache dir: {self.cachedir} . Message: {e}"
                )
                self.__can_cache = False
            else:
                self.__can_cache = True
            self.cachefile = self.cachedir / f"{self.__eqcode}.npz"

            if self.cachefile.exists():
                self.load_compressed_csr_solutions()
            else:
                self._compressed_csr_solutions: csr_array | None = None

            self._num_sol_each_p: list[int] | None = None
            self._num_solutions: int | None = None

            self.batchsize = 100000  # handle by batch to save memory

    def load_compressed_csr_solutions(self):
        self._compressed_csr_solutions = scipy.sparse.load_npz(self.cachefile)

    def cache_compressed_csr_solutions(self, sparsed_solutions: csr_array):
        try:
            scipy.sparse.save_npz(self.cachefile, sparsed_solutions)
        except Exception as e:
            warnings.warn(f"Fail to cache solutions to {self.cachefile} . Message: {e}")

    @property
    def compressed_csr_solutions(self) -> csr_array:
        if self._compressed_csr_solutions is None:
            self.compressed_solve()
        if self.__can_cache and not self.cachefile.exists():
            self.cache_compressed_csr_solutions(self._compressed_csr_solutions)
        return self._compressed_csr_solutions

    @property
    def compressed_solutions(self) -> np.ndarray:
        return self.compressed_csr_solutions.toarray().reshape(-1, self.nu, self.n_Ac)

    @property
    def num_sol_each_p(self):
        if self._num_sol_each_p is None:
            num_p = self.compressed_csr_solutions.shape[0]
            num_batch, remainder = divmod(num_p, self.batchsize)
            num_batch += int(remainder > 0)
            nsol_each_p = []
            for ibatch in range(num_batch):
                sli = slice(ibatch * self.batchsize, (ibatch + 1) * self.batchsize)
                compressed_sols = self.compressed_csr_solutions[sli].toarray()
                compressed_sols = compressed_sols.reshape(-1, self.nu, self.n_Ac)
                bc_sols = compressed_sols[..., self.np :]  # b_comprssed part
                for bc_sol in bc_sols:
                    n = math.prod(
                        math.prod(
                            math.comb(ibc + iv - 1, ibc)
                            for ibc, iv in zip(bc_by_u, self.v_list)  # expand by v
                        )
                        for bc_by_u in bc_sol  # expand by u
                    )
                    nsol_each_p.append(n)
            self._num_sol_each_p = nsol_each_p
        return self._num_sol_each_p

    @property
    def num_solutions(self) -> int:
        if self._num_solutions is None:
            self._num_solutions = sum(self.num_sol_each_p)
        return self._num_solutions

    @property
    def has_solutions(self) -> bool:
        num_p = self.compressed_csr_solutions.shape[0]
        return num_p > 0

    @staticmethod
    def mixed_bases_convention(decimal: int, bases):
        result = [0] * len(bases)
        for i, b in enumerate(bases[::-1]):
            decimal, result[len(bases) - i - 1] = divmod(decimal, b)
        return result

    def get_solution(self, index: int):
        if (index < -self.num_solutions) or (index >= self.num_solutions):
            raise IndexError(f"index {index} out of range {self.num_solutions}.")
        elif index < 0:
            index += self.num_solutions
        passed_nsols = 0
        for pidx, accumulate_nsols in enumerate(accumulate(self.num_sol_each_p)):
            if index < accumulate_nsols:
                break
            passed_nsols += accumulate_nsols
        else:
            raise IndexError(f"index {index} out of range {self.num_solutions}.")
        compressed_sol = self.compressed_csr_solutions[[pidx], :].toarray()
        compressed_sol = compressed_sol.reshape(self.nu, self.n_Ac)
        p_sol = compressed_sol[:, : self.np]
        q_compressed_sol = compressed_sol[:, self.np :]

        q_flattened_solvers = [
            NonNegLinDiophSolver(a=[1] * v, b=q)
            for q, v in zip(
                q_compressed_sol.flatten(),
                chain(*repeat(self.v_list, self.nu)),
                strict=True,
            )
        ]
        q_flattened_num_sols = [solver.num_solutions for solver in q_flattened_solvers]
        required_q_indices = self.mixed_bases_convention(
            decimal=index - passed_nsols,
            bases=q_flattened_num_sols,
        )
        q_sol_stacked = [
            solver.solutions[idx]
            for solver, idx in zip(q_flattened_solvers, required_q_indices)
        ]
        q_sol = np.hstack(q_sol_stacked).reshape(self.nu, self.nq)
        sol = np.zeros((self.nu, len(self.A)), dtype=self.prec)
        sol[:, self.pindices] = p_sol
        sol[:, self.qindices] = q_sol
        return sol

    def compressed_solve(self):
        p_solver = ExBinDivLinSolver(a=self.p, b=self.u, beta=self.gcd_q)
        p_solutions = p_solver.solutions
        compressed_csr_sols = []
        for pidx, (p, u_q) in enumerate(p_solutions):
            u_star_q = np.floor_divide(u_q, self.gcd_q)
            q_compressed_sols_by_each_u_star_q = [
                NonNegLinDiophSolver(a=self.q_star, b=int(u_star_q_k)).solutions
                for u_star_q_k in u_star_q
            ]
            q_compressed_sols = np.array(
                list(product(*q_compressed_sols_by_each_u_star_q)), dtype=self.prec
            )
            for batched_q_sols in batched(q_compressed_sols, self.batchsize):
                sols = np.zeros(
                    (len(batched_q_sols), self.nu, self.n_Ac),
                    dtype=self.prec,
                )
                sols[..., : self.np] = p
                sols[..., self.np :] = batched_q_sols
                compressed_csr_sols.append(
                    csr_array(sols.reshape(sols.shape[0], -1), dtype=self.prec)
                )
        if len(compressed_csr_sols) == 0:
            self._compressed_csr_solutions = csr_array(
                (0, self.n_Ac * self.nu),
                dtype=self.prec,
            )
        else:
            self._compressed_csr_solutions = scipy.sparse.vstack(compressed_csr_sols)

    def self_verify(self, strict=False) -> bool:
        if np.any(self.compressed_solutions[..., : self.np] > 1):
            warnings.warn(f"Irrepeatable sites are repeated occupied!", stacklevel=2)
            return False
        if np.any(self.compressed_solutions @ self.Ac - self.u) != 0:
            warnings.warn(f"Some are not a feasible solution!", stacklevel=2)
            return False
        if strict:
            warnings.warn(f"Checking all {self.num_solutions} solutions...")
            for idx in range(self.num_solutions):
                sol = self.get_solution(idx)
                self.verify(sol)
                if (idx > 0) and (idx % self.batchsize == 0):
                    warnings.warn(f"Checked {idx} solutions...")
        return True

    def verify(self, solution):
        if np.any(solution < 0):
            warnings.warn(f"Some solution is negative!", stacklevel=2)
            return False
        if np.any(solution[..., self.pindices] > 1):
            warnings.warn(f"Irrepeatable sites are repeated occupied!", stacklevel=2)
            return False
        if np.any(solution @ self.A - self.u) != 0:
            warnings.warn(f"Some are not a feasible solution!", stacklevel=2)
            return False
        return True


def verify_spg(spg_num_list, numb_atoms_list, dim=3) -> list:
    verified_spg = []
    if dim == 4:
        for spg_num in spg_num_list:
            solver = WyckCombSolver(group=spg_num, num_atoms=numb_atoms_list, dim=dim)
            if solver.has_solutions:
                verified_spg.append(spg_num)
        print(
            f"These are valid space group when atom list is {numb_atoms_list}: {verified_spg} "
        )
        return verified_spg
    else:
        for spg_num in spg_num_list:
            group = Group(spg_num, dim=dim)
            supercell = np.array(numb_atoms_list) * group.prim2unit_factor
            min_numb_atom = min(supercell)
            min_multiplicity = min(group.multiplicity)
            if min_numb_atom >= min_multiplicity and np.all(
                [(numb_atom % min_multiplicity) == 0 for numb_atom in supercell]
            ):
                verified_spg.append(spg_num)
        return verified_spg
