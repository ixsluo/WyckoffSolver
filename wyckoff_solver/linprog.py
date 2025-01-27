# Linear Programming Problems

import math
import pickle
import warnings
from collections.abc import Iterator
from itertools import chain, product
from pathlib import Path
from typing import Any, cast, Literal

import numpy as np
import numpy.typing as npt
import scipy
from scipy.sparse import csr_array

from .cache import PROJECT_CACHEDIR
from .dioph import nonneglindiophsolver_module
from .utils.hash import hash_uints
from .utils.singleton import ParamEncodeBase, ParamSingletonFactory


class ExBinDivLinSolver(ParamEncodeBase, metaclass=ParamSingletonFactory.create_metaclass("ExBinDivLinSolver")):
    r"""Exclusive Binary Divisible Linear Porgramming Problem

    See constrains in notes.

    Let :math:`\boldsymbol{b}` and :math:`\boldsymbol{a}` shape as (k,) and (i,),
    then the solution :math:`\boldsymbol{x}` shapes as (k,i).

    :math:`\boldsymbol{x}` as well as :math:`\boldsymbol{b}^{\prime} := \boldsymbol{b} - \boldsymbol{a}\boldsymbol{x}` are returned.

    Parameters
    ----------
    a : array_like
        Coefficients vector :math:`\boldsymbol{a}`. Must be integers.
    b : array_like
        Constants vector :math:`\boldsymbol{b}`. Must be integers.
    beta : int
        Divisor :math:`beta`.
    cache : bool
        If True, try to write cache when calling :py:meth:`solve`.

    Warnings
    --------
    Calling :py:meth:`solve` will always solve the equation and won't cache the results.
    It is suggested to call `solutions` to using cache and return all solutions.

    Notes
    -----
    .. math::
        :nowrap:

        \begin{gather*}
            \beta | \boldsymbol{b}_k - \boldsymbol{a}_i\boldsymbol{x}_{ki}, \quad k=0,1,...  \\
            \boldsymbol{b}_k - \boldsymbol{a}_i\boldsymbol{x}_{ki} \ge 0, \quad k=0,1,...    \\
            0 \le \boldsymbol{x}_{ki} \le 1, \quad k,i=0,1,...  \\
            \sum_{k}\boldsymbol{x}_{ki} \le 1, \quad j=0,1,...  \\
        \end{gather*}

    """

    @classmethod
    def encode_arguments(cls, a, b, beta, **kwargs):
        return hash_uints(chain((len(a), len(b)), a, b, (beta,)))

    def __init__(
        self,
        a: npt.NDArray[np.int_] | list[int],
        b: npt.NDArray[np.int_] | list[int],
        beta: int,
    ):
        self.a = np.array(a)
        self.b = np.array(b)
        self.beta = int(beta)

        self.cachedir = PROJECT_CACHEDIR / f"{self.__class__.__name__}"
        try:
            self.cachedir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warnings.warn(f"Fail to create cache dir: {self.cachedir}. Message: {e}")
            self.cachefile = None
        else:
            self.cachefile = self.cachedir / f"{self._cls_code}.pkl"

        self._solutions: list[tuple[list[list[int]], list[int]]] | None = None
        self._num_solutions: int | None = None

    def _load_solutions(self):
        with self.cachefile.open("rb") as f:
            return pickle.load(f)

    def _cache_solutions(self):
        if self._solutions is None:
            warnings.warn("Solutions not yet been solved. Refuse to cache.")
            return
        try:
            with self.cachefile.open("wb") as f:
                pickle.dump(self._solutions, f)
        except Exception as e:
            warnings.warn(f"Fail to cache solutions to {self.cachefile} . Message: {e}")

    @property
    def solutions(self):
        if self._solutions is None:
            if self.cachefile is not None and self.cachefile.exists():
                self._solutions = self._load_solutions()
            else:
                self._solutions = self.solve()
        if self.cachefile is not None and not self.cachefile.exists():
            self._cache_solutions()
        return self._solutions

    def count_solutions(self):
        return len(self.solutions)

    @property
    def num_solutions(self) -> int:
        if self._num_solutions is None:
            self._num_solutions = self.count_solutions()
        return self._num_solutions

    def solve(self):
        return list(self.i_solve())

    def i_solve(self) -> Iterator[tuple[list[list[int]], list[int]]]:
        r"""solve the solutions

        Yields
        ------
            x : list[list[int]]
                Solution :math:`\boldsymbol{x}` matrix.
            b_prime : list[int]
                Resulting part of :math:`\boldsymbol{b}^{\prime} := \boldsymbol{b} - \boldsymbol{a}\boldsymbol{x}`.
        """
        # add a first row as placeholder to fill ones
        for rowindices in product(range(len(self.b) + 1), repeat=len(self.a)):
            if len(self.a) == 0:
                x = np.zeros((len(self.b), 0), dtype=int)  # handle empty self.a
                b_prime = self.b
            else:
                x = np.zeros((1 + len(self.b), len(self.a)), dtype=int)
                ones_mask = self._get_ones_mask(rowindices)
                # cython compatibility, same as *ones_mask
                x[ones_mask[0], ones_mask[1]] = 1
                x = x[1:, :]  # remove first row
                b_prime = self.b - np.einsum('i,ki->k', self.a, x)
            if np.all(b_prime >= 0) and np.all(b_prime % self.beta == 0):
                yield x.tolist(), b_prime.tolist()

    def _get_ones_mask(self, rowindices):
        # ones_mask = np.roll(np.array(list(zip(*enumerate(rowindices)))), 1, axis=0)
        # the following is faster
        ones_mask = zip(*((rowidx, colidx) for colidx, rowidx in enumerate(rowindices)))
        return list(ones_mask)

    def verify_solutions(self, solutions: list[tuple[list[list[int]], list[int]]]):
        num_solutions = len(solutions)
        x_cat, b_prime_cat = list(map(lambda x: list(chain(*x)), zip(*solutions)))
        # beta | b'
        if np.any(np.remainder(b_prime_cat, self.beta) != 0):
            return False
        # b' = b - a * x
        res = np.subtract(
            b_prime_cat,
            np.tile(self.b, num_solutions) - np.einsum("i,ki", self.a, x_cat),
        )
        if np.any(res != 0):
            return False
        if np.any(np.array(x_cat) > 1):
            return False
        return True


class NonNegLinDiophSolver(ParamEncodeBase, metaclass=ParamSingletonFactory.create_metaclass("NonNetLinDiophSolver")):
    r"""Non-negative Linear Diophantine Problem

    See constrains in notes.

    Let :math:`\boldsymbol{a}` shape as (i,), then the solution :math:`\boldsymbol{x}` shapes as (i,).

    The generation function:

    .. math::

       G(x) = \prod_{i} (1 + x^{\boldsymbol{a}_i} + x^{2\boldsymbol{a}_i} + ...)

    the number of solutions equals to the coefficient of term :math:`\boldsymbol{x}^b`

    Count the number of solutions by dynamic programming method, and find all solutions
    by backtracing method.

    Given :math:`\boldsymbol{a}` in reversed order can improve the performance.

    Parameters
    ----------
    a : array_like
        Coefficients vector :math:`\boldsymbol{a}`. Must be integers.
    b : int
        Constants integer :math:`b`. Must be integer.

    Warnings
    --------
    Calling :py:meth:`solve` will always solve the equation and won't cache the results.
    It is suggested to call `solutions` to using cache and return all solutions.

    Notes
    -----
    .. math::
        :nowrap:

        \begin{gather*}
            \boldsymbol{a}_i\boldsymbol{x}_{i} = b    \\
            \boldsymbol{x}_{i} \ge 0, \quad j=0,1,... \\
        \end{gather*}
    """

    @classmethod
    def encode_arguments(cls, a, b, **kwargs) -> str:
        return hash_uints(chain(a, (b,)))

    def __init__(
        self,
        a: npt.NDArray[np.int_] | list[int],
        b: int,
    ):
        if not hasattr(self, "_initialized"):
            self._initialized = True

            self.a = np.array(a)
            self.b = int(b)

            self.cachedir = PROJECT_CACHEDIR / f"{self.__class__.__name__}"
            try:
                self.cachedir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.warn(f"Fail to create cache dir: {self.cachedir}. Message: {e}")
                self.cachefile = None
            else:
                self.cachefile = self.cachedir / f"{self._cls_code}.npz"

            self._solutions: np.ndarray | None = None
            self._num_solutions: int | None = None

    def _load_solutions(self) -> np.ndarray:
        return scipy.sparse.load_npz(self.cachefile).toarray()

    def _cache_solutions(self):
        if self._solutions is None:
            warnings.warn("Solutions not yet been solved. Refuse to cache.")
            return
        try:
            solutions = csr_array(self._solutions)
            scipy.sparse.save_npz(self.cachefile, solutions)
        except Exception as e:
            warnings.warn(f"Fail to cache solutions to {self.cachefile}. Message: {e}")

    def count_solutions(self, backend: Literal["fortran", "python"] = "fortran"):
        if backend == "fortran":
            return self.count_solutions_f90()
        elif backend == "python":
            return self.count_solutions_py()
        else:
            raise ValueError(f"Unknown backend: {backend}.")

    def count_solutions_f90(self):
        return nonneglindiophsolver_module.count_solutions(self.a, self.b)

    def count_solutions_py(self):
        if all(map(lambda x: x == 1, self.a)):
            return math.comb(len(self.a) - 1 + self.b, len(self.a) - 1)
        dp = [1] + [0] * self.b
        for a in self.a:
            for j in range(a, self.b + 1):
                dp[j] += dp[j - a]
        return dp[self.b]

    @property
    def solutions(self) -> np.ndarray:
        if self._solutions is None:
            if self.cachefile is not None and self.cachefile.exists():
                self._solutions = self._load_solutions()
            else:
                self._solutions = self.solve()
        if self.cachefile is not None and not self.cachefile.exists():
            self._cache_solutions()
        return self._solutions

    @property
    def num_solutions(self) -> int:
        if self._num_solutions is None:
            self._num_solutions = self.count_solutions()
        return self._num_solutions

    def solve(self, backend: Literal["fortran", "python"] = "fortran"):
        if backend == "fortran":
            return self.solve_f90()
        elif backend == "python":
            return self.solve_py()
        else:
            raise ValueError(f"Unknown backend: {backend}.")

    def solve_f90(self) -> np.ndarray:
        self._solutions = nonneglindiophsolver_module.generate_solutions(
            self.a, self.b, self.num_solutions
        )
        self._solutions = cast(np.ndarray, self._solutions)
        return self._solutions

    def solve_py(self):
        solutions = self._generate_solutions()
        if len(solutions) == 0:
            self._solutions = np.zeros((0, len(self.a)), dtype=int)
        else:
            self._solutions = np.array(solutions)
        return self._solutions

    def _generate_solutions(self) -> list[list[int]]:
        def backtrack(target, index, current):
            if target == 0:
                yield current.copy()
                return
            if target < 0 or index >= len(self.a):
                return
            for i in range(target // self.a[index] + 1):
                current[index] = i
                yield from backtrack(target - i * self.a[index], index + 1, current)
                current[index] = 0

        return list(backtrack(self.b, 0, [0] * len(self.a)))

    def verify_solutions(self, solutions) -> bool:
        if np.all(np.einsum("i,ki->k", self.a, solutions) == self.b):
            return True
        else:
            return False
