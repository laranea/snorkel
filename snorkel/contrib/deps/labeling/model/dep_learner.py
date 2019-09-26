import warnings
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
import scipy as sp


class DependencyLearner(object):
    """A model for learning LF dependencies.

    This class learns the dependencies among labeling functions.
    It is based on the approach in
    [Learning Dependency Structures for Weak Supervision Models]
    (https://arxiv.org/pdf/1903.05844.pdf), published in ICML'19.
    In this approach, we use robust principal component analysis (rPCA)
    based algorithm to decompose the inverse covariance matrix into a sparse
    component that encodes the dependency structure and a low rank component
    due to marginalizing over the latent true label variable.

    Examples
    --------
    >>> dep_learner = DependencyLearner()
    >>> dep_learner = DependencyLearner(cardinality=3)

    Parameters
    ----------
    cardinality
        Number of classes, by default 2

    Attributes
    ----------
    cardinality
        Number of classes, by default 2
    """

    def __init__(self, cardinality: int = 2) -> None:
        self.cardinality = cardinality

    def _force_singleton(
        self, deps: List[Tuple[int, int]], M: int
    ) -> List[Tuple[int, int]]:
        """Force singleton separator assumption given list of dependencies.

        This translates to converting chain dependencies to fully connected clusters of dependencies.

        More information on singleton separator assumption in [AAAI'19](https://arxiv.org/pdf/1810.02840.pdf) and [ICML'19](https://arxiv.org/pdf/1903.05844.pdf) papers.
        """
        # remove duplicated pairs
        deps_singleton = []
        for i, j in deps:
            if i < j:
                deps_singleton.append((i, j))

        # add edges to convert chain deps to cluster deps
        for i, j in deps:
            for k, l in deps:
                if (i == k) and (j < l):
                    deps_singleton.append((j, l))
                if (j == l) and (i < k):
                    deps_singleton.append((i, k))
                if (j == k) and (i < l):
                    deps_singleton.append((i, l))
                if (i == l) and (j < k):
                    deps_singleton.append((j, k))
        all_deps = list(set(deps_singleton))

        if len(all_deps) == sp.special.comb(M, 2):
            raise ValueError(
                "Dependency structure is fully connected. Rerun with higher thresh_mult."
            )
        return all_deps

    def _get_deps_from_inverse_sig(
        self, J: np.ndarray, thresh: float
    ) -> List[Tuple[int, int]]:
        """Select values larger than thresh in J as dependent LF indices."""
        deps = []
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                if abs(J[i, j]) > thresh:
                    deps.append((i, j))
        return deps

    def fit(
        self,
        L: np.ndarray,
        thresh_mult: Optional[float] = 0.5,
        gamma: Optional[float] = 1e-8,
        lam: Optional[float] = 0.1,
        verbose: Optional[bool] = False,
    ) -> List[Tuple[int, int]]:
        r"""Learn dependencies among LFs.

        Learn dependencies using robust PCA based method. In case of multi-class, the method considers log2(cardinality) splits to calculate agreements and disagreements. Solves the following problem

        (\hat{S}, \hat{L}) = argmin \mathcal{L}(S - L, \Sigma_O) + lam*(gamma*||S||_1 + ||L||_*)

        s.t. S - L \succ 0, L \succeq 0

        Note: Best performance requires hyperparameter search over the thresh parameter.

        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        thresh_mult
            Threshold multiplier for selecting thresh_mult * max off diagonal entry from sparse matrix
        gamma
            Parameter in objective function related to sparsity
        lam
            Parameter in objective function related to sparsity and low rank
        verbose
            Whether solver is verbose

        Raises
        ------
        ValueError
            If L has higher cardinality than passed in or solver fails

        Returns
        -------
        List[Tuple[int, int]]
            List of tuples denoting dependencies among LFs
        """

        N = float(np.shape(L)[0])
        M = np.shape(L)[1]
        O_all = np.zeros((M, M))

        if L.max() + 1 > self.cardinality:
            raise ValueError(
                f"Does not match DependencyLearner cardinality={self.cardinality}, L has cardinality {L.max()+1}"
            )

        # calculate agreement-disagreement rates based on random class splits
        L_shift = np.copy(L)
        split_list = np.random.choice(
            range(1, self.cardinality),
            size=np.int(np.log2(self.cardinality)),
            replace=False,
        )
        for class_thresh in split_list:
            L_shift[L_shift == -1] = self.cardinality + 1
            L_shift[L_shift < class_thresh] = -1
            L_shift[L_shift == self.cardinality + 1] = 0
            L_shift[L_shift >= class_thresh] = 1
            O_all += (np.dot(L_shift.T, L_shift)) / (N - 1) - np.outer(
                np.mean(L_shift, axis=0), np.mean(L_shift, axis=0)
            )
        sigma_O = O_all / float(len(split_list))

        # set variables for cvxpy
        O = 1 / 2 * (sigma_O + sigma_O.T)
        O_root = np.real(sp.linalg.sqrtm(O))
        L_cvx = cp.Variable([M, M], PSD=True)  # low-rank matrix
        S = cp.Variable([M, M], PSD=True)  # sparse matrix
        R = cp.Variable([M, M], PSD=True)  # S-L matrix

        # define objective and constraints
        objective = cp.Minimize(
            0.5 * (cp.norm(R * O_root, "fro") ** 2)
            - cp.trace(R)
            + lam * (gamma * cp.pnorm(S, 1) + cp.norm(L_cvx, "nuc"))
        )
        constraints = [R == S - L_cvx, L_cvx >> 0]

        # solve problem with cvxpy
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=verbose)

        # set threshold based on max. off diagonal entries
        J_hat = S.value
        if J_hat is None:
            raise ValueError(
                "There may be no dependencies among LFs. Otherwise, try different gamma and lambda values."
            )
        off_diag = (np.abs(J_hat) - np.diag(np.diag(np.abs(J_hat)))).ravel()
        q1, q3 = np.percentile(np.sort(off_diag), [25, 75])
        outlier_bound = q3 + 2.0 * (q3 - q1)
        thresh = thresh_mult * np.max(off_diag)

        # add warning if no outliers present in off diagonal entries
        if np.max(off_diag) < outlier_bound:
            warnings.warn(
                "There may be no real dependencies among LFs, returned list contains extraneous dependencies. Include thresh_mult = 1.0 in parameter search."
            )

        # find dependencies
        deps_all = self._get_deps_from_inverse_sig(J_hat, thresh)
        deps = self._force_singleton(deps_all, M)
        return deps