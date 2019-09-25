import cvxpy as cp
import numpy as np
import scipy as sp

from itertools import combinations


class DependencyLearner(object):
    #TODO: remove k once multi-class is implemented
    def __init__(self, k=2):
        self.k = k
        if self.k != 2:
            raise NotImplementedError("Dependency learning only works for k=2")

    def _force_singleton(self, deps):
        deps_singleton = []
        for i,j in deps:
            if i < j:
                deps_singleton.append((i,j))

        for i,j in deps:
            for k,l in deps:
                if (i == k) and (j < l):
                    deps_singleton.append((j,l)) 
                if (j == l) and (i < k):
                    deps_singleton.append((i,k))
                if (j == k) and (i < l):
                    deps_singleton.append((i,l))
                if (i == l) and (j < k):
                    deps_singleton.append((j,k))
        return list(set(deps_singleton))

    def _get_deps_from_inverse_sig(self, J, thresh):
        deps = []
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                if abs(J[i,j]) > thresh:
                    deps.append((i,j))
        return deps

    def fit(self, L, thresh=0.5, obj_option='default', const_option='default'):
        # sum selected disagreements
        N = float(np.shape(L)[0])
        M = np.shape(L)[1]
        cardinality = np.max(L) + 1
        L_shift = np.copy(L)
        O_all = np.zeros((M, M))

        count_combos = 0
        split_list = np.random.choice(range(1, cardinality), size=np.int(np.log2(cardinality)), replace=False)
        for class_thresh in split_list:
            count_combos +=1
            L_shift = np.copy(L)
            L_shift[L_shift == -1] = cardinality + 1
            L_shift[L_shift <= class_thresh] = -1
            L_shift[L_shift == cardinality + 1] = 0
            L_shift[L_shift > class_thresh] = 1
            O_all += (np.dot(L_shift.T,L_shift))/(N-1) -  np.outer(np.mean(L_shift,axis=0), np.mean(L_shift,axis=0))

        sigma_O = O_all/float(count_combos)

        #bad code
        O = 1/2*(sigma_O+sigma_O.T)
        O_root = np.real(sp.linalg.sqrtm(O))

        # low-rank matrix
        L_cvx = cp.Variable([M,M], PSD=True)

        # sparse matrix
        S = cp.Variable([M,M], PSD=True)

        # S-L matrix
        R = cp.Variable([M,M], PSD=True)

        #reg params
        lam = 1/np.sqrt(M)
        gamma = 1e-8

        objective = cp.Minimize(0.5*(cp.norm(R*O_root, 'fro')**2) - cp.trace(R) + lam*(gamma*cp.pnorm(S,1) + cp.norm(L_cvx, "nuc")))
        constraints = [R == S - L_cvx, L_cvx>>0]

        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=False, solver=cp.CVXOPT)
        opt_error = prob.value

        #extract dependencies
        J_hat = S.value

        #AUTO THRESH SETTING BASED ON OFF_DIAG MAXIMUM
        thresh = 0.5*np.max(np.abs(J_hat) - np.diag(np.diag(J_hat)))
        deps_all = self._get_deps_from_inverse_sig(J_hat, thresh)
        deps = self._force_singleton(deps_all)
        return deps, J_hat