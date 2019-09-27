import logging
import random
from functools import partial
from typing import Any, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import scipy as sp
import torch

from snorkel.contrib.deps.labeling.model.dep_learner import DependencyLearner
from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel, TrainConfig
from snorkel.utils.config_utils import merge_config


class DependencyAwareLabelModel(LabelModel):
    """A LabelModel that handles dependencies and learn associated weights to assign training labels."""

    def _loss_inv_mu(self, l2: float = 0) -> torch.Tensor:
        loss_1 = torch.norm(self.Q - self.mu @ self.P @ self.mu.t()) ** 2
        loss_2 = torch.norm(torch.sum(self.mu @ self.P, 1) - torch.diag(self.O)) ** 2
        return loss_1 + loss_2 + self.loss_l2(l2=l2)

    def _robust_pca_Q(
        self, L: np.ndarray, gamma: Optional[float] = 1e-8, lam: Optional[float] = 0.1
    ) -> np.ndarray:
        N = float(np.shape(L)[0])
        M = np.shape(L)[1]
        sigma_O = (np.dot(L.T, L)) / (N - 1) - np.outer(
            np.mean(L, axis=0), np.mean(L, axis=0)
        )
        O_root = np.real(sp.linalg.sqrtm(sigma_O))

        L_cvx = cp.Variable([M, M], PSD=True)
        S = cp.Variable([M, M], PSD=True)
        R = cp.Variable([M, M], PSD=True)

        objective = cp.Minimize(
            0.5 * (cp.norm(R * O_root, "fro") ** 2)
            - cp.trace(R)
            + lam * (gamma * cp.pnorm(S, 1) + cp.norm(L_cvx, "nuc"))
        )
        constraints = [R == S - L_cvx, L_cvx >> 0]

        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)
        U, s, V = np.linalg.svd(L_cvx.value)
        Z = np.sqrt(s[: self.cardinality]) * U[:, : self.cardinality]
        O = self.O.numpy()
        I_k = np.eye(self.cardinality)
        return O @ Z @ np.linalg.inv(I_k + Z.T @ O @ Z) @ Z.T @ O

    def _fit_loss(self, loss_fn):
        # Restore model if necessary
        start_iteration = 0

        # Train the model
        metrics_hist = {}  # The most recently seen value for all metrics
        for epoch in range(start_iteration, self.train_config.n_epochs):
            self.running_loss = 0.0
            self.running_examples = 0

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass to calculate the average loss per example
            loss = self._loss_mu(l2=self.train_config.l2)
            if torch.isnan(loss):
                msg = "Loss is NaN. Consider reducing learning rate."
                raise Exception(msg)

            # Backward pass to calculate gradients
            # Loss is an average loss per example
            loss.backward()

            # Perform optimizer step
            self.optimizer.step()

            # Calculate metrics, log, and checkpoint as necessary
            metrics_dict = self._execute_logging(loss)
            metrics_hist.update(metrics_dict)

            # Update learning rate
            self._update_lr_scheduler(epoch)

    def fit_with_deps(
        self,
        L_train: np.ndarray,
        Y_dev: Optional[np.ndarray] = None,
        learn_deps: Optional[bool] = True,
        thresh_mult: Optional[float] = 0.5,
        gamma: Optional[float] = 1e-8,
        lam: Optional[float] = 0.1,
        deps: Optional[List[Tuple[int, int]]] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Train label model using dependencies (if given).

        Train label model to estimate mu, the parameters used to combine LFs.

        Parameters
        ----------
        L_train
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y_dev
            Gold labels for dev set for estimating class_balance, by default None
        learn_deps
            Whether to learn dependencies, by default True
        thresh_mult
            Threshold multiplier for selecting thresh_mult * max off diagonal entry from sparse matrix
        gamma
            Parameter in objective function related to sparsity
        lam
            Parameter in objective function related to sparsity and low rank
        deps
            Optional list of pairs of correlated LF indices.
        class_balance
            Each class's percentage of the population, by default None
        **kwargs
            Arguments for changing train config defaults

        Raises
        ------
        Exception
            If loss in NaN

        Examples
        --------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> Y_dev = [0, 1, 0]
        >>> label_model = DependencyAwareLabelModel(verbose=False)
        >>> label_model.fit_with_deps(L, deps=[(0, 2)])  # doctest: +SKIP
        >>> label_model.fit_with_deps(L, deps=[(0, 2)], Y_dev=Y_dev)  # doctest: +SKIP
        >>> label_model.fit_with_deps(L, deps=[(0, 2)], class_balance=[0.7, 0.3])  # doctest: +SKIP
        >>> label_model.fit_with_deps(L, learn_deps=True)  # doctest: +SKIP
        """
        # Set random seed
        self.train_config: TrainConfig = merge_config(  # type:ignore
            TrainConfig(), kwargs  # type:ignore
        )
        # Update base config so that it includes all parameters
        random.seed(self.train_config.seed)
        np.random.seed(self.train_config.seed)
        torch.manual_seed(self.train_config.seed)

        L_shift = L_train + 1  # convert to {0, 1, ..., k}
        if L_shift.max() > self.cardinality:
            raise ValueError(
                f"L_train has cardinality {L_shift.max()}, cardinality={self.cardinality} passed in."
            )

        self._set_constants(L_shift)
        self._set_class_balance(class_balance, Y_dev)

        if learn_deps:
            self.dep_learner = DependencyLearner(cardinality=self.cardinality)
            learned_deps = self.dep_learner.fit(
                L_train, thresh_mult, gamma, lam, verbose=self.config.verbose
            )
        else:
            learned_deps = []

        if deps is None:
            deps = learned_deps
        elif learned_deps != []:
            deps += learned_deps
        self.deps = deps
        self._set_dependencies(self.deps or [])

        lf_analysis = LFAnalysis(L_train)
        self.coverage = lf_analysis.lf_coverages()

        # Compute O and initialize params
        if self.config.verbose:  # pragma: no cover
            logging.info("Computing O...")
        self._generate_O(L_shift)
        self._init_params()

        # Set model to train mode
        self.train()

        # Move model to GPU
        if self.config.verbose and self.config.device != "cpu":  # pragma: no cover
            logging.info("Using GPU...")
        self.to(self.config.device)

        # Set training components
        self._set_logger()
        self._set_optimizer()
        self._set_lr_scheduler()

        if self.higher_order:
            self.Q = self._robust_pca_Q(
                self._get_augmented_label_matrix(L_shift), lam, gamma
            )
            self._fit_loss(partial(self._loss_inv_mu, l2=self.train_config.l2))
        else:
            self._fit_loss(partial(self._loss_mu, l2=self.train_config.l2))

        # Post-processing operations on mu
        self._clamp_params()
        self._break_col_permutation_symmetry()

        # Return model to eval mode
        self.eval()

        # Print confusion matrix if applicable
        if self.config.verbose:  # pragma: no cover
            logging.info("Finished Training")
