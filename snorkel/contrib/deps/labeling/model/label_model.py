import logging
import random
from functools import partial
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel, TrainConfig
from snorkel.utils.config_utils import merge_config


class DependencyAwareLabelModel(LabelModel):
    def _generate_O_inv(self, L: np.ndarray):
        """Form the *inverse* overlaps matrix"""
        self._generate_O(L)
        self.O_inv = torch.from_numpy(np.linalg.inv(self.O.numpy())).float()

    def _init_params(self) -> None:
        super()._init_params()
        if self.inv_form:
            self.Z = nn.Parameter(torch.randn(self.d, self.cardinality)).float()

    def _get_Q(self):
        """Get the model's estimate of Q = \mu P \mu^T
        We can then separately extract \mu subject to additional constraints,
        e.g. \mu P 1 = diag(O).
        """
        Z = self.Z.detach().clone().numpy()
        O = self.O.numpy()
        I_k = np.eye(self.cardinality)
        return O @ Z @ np.linalg.inv(I_k + Z.T @ O @ Z) @ Z.T @ O

    def _loss_inv_Z(self):
        return torch.norm((self.O_inv + self.Z @ self.Z.t())[self.mask]) ** 2

    def _loss_inv_mu(self, l2: float = 0) -> torch.Tensor:
        loss_1 = torch.norm(self.Q - self.mu @ self.P @ self.mu.t()) ** 2
        loss_2 = torch.norm(torch.sum(self.mu @ self.P, 1) - torch.diag(self.O)) ** 2
        return loss_1 + loss_2 + self.loss_l2(l2=l2)

    def fit_loss(self, loss_fn):
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
        deps: Optional[List[Tuple[int, int]]] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Train label model.

        Train label model to estimate mu, the parameters used to combine LFs.

        Parameters
        ----------
        L_train
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y_dev
            Gold labels for dev set for estimating class_balance, by default None
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
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> label_model.fit(L, Y_dev=Y_dev)
        >>> label_model.fit(L, class_balance=[0.7, 0.3])
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
        self._set_dependencies(deps or [])
        # Whether to compute O_inv for handling dependencies.
        self.inv_form = len(self.deps) > 0
        lf_analysis = LFAnalysis(L_train)
        self.coverage = lf_analysis.lf_coverages()

        # Compute O and initialize params
        if self.config.verbose:  # pragma: no cover
            logging.info("Computing O...")
        if self.inv_form:
            self._generate_O_inv(L_shift)
        else:
            self._generate_O(L_shift)
        self._init_params()

        # Estimate \mu
        if self.config.verbose:  # pragma: no cover
            logging.info("Estimating \mu...")

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

        if self.inv_form:
            # Estimate Z.
            self.fit_loss(self._loss_inv_Z)
            self.Q = torch.from_numpy(self._get_Q()).float()
            # Estimate mu.
            self._set_optimizer()
            self._set_lr_scheduler()
            self.fit_loss(partial(self._loss_inv_mu, l2=self.train_config.l2))
        else:
            self.fit_loss(partial(self._loss_mu, l2=self.train_config.l2))

        # Post-processing operations on mu
        self._clamp_params()
        self._break_col_permutation_symmetry()

        # Return model to eval mode
        self.eval()

        # Print confusion matrix if applicable
        if self.config.verbose:  # pragma: no cover
            logging.info("Finished Training")
