from typing import Any, List, Optional, Tuple

import numpy as np

from snorkel.contrib.deps.labeling.model.label_model import DependencyAwareLabelModel
from snorkel.labeling import SklearnLabelModel
from snorkel.utils.lr_schedulers import LRSchedulerConfig
from snorkel.utils.optimizers import OptimizerConfig


class SklearnDependencyAwareLabelModel(SklearnLabelModel):
    """A sklearn wrapper for DependencyAwareLabelModel and DependencyLearner.

    Uses output of create_param_search_data.

    Note that all hyperparameters for the fit and score functions are accepted at the time the class is defined.

    Parameters
    ----------
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
    cardinality
         Number of classes, by default 2
    verbose
         Whether to include print statements
    device
        What device to place the model on ('cpu' or 'cuda:0', for example)
    metric
        The metric to report with score()
    tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions
    n_epochs
        The number of epochs to train (where each epoch is a single optimization step)
    lr
        Base learning rate (will also be affected by lr_scheduler choice and settings)
    l2
        Centered L2 regularization strength
    optimizer
        Which optimizer to use (one of ["sgd", "adam", "adamax"])
    optimizer_config
        Settings for the optimizer
    lr_scheduler
        Which lr_scheduler to use (one of ["constant", "linear", "exponential", "step"])
    lr_scheduler_config
        Settings for the LRScheduler
    prec_init
        LF precision initializations / priors
    seed
        A random seed to initialize the random number generator with
    log_freq
        Report loss every this many epochs (steps)
    mu_eps
        Restrict the learned conditional probabilities to [mu_eps, 1-mu_eps]
    """

    def __init__(
        self,
        learn_deps: Optional[bool] = True,
        thresh_mult: Optional[float] = 0.5,
        gamma: Optional[float] = 1e-8,
        lam: Optional[float] = 0.1,
        deps: Optional[List[Tuple[int, int]]] = None,
        cardinality: int = 2,
        verbose: bool = False,
        device: str = "cpu",
        metric: str = "accuracy",
        tie_break_policy: str = "abstain",
        n_epochs: int = 100,
        lr: float = 0.01,
        l2: float = 0.0,
        optimizer: str = "sgd",
        optimizer_config: Optional[OptimizerConfig] = None,
        lr_scheduler: str = "constant",
        lr_scheduler_config: Optional[LRSchedulerConfig] = None,
        prec_init: float = 0.7,
        seed: int = np.random.randint(1e6),
        log_freq: int = 10,
        mu_eps: Optional[float] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:

        self.learn_deps = learn_deps
        self.thresh_mult = thresh_mult
        self.gamma = gamma
        self.lam = lam
        self.deps = deps
        super().__init__(
            cardinality,
            verbose,
            device,
            metric,
            tie_break_policy,
            n_epochs,
            lr,
            l2,
            optimizer,
            optimizer_config,
            lr_scheduler,
            lr_scheduler_config,
            prec_init,
            seed,
            log_freq,
            mu_eps,
            class_balance,
            **kwargs,
        )
        self.label_model = DependencyAwareLabelModel(
            cardinality=self.cardinality, verbose=self.verbose, device=self.device
        )

    def fit(
        self, L: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> "SklearnDependencyAwareLabelModel":
        """
        Train label model.

        Parameters
        ----------
        L
             An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y
             Placeholder, not used for training model.

        Returns
        -------
        SklearnDependencyAwareLabelModel
        """
        self.label_model.fit_with_deps(
            L_train=L,
            learn_deps=self.learn_deps,
            thresh_mult=self.thresh_mult,
            gamma=self.gamma,
            lam=self.lam,
            deps=self.deps,
            class_balance=self.class_balance,
            n_epochs=self.n_epochs,
            lr=self.lr,
            l2=self.l2,
            optimizer=self.optimizer,
            optimizer_config=self.optimizer_config,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_config=self.lr_scheduler_config,
            prec_init=self.prec_init,
            seed=self.seed,
            log_freq=self.log_freq,
            mu_eps=self.mu_eps,
        )

        return self
