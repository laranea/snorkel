import unittest
from typing import List, Set, Tuple

import numpy as np
import pytest

from snorkel.contrib.deps.labeling.model.label_model import DependencyAwareLabelModel
from snorkel.labeling import LabelModel
from snorkel.labeling.model.label_model import TrainConfig


def generate_synthetic_data(
    n: int,
    corr: float,
    class_balance: List[float],
    l_probs: np.ndarray,
    l_groups: List[Set[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic label matrix L with given parameters.

    Parameters
    ----------
    n
        Number of rows in label matrix.
    corr
        Float factor determining strength of correlation between dependent columns.
    class _balance
        List of priors for each label value.
    l_probs
        Conditional probabilities of each entry in each row of the label matrix given true label.
    l_groups
        List of sets of dependent (correlated) columns.

    Returns
    -------
    y_true: 1D int np array of true labels.
    l: 2D int np array representing label matrix
    """
    cardinality = len(class_balance)
    y_true = np.random.choice(cardinality, n, p=class_balance)

    def generate_correlated(num: int = 2):
        """Generate num correlated label columns."""
        ls = [[] for _ in range(num)]
        for y in y_true:
            if np.random.choice(2, p=[1.0 - corr, corr]):
                v = np.random.choice(cardinality + 1, p=l_probs[y])
                for l in ls:
                    l.append(v)
            else:
                for l in ls:
                    l.append(np.random.choice(cardinality + 1, p=l_probs[y]))
        return [np.array(l) for l in ls]

    def generate_ls(sets):
        """Generate label columns given sets of dependent indexes."""
        ls = [None] * sum(map(len, sets))
        for s in sets:
            ls_gen = generate_correlated(num=len(s))
            for i, idx in enumerate(s):
                ls[idx] = ls_gen[i]
        return ls

    ls = generate_ls(l_groups)
    l = np.vstack(ls).T - 1
    return y_true, l


class LabelModelTest(unittest.TestCase):
    def _set_up_model(
        self,
        L: np.ndarray,
        class_balance: List[float] = [0.5, 0.5],
        deps: List[Tuple[int, int]] = [(0, 2)],
    ):
        label_model = DependencyAwareLabelModel(
            cardinality=len(class_balance), verbose=False
        )
        label_model.train_config = TrainConfig()  # type: ignore
        L_shift = L + 1
        label_model._set_constants(L_shift)
        label_model._set_class_balance(class_balance, None)
        label_model._set_structure(deps)
        label_model._generate_O(L_shift)
        label_model._init_params()
        return label_model

    def test_generate_O(self):
        L = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]])
        label_model = self._set_up_model(L)

        # O = (1/n) * L^TL = L^TL/5
        true_O = (
            np.array(
                [
                    [3, 0, 1, 2, 2, 1, 2, 1, 0, 0],
                    [0, 2, 1, 1, 1, 1, 0, 0, 1, 1],
                    [1, 1, 2, 0, 1, 1, 1, 0, 0, 1],
                    [2, 1, 0, 3, 2, 1, 1, 1, 1, 0],
                    [2, 1, 1, 2, 3, 0, 2, 0, 1, 0],
                    [1, 1, 1, 1, 0, 2, 0, 1, 0, 1],
                    [2, 0, 1, 1, 2, 0, 2, 0, 0, 0],
                    [1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
                ]
            )
            / 5.0
        )
        np.testing.assert_array_almost_equal(label_model.O.numpy(), true_O)

    def test_augmented_L_construction(self):
        L = np.array([[0, -1, 2], [-1, 0, 1], [2, 1, 0], [1, 2, 1]])
        L_shift = L + 1
        lm = DependencyAwareLabelModel(cardinality=3, verbose=False)
        lm._set_constants(L_shift)
        lm._set_class_balance(None, None)
        lm._set_structure([(0, 2)])
        L_aug = lm._get_augmented_label_matrix(L_shift)
        expected_L = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        np.testing.assert_array_almost_equal(L_aug, expected_L)

        # Check the clique entries
        print(lm.c_tree.node[0])
        # Size 2 clique 0
        self.assertSetEqual(lm.c_tree.node[0]["members"], frozenset({0, 2}))
        self.assertEqual(lm.c_tree.node[0]["start_index"], 9)
        self.assertEqual(lm.c_tree.node[0]["end_index"], 18)
        # Singleton clique 1
        self.assertSetEqual(lm.c_tree.node[1]["members"], frozenset({1}))
        self.assertEqual(lm.c_tree.node[1]["start_index"], 3)
        self.assertEqual(lm.c_tree.node[1]["end_index"], 6)

    def test_init_params(self):
        L = np.array([[0, 1, 0], [0, -1, 0]])
        label_model = self._set_up_model(L, class_balance=[0.6, 0.4])

        # mu_init = P(\lf=y|Y=y) = clamp(P(\lf=y) * prec_i / P(Y=y), (0,1))
        # mu_init[lf0, lf2 = 1 | Y = 1] = clamp(1.0 * 0.7 / 0.6) = 1.0 since P(lf = 1) = 1.0
        # mu_init[lf1 | Y = 2] = clamp(0.5 * 0.7 / 0.4) = 0.875 since P(lf = 2) = 0.5
        mu_init = label_model.mu_init.numpy()
        true_mu_init = np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.875],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(mu_init, true_mu_init)

        # mu_init = P(\lf=y|Y=y) = clamp(P(\lf=y) * prec_i / P(Y=y), (0,1))
        # mu_init[lf0, lf2 = 1 | Y = 1] = clamp(1.0 * 0.7 / 0.6) = 1.0 since P(lf = 1) = 1.0
        # mu_init[lf1 = 2 | Y = 2] = clamp(0.5 * 0.7 / 0.7) = 0.5 since P(lf = 2) = 0.5
        label_model._set_class_balance(class_balance=[0.3, 0.7], Y_dev=None)
        label_model._init_params()

        mu_init = label_model.mu_init.numpy()
        true_mu_init = np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.5],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(mu_init, true_mu_init)

    def test_build_mask(self):
        L = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]])
        label_model = self._set_up_model(L)

        # block diagonal with 0s for dependent LFs
        # without deps, k X k block of 0s down diagonal
        true_mask = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            ]
        )

        mask = label_model.mask.numpy()
        np.testing.assert_array_equal(mask, true_mask)


@pytest.mark.complex
class TestLabelModelAdvanced(unittest.TestCase):
    """Advanced (marked complex) tests for the LabelModel."""

    def test_label_model_basic(self) -> None:
        """Test the LabelModel's estimate of P and Y on a simple synthetic dataset."""
        cond_probs = np.array([[0.5, 0.4, 0.1], [0.5, 0.1, 0.4]])
        y_true, L = generate_synthetic_data(
            500000, 0.8, [0.3, 0.7], cond_probs, [{0, 1, 4}, {2}, {3}, {5}, {6}]
        )

        np.random.seed(123)

        # Train LabelModel
        lm = LabelModel(cardinality=2)
        lm.fit(L, n_epochs=5000)
        score = lm.score(L, y_true)

        # Train DependencyAwareLabelModel
        dalm = DependencyAwareLabelModel(cardinality=2)
        dalm.fit_with_deps(L, deps=[(0, 1), (0, 4), (1, 4)], n_epochs=5000)
        d_score = dalm.score(L, y_true)

        lf_accuracies = dalm.get_weights()
        for acc in lf_accuracies:
            self.assertAlmostEqual(acc, 0.8, delta=0.05)

        # Test predicted labels
        self.assertGreaterEqual(score["accuracy"], 0.75)
        self.assertGreaterEqual(d_score["accuracy"], score["accuracy"])

    def test_label_model_multiclass(self) -> None:
        """Test the LabelModel's estimate of P and Y on a simple synthetic dataset."""
        cond_probs = np.array(
            [[0.5, 0.4, 0.05, 0.05], [0.5, 0.05, 0.4, 0.05], [0.5, 0.05, 0.05, 0.4]]
        )
        y_true, L = generate_synthetic_data(
            100000, 0.8, [0.3, 0.5, 0.2], cond_probs, [{0, 1}, {2, 4}, {3}, {5}, {6}]
        )

        np.random.seed(123)

        # Train LabelModel
        lm = LabelModel(cardinality=3)
        lm.fit(L, n_epochs=5000)
        score = lm.score(L, y_true)

        # Train DependencyAwareLabelModel
        dalm = DependencyAwareLabelModel(cardinality=3)
        dalm.fit_with_deps(L, deps=[(0, 1), (2, 4)], n_epochs=5000)
        d_score = dalm.score(L, y_true)

        # Test predicted labels
        self.assertGreaterEqual(score["accuracy"], 0.75)
        self.assertGreaterEqual(d_score["accuracy"], score["accuracy"])


if __name__ == "__main__":
    unittest.main()
