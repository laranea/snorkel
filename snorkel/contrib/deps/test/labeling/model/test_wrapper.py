import unittest
from typing import List, Set, Tuple

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, train_test_split

from snorkel.contrib.deps.labeling.model.wrapper import SklearnDependencyAwareLabelModel


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


@pytest.mark.complex
class DependencyAwareLabelModelWrapperTest(unittest.TestCase):
    def test_search_deps(self):
        cond_probs = np.array([[0.5, 0.4, 0.1], [0.5, 0.1, 0.4]])
        y_true, L = generate_synthetic_data(
            100000, 0.8, [0.3, 0.7], cond_probs, [{0, 1, 4}, {2}, {3}, {5}, {6}]
        )
        np.random.seed(123)
        L_train, L_dev, Y_train, Y_dev = train_test_split(
            L, y_true, test_size=0.2, random_state=123
        )

        # Train LabelModel
        label_model = SklearnDependencyAwareLabelModel()
        L, Y, cv_split = label_model.create_param_search_data(L_train, L_dev, Y_dev)

        param_grid = [{"thresh_mult": [1.0, 0.5], "metric": ["accuracy"]}]
        clf = GridSearchCV(label_model, param_grid, cv=cv_split)
        clf.fit(L, Y)

        # pick the option with dependencies (thresh < 1.0)
        self.assertEqual(clf.best_params_, {"thresh_mult": 0.5, "metric": "accuracy"})
        self.assertEqual(clf.best_index_, 1)

    def test_search_ind(self):
        cond_probs = np.array([[0.5, 0.4, 0.1], [0.5, 0.1, 0.4]])
        y_true, L = generate_synthetic_data(
            100000, 0.8, [0.3, 0.7], cond_probs, [{0}, {1}, {2}, {3}, {4}, {5}, {6}]
        )
        np.random.seed(123)
        L_train, L_dev, Y_train, Y_dev = train_test_split(
            L, y_true, test_size=0.2, random_state=123
        )

        # Train LabelModel
        label_model = SklearnDependencyAwareLabelModel()
        L, Y, cv_split = label_model.create_param_search_data(L_train, L_dev, Y_dev)

        param_grid = [{"thresh_mult": [1.0, 0.75], "metric": ["accuracy"]}]
        clf = GridSearchCV(label_model, param_grid, cv=cv_split, error_score=0.0)
        clf.fit(L, Y)

        # pick the option with no dependencies (thresh = 1.0)
        self.assertEqual(clf.best_params_, {"thresh_mult": 1.0, "metric": "accuracy"})
        self.assertEqual(clf.best_index_, 0)


if __name__ == "__main__":
    unittest.main()
