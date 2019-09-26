import unittest
from collections import defaultdict

import numpy as np
from numpy.random import choice, random

from snorkel.contrib.deps.labeling.model.dep_learner import DependencyLearner


# Helper functions for data generator
def logistic_fn(x):
    return 1 / (1 + np.exp(-x))


def choose_other_label(k, y):
    """Given a cardinality k and true label y, return random value in
    {1,...,k} \ {y}."""
    return choice(list(set(range(1, k + 1)) - set([y])))


def indpm(x, y):
    """Plus-minus indicator function"""
    return 1 if x == y else -1


# Data generator for dependencies among LFs
class DepsDataGenerator(object):
    """Generates a synthetic single-task L and Y matrix with dependencies

    Args:
        n: (int) The number of data points
        m: (int) The number of labeling sources
        k: (int) The cardinality of the classification task
        class_balance: (np.array) each class's percentage of the population
        theta_range: (tuple) The min and max possible values for theta, the
            class conditional accuracy for each labeling source
        edge_prob: edge density in the graph of correlations between sources
        edges: (list) The list of edges representing correlations between sources
        theta_edge_range: The min and max possible values for theta_edge, the
            strength of correlation between correlated sources

    The labeling functions have class-conditional accuracies, and
    class-unconditional pairwise correlations forming a tree-structured graph.

    Note that k = the # of true classes; thus source labels are in {0,1,...,k}
    because they include abstains.
    """

    def __init__(
        self,
        n,
        m,
        k=2,
        class_balance=None,
        theta_range=(1.0, 1.5),
        edge_prob=0.0,
        edges=None,
        theta_edge_range=(0.75, 1),
        **kwargs,
    ):
        self.n = n
        self.m = m
        self.k = k

        if edges is None:
            # Generate correlation structure: edges self.E, parents dict self.parent
            self._generate_edges(edge_prob)
        else:
            # Save correlation structure: edges self.E, parents dict self.parent
            self._save_edges(edges)

        # Generate class-conditional LF & edge parameters, stored in self.theta
        self._generate_params(theta_range, theta_edge_range)

        # Generate class balance self.p
        if class_balance is None:
            self.p = np.full(k, 1 / k)
        else:
            self.p = class_balance

        # Generate the true labels self.Y and label matrix self.L
        self._generate_label_matrix()

        # Compute the conditional clique probabilities
        self._get_conditional_probs()

        # Correct output type
        self.L = np.array(self.L - 1, dtype=np.int)
        self.Y = self.Y - 1

    def _generate_edges(self, edge_prob):
        """Generate a random tree-structured dependency graph based on a
        specified edge probability.

        Also create helper data struct mapping child -> parent.
        """
        self.E, self.parent = [], {}
        for i in range(self.m):
            if random() < edge_prob and i > 0:
                p_i = choice(i)
                self.E.append((p_i, i))
                self.parent[i] = p_i

    def _save_edges(self, edges):
        """Save tree-structured dependency graph based on a
        specified list of edges.

        Also create helper data struct mapping child -> parent.
        """
        self.E, self.parent = [], {}
        for p_i, i in edges:
            self.E.append((p_i, i))
            self.parent[i] = p_i

    def _generate_params(self, theta_range, theta_edge_range):
        self.theta = defaultdict(float)
        for i in range(self.m):
            t_min, t_max = min(theta_range), max(theta_range)
            self.theta[i] = (t_max - t_min) * random(self.k + 1) + t_min

        # Choose random weights for the edges
        te_min, te_max = min(theta_edge_range), max(theta_edge_range)
        for (i, j) in self.E:
            w_ij = (te_max - te_min) * random() + te_min
            self.theta[(i, j)] = w_ij
            self.theta[(j, i)] = w_ij

    def _P(self, i, li, j, lj, y):
        return np.exp(
            self.theta[i][y] * indpm(li, y) + self.theta[(i, j)] * indpm(li, lj)
        )

    def P_conditional(self, i, li, j, lj, y):
        """Compute the conditional probability
            P_\theta(li | lj, y)
            =
            Z^{-1} exp(
                theta_{i|y} \indpm{ \lambda_i = Y }
                + \theta_{i,j} \indpm{ \lambda_i = \lambda_j }
            )
        In other words, compute the conditional probability that LF i outputs
        li given that LF j output lj, and Y = y, parameterized by
            - a class-conditional LF accuracy parameter \theta_{i|y}
            - a symmetric LF correlation paramter \theta_{i,j}
        """
        Z = np.sum([self._P(i, _li, j, lj, y) for _li in range(self.k + 1)])
        return self._P(i, li, j, lj, y) / Z

    def _generate_label_matrix(self):
        """Generate an [n,m] label matrix with entries in {0,...,k}"""
        self.L = np.zeros((self.n, self.m))
        self.Y = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            y = choice(self.k, p=self.p) + 1  # Note that y \in {1,...,k}
            self.Y[i] = y
            for j in range(self.m):
                p_j = self.parent.get(j, 0)
                prob_y = self.P_conditional(j, y, p_j, self.L[i, p_j], y)
                prob_0 = self.P_conditional(j, 0, p_j, self.L[i, p_j], y)
                p = np.ones(self.k + 1) * (1 - prob_y - prob_0) / (self.k - 1)
                p[0] = prob_0
                p[y] = prob_y
                self.L[i, j] = choice(self.k + 1, p=p)

    def _get_conditional_probs(self):
        """Compute the true clique conditional probabilities P(\lC | Y) by
        counting given L, Y; we'll use this as ground truth to compare to.

        Note that this generates an attribute, self.c_probs, that has the same
        definition as returned by `LabelModel.get_conditional_probs`.

        TODO: Can compute these exactly if we want to implement that.
        """
        # TODO: Extend to higher-order cliques again
        self.c_probs = np.zeros((self.m * (self.k + 1), self.k))
        for y in range(1, self.k + 1):
            Ly = self.L[self.Y == y]
            for ly in range(self.k + 1):
                self.c_probs[ly :: (self.k + 1), y - 1] = (
                    np.where(Ly == ly, 1, 0).sum(axis=0) / Ly.shape[0]
                )


class DependencyLearnerTest(unittest.TestCase):
    def test_cardinality(self):
        dep_learner = DependencyLearner()
        L = np.array([[-1, 0, 2], [1, 2, 3], [-1, -1, 0]])
        with self.assertRaisesRegex(
            ValueError, "Does not match DependencyLearner cardinality"
        ):
            dep_learner.fit(L)

    def test_singleton(self):
        dep_learner = DependencyLearner()
        deps = [(0, 1), (1, 2)]
        deps_correct = dep_learner._force_singleton(deps, 5)
        self.assertEqual(set(deps_correct), set([(0, 1), (0, 2), (1, 2)]))

    def test_fully_connected(self):
        dep_learner = DependencyLearner()
        deps = [(0, 1), (1, 2)]
        with self.assertRaisesRegex(
            ValueError, "Dependency structure is fully connected"
        ):
            dep_learner._force_singleton(deps, 3)

    def test_deps_from_inverse(self):
        thresh = 0.5
        J = np.array([[1.0, 0.75, 0.25], [0.95, 0.3, 0.5]])
        dep_learner = DependencyLearner()
        idx = dep_learner._get_deps_from_inverse_sig(J, thresh)
        self.assertEqual(set(idx), set([(0, 0), (0, 1), (1, 0)]))

    def test_no_deps(self):
        L = np.array([[1, 1, 1], [-1, 0, 1], [-1, -1, 0]])
        dep_learner = DependencyLearner()
        with self.assertRaisesRegex(
            ValueError, "Dependency structure is fully connected"
        ):
            dep_learner.fit(L)

    def test_dep_learning(self):
        np.random.seed(1234)
        K = 2
        M = 10
        N = 10000
        true_edges = [(0, 1), (2, 4)]

        data = DepsDataGenerator(N, M, K, edges=true_edges)
        dep_learner = DependencyLearner(cardinality=K)
        learned_deps = dep_learner.fit(data.L)
        self.assertEqual(set(learned_deps), set(true_edges))

        K = 5
        M = 12
        N = 10000
        true_edges = [(0, 1), (2, 4)]

        data = DepsDataGenerator(N, M, K, edges=true_edges)
        dep_learner = DependencyLearner(cardinality=K)
        learned_deps = dep_learner.fit(data.L)
        self.assertEqual(set(learned_deps), set(true_edges))


if __name__ == "__main__":
    unittest.main()