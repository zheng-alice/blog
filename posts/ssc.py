import numpy as np
from cvxpy import Variable, Minimize, norm, diag, Problem, ECOS, SolverError
from sklearn.cluster import SpectralClustering

n_samples = 200
seed = 5805
rng = np.random.RandomState(seed)

# The 'X'-looking dataset

linear = np.concatenate((
        rng.rand(n_samples//2, 2),                              # □ Square
        rng.rand(n_samples//4, 2) * 0.05
            + np.repeat(rng.rand(n_samples//4, 1), 2, axis=1),  # / Diagonal
        rng.rand(n_samples//4, 2) * 0.05
            + np.repeat(rng.rand(n_samples//4, 1), 2, axis=1)   # \ Diagonal
            * [1, -1] + [0, 1]
    ), axis=0) - [0.5, 0.5], None

class SparseSubspaceClustering:

    def __init__(self, n_clusters, use_E=True, use_Z=True):
        self.n_clusters = n_clusters
        self.use_E = use_E
        self.use_Z = use_Z

    def fit(self, Y):
        Y = Y.T
        d, n = Y.shape
        assert d < n

        ## 1. Solve sparse optimization problem

        # Starting form, equation (5)
        C = Variable((n, n))
        objective = norm(C,1)
        constraint = Y @ C

        # Account for outliers
        if self.use_E:
            mu_e = np.partition(np.sum(np.abs(Y), axis=0), -2)[-2]
            l_e = 20 / mu_e
            E = Variable((d, n))
            objective += l_e * norm(E,1)
            constraint += E

        # Account for noise
        if self.use_Z:
            G = np.abs(Y.T @ Y)
            mu_z = np.min(np.max(G - np.diag(np.diag(G)), axis=1))
            l_z = 800 / mu_z
            Z = Variable((d, n))
            objective += l_z/2 * norm(Z,"fro")**2
            constraint += Z

        constraints = [Y == constraint, diag(C) == 0]
        prob = Problem(Minimize(objective), constraints)
        try:
            prob.solve(solver=ECOS)

            ## 2. Normalize the columns of C
            C = C.value / np.max(np.abs(C.value), axis=0)

            ## 3. Form the weights of a similarity graph
            W = np.abs(C)
            W = W + W.T

            ## 4. Apply spectral clustering on the graph
            self.labels_ = SpectralClustering(
                    n_clusters=self.n_clusters,
                    affinity='precomputed'
                ).fit_predict(W)
        except SolverError:
            self.labels_ = np.zeros(n)
