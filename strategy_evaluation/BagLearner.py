"""
Implements a generic bagging ensemble that trains multiple base learners on bootstrap samples
of the data and combines their predictions via majority vote to produce robust classifications.
"""

import numpy as np
from scipy import stats

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose = False):
        self.verbose = verbose
        self.bags = bags
        self.boost = boost
        self.kwargs = {} if kwargs is None else kwargs

        # build the forest of learners
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**self.kwargs))

    def add_evidence(self, data_x, data_y):
        """
        Train each model on a bootstrap sample of the data.
        Bootstrap sample = sample N rows with replacement.
        """
        X = np.asarray(data_x, dtype= float)
        y = np.asarray(data_y, dtype= float)
        n_rows = X.shape[0]

        for model in self.learners:
            sample_idx = np.random.choice(n_rows, size=n_rows, replace=True)
            model.add_evidence(X[sample_idx], y[sample_idx])

    def query(self, features):
        """
        Query all learners and return the per-sample majority vote.
        """
        X = np.asarray(features, dtype=float)
        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([])

        # preds shape: (bags, n_samples)
        preds = np.zeros((self.bags, n_samples))
        for i in range(self.bags):
            preds[i, :] = self.learners[i].query(X)

        # majority vote along the bag axis; stats.mode returns an object; .mode is the array of modes
        majority = stats.mode(preds, axis=0)[0][0]
        return majority

    def author(self):
        return "stwisha3"

    def study_group(self):
        return "stwisha3"






