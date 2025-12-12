import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = bool(verbose)
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner,kwargs={}, bags=20, boost=False, verbose=False) for _ in range(20)]
    def add_evidence(self, X, y):
        for L in self.learners:
            L.add_evidence(X, y)
    def query(self, X):
        preds = [L.query(X) for L in self.learners]
        return np.mean(np.vstack(preds), axis=0)
    def author(self):
        return "stwisha3"