import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose = False):
        self.verbose = verbose
        self.bags = bags
        self.boost = boost
        self.learner = []
        kwargs = {} if kwargs is None else kwargs
        for i in range(bags):
            self.learner.append(learner(**kwargs))

    def add_evidence(self, data_x, data_y):
        X = np.asarray(data_x, dtype= float)
        y = np.asarray(data_y, dtype= float)
        n_rows = X.shape[0]

        for model in self.learner:
            indices = np.random.choice(n_rows, size=n_rows)
            model.add_evidence(X[indices], y[indices])

    def query(self, points):
        all_pred = []
        for model in self.learner:
            pred = model.query(points)
            all_pred.append(pred)

        return np.mean(np.array(all_pred), axis=0)

    def author(self):
        return "stwisha3"

    def study_group(self):
        return "stwisha3"






