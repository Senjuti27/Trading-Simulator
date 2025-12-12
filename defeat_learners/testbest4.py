""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Test best4 data generator.  (c) 2016 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import math  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import DTLearner as dt  		  	   		 	 	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  		  	   		 	 	 		  		  		    	 		 		   		 		  
from gen_data import best_4_dt, best_4_lin_reg  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
# compare two learners' rmse out of sample  		  	   		 	 	 		  		  		    	 		 		   		 		  
def compare_os_rmse(learner1, learner2, x, y):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Compares the out-of-sample root mean squared error of your LinRegLearner and DTLearner.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param learner1: An instance of LinRegLearner  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type learner1: class:'LinRegLearner.LinRegLearner'  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param learner2: An instance of DTLearner  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type learner2: class:'DTLearner.DTLearner'  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param x: X data generated from either gen_data.best_4_dt or gen_data.best_4_lin_reg  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type x: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param y: Y data generated from either gen_data.best_4_dt or gen_data.best_4_lin_reg  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type y: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The root mean squared error of each learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train_rows = int(math.floor(0.6 * x.shape[0]))  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_rows = x.shape[0] - train_rows  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train = np.random.choice(x.shape[0], size=train_rows, replace=False)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test = np.setdiff1d(np.array(range(x.shape[0])), train)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train_x = x[train, :]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train_y = y[train]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_x = x[test, :]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_y = y[test]  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # train the learners  		  	   		 	 	 		  		  		    	 		 		   		 		  
    learner1.add_evidence(train_x, train_y)  # train it  		  	   		 	 	 		  		  		    	 		 		   		 		  
    learner2.add_evidence(train_x, train_y)  # train it  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # evaluate learner1 out of sample  		  	   		 	 	 		  		  		    	 		 		   		 		  
    pred_y = learner1.query(test_x)  # get the predictions  		  	   		 	 	 		  		  		    	 		 		   		 		  
    rmse1 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # evaluate learner2 out of sample  		  	   		 	 	 		  		  		    	 		 		   		 		  
    pred_y = learner2.query(test_x)  # get the predictions  		  	   		 	 	 		  		  		    	 		 		   		 		  
    rmse2 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return rmse1, rmse2  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Performs a test of your code and prints the results  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # create two learners and get data  		  	   		 	 	 		  		  		    	 		 		   		 		  
    lrlearner = lrl.LinRegLearner(verbose=False)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    dtlearner = dt.DTLearner(verbose=False, leaf_size=1)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    x, y = best_4_lin_reg()  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # compare the two learners  		  	   		 	 	 		  		  		    	 		 		   		 		  
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # share results  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print()  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("best_4_lin_reg() results")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"RMSE LR    : {rmse_lr}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"RMSE DT    : {rmse_dt}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if rmse_lr < 0.9 * rmse_dt:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        print("LR < 0.9 DT:  pass")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        print("LR >= 0.9 DT:  fail")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # get data that is best for a random tree  		  	   		 	 	 		  		  		    	 		 		   		 		  
    lrlearner = lrl.LinRegLearner(verbose=False)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    dtlearner = dt.DTLearner(verbose=False, leaf_size=1)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    x, y = best_4_dt()  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # compare the two learners  		  	   		 	 	 		  		  		    	 		 		   		 		  
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # share results  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print()  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("best_4_dt() results")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"RMSE LR    : {rmse_lr}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"RMSE DT    : {rmse_dt}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if rmse_dt < 0.9 * rmse_lr:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        print("DT < 0.9 LR:  pass")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        print("DT >= 0.9 LR:  fail")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
# if __name__ == "__main__":
#     test_code()

if __name__ == "__main__":
    import numpy as np
    from gen_data import best_4_lin_reg, best_4_dt
    from LinRegLearner import LinRegLearner
    from DTLearner import DTLearner

    test_code()

    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    def run_once(gen_fn, seed):
        X, y = gen_fn(seed)
        n = len(y)
        rs = np.random.RandomState(seed ^ 0xBAD5EED)
        idx = rs.permutation(n)
        ntr = int(0.6 * n)
        Xtr, Ytr = X[idx[:ntr]], y[idx[:ntr]]
        Xte, Yte = X[idx[ntr:]], y[idx[ntr:]]

        lr = LinRegLearner()
        lr.add_evidence(Xtr, Ytr)
        ylr = lr.query(Xte)

        dt = DTLearner(leaf_size=1)
        dt.add_evidence(Xtr, Ytr)
        ydt = dt.query(Xte)

        return rmse(Yte, ylr), rmse(Yte, ydt)

    # Generate 100 random seeds
    seeds = np.random.randint(0, 2**31 , size=100)

    # Test best_4_lin_reg
    success_lin = 0
    for s in seeds:
        r_lr, r_dt = run_once(best_4_lin_reg, int(s))
        if r_lr < 0.9 * r_dt:   # LinReg must be 10% better
            success_lin += 1
    print(f"best_4_lin_reg: {success_lin}/100 successes")

    # Test best_4_dt
    success_dt = 0
    for s in seeds:
        r_lr, r_dt = run_once(best_4_dt, int(s))
        if r_dt < 0.9 * r_lr:   # DT must be 10% better
            success_dt += 1
    print(f"best_4_dt: {success_dt}/100 successes")
