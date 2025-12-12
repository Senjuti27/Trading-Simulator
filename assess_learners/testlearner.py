""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    # inf = open(sys.argv[1])
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    data = data[1:, 1:]  # remove date and header

# # pick file: arg if given, else default to Data/simple.csv
    # path = sys.argv[1] if len(sys.argv) == 2 else os.path.join(os.path.dirname(__file__), "Data", "Istanbul.csv")
    # print(f"[using] {path}")
    #
    # # read CSV (allowed: genfromtxt). If Istanbul, skip header & drop date col.
    # skip_header = 1 if path.endswith("Istanbul.csv") else 0
    # data = np.genfromtxt(path, delimiter=",", skip_header=skip_header)
    #
    # # handle 1-row files
    # if data.ndim == 1:
    #     data = data.reshape(1, -1)
    #
    # # drop date col for Istanbul
    # if path.endswith("Istanbul.csv") and data.shape[1] > 1:
    #     data = data[:, 1:]
    #
    # # optional: drop NaN rows
    # if np.isnan(data).any():
    #     data = data[~np.isnan(data).any(axis=1)]

    rng = np.random.default_rng(0)  # or: rng = np.random.default_rng()
    perm = rng.permutation(data.shape[0])  # random order of row indices
    data = data[perm]  # shuffle rows in-place for this ru

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(test_x.shape)
    print(test_y.shape)

    # create a learner and train it
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")

    print("##################################################")

    ### Q1: DTLearner testing for leaf_size 1-51
    rmse_in_sample = []
    rmse_out_sample = []
    for i in range(1, 51):
        # create a learner and train it
        learner = dt.DTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)
        # evaluate in sample
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmse_in_sample.append(rmse)

        print("In sample results")
        print(f"RMSE: {rmse_in_sample}")
        c = np.corrcoef(pred_y, y=train_y)
        print(f"corr: {c[0, 1]}")


        # evaluate out of sample
        pred_y2 = learner.query(test_x)
        rmse2 = math.sqrt(((test_y - pred_y2) ** 2).sum() / test_y.shape[0])
        rmse_out_sample.append(rmse2)

        print("Out of sample results")
        print(f"RMSE: {rmse_out_sample}")
        c = np.corrcoef(pred_y2, y=test_y)
        print(f"corr: {c[0, 1]}")

    # plotting the figure
    plt.plot(rmse_in_sample)
    plt.plot(rmse_out_sample)
    plt.title("RMSE vs Leaf Size for DTLearner")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 50)
    plt.ylabel("RMSE")
    # plt.ylim(0, 0.01)
    plt.grid(True)
    plt.legend(["In-Sample", "Out-Sample"])
    plt.savefig("Q1.png")
    # plt.show()
    plt.close("all")

    print("##################################################")

    ### Q2: RTLearner testing for leaf_size 1-50
    rmse_in_sample = []
    rmse_out_sample = []
    for i in range(1, 51):
        # create a learner and train it
        learner = rt.RTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)
        # evaluate in sample
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmse_in_sample.append(rmse)

        print("In sample results")
        print(f"RMSE: {rmse_in_sample}")
        c = np.corrcoef(pred_y, y=train_y)
        print(f"corr: {c[0, 1]}")

        # evaluate out of sample
        pred_y2 = learner.query(test_x)
        rmse2 = math.sqrt(((test_y - pred_y2) ** 2).sum() / test_y.shape[0])
        rmse_out_sample.append(rmse2)

        print("Out of sample results")
        print(f"RMSE: {rmse_out_sample}")
        c = np.corrcoef(pred_y2, y=test_y)
        print(f"corr: {c[0, 1]}")

    print("##################################################")


    ### Q2: BagLearner testing for leaf_size 1-50: using different learners for 20 bags
    rmse_in_sample = []
    rmse_out_sample = []
    for i in range(1, 51):
        # create a learner and train it
        # learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=False)
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=False)
        # learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=False)
        # learner = bl.BagLearner(learner=bl.BagLearner, kwargs={"learner": dt.DTLearner, "kwargs": {"leaf_size": i} , "bags" :20, "boost": False}, bags=20, boost=False, verbose=False)

        learner.add_evidence(train_x, train_y)
        # evaluate in sample
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmse_in_sample.append(rmse)

        print("In sample results")
        print(f"RMSE: {rmse_in_sample}")
        c = np.corrcoef(pred_y, y=train_y)
        print(f"corr: {c[0, 1]}")

        # evaluate out of sample
        pred_y2 = learner.query(test_x)
        rmse2 = math.sqrt(((test_y - pred_y2) ** 2).sum() / test_y.shape[0])
        rmse_out_sample.append(rmse2)

        print("Out of sample results")
        print(f"RMSE: {rmse_out_sample}")
        c = np.corrcoef(pred_y2, y=test_y)
        print(f"corr: {c[0, 1]}")

    # plotting the figure
    plt.plot(rmse_in_sample)
    plt.plot(rmse_out_sample)
    plt.title("RMSE vs Leaf Size for BagLearner with 20 Bags")
    plt.xlabel("Leaf Size")
    plt.grid(True)
    plt.xlim(1, 50)
    plt.ylabel("RMSE")
    # plt.ylim(0, 0.01)
    # plt.show()
    plt.legend(["In-Sample", "Out-Sample"])
    plt.savefig("Q2.png")
    plt.close("all")

    ### Q3: DTLearner vs RTLearner
    # a) measuring training time for both with leaf_size 1-20
    time_dt = []
    time_rt = []
    # create a DTlearner and train it
    for i in range(1, 51):
        start = time.time()
        learner = dt.DTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)
        end = time.time()
        time_dt.append(end - start)
    # create a RTlearner and train it
    for j in range(1, 51):
        start2 = time.time()
        learner2 = rt.RTLearner(leaf_size=j, verbose=True)
        learner2.add_evidence(train_x, train_y)
        end2 = time.time()
        time_rt.append(end2 - start2)

    # plotting the figure
    plt.plot(time_dt)
    plt.plot(time_rt)
    plt.grid(True)
    plt.title("Training Time for DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 50)
    plt.ylabel("Time (s)")
    # plt.ylim(0, 2)
    plt.legend(["DTLearner", "RTLearner"])
    plt.savefig("Q3a.png")
    plt.close("all")

    # b) measuring Mean Absolute Error (MAE) for training both with leaf_size 1-20
    mae_dt_train = []
    mae_rt_train = []
    mae_dt_test = []
    mae_rt_test = []
    for i in range(1, 51):
        # create a DTlearner and train it
        learner = dt.DTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)
        # create a RTlearner and train it
        learner2 = rt.RTLearner(leaf_size=i, verbose=True)
        learner2.add_evidence(train_x, train_y)
        # evaluate DTlearner in sample
        pred_y = learner.query(train_x)
        pred_y = np.array(pred_y)
        train_y = np.array(train_y)
        mae = np.mean(np.abs(train_y - pred_y))
        mae_dt_train.append(mae)

        pred_y_test = learner.query(test_x)
        pred_y_test = np.array(pred_y_test)
        test_y = np.array(test_y)
        mae = np.mean(np.abs(test_y - pred_y_test))
        mae_dt_test.append(mae)

        # evaluate RTlearner in sample
        pred_y2 = learner2.query(train_x)
        pred_y2 = np.array(pred_y2)
        mae2 = np.mean(np.abs(train_y - pred_y2))
        mae_rt_train.append(mae2)

        pred_y2_test = learner2.query(test_x)
        pred_y2_test = np.array(pred_y2_test)
        test_y = np.array(test_y)
        mae2 = np.mean(np.abs(test_y - pred_y2_test))
        mae_rt_test.append(mae2)

    # plotting the figure
    plt.plot(mae_dt_train)
    plt.plot(mae_dt_test)
    plt.plot(mae_rt_train)
    plt.plot(mae_rt_test)
    plt.grid(True)
    plt.title("MAE for DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 50)
    plt.ylabel("MAE")
    # plt.ylim(0, 0.8)
    plt.legend(["In sample DTLearner", "Out of sample DTLearner", "In sample RTLearner", "Out of sample RTLearner",  ])
    plt.savefig("Q3b.png")
    plt.close("all")


