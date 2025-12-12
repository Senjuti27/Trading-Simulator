""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Senjuti Twisha  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: stwisha3	  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 904080731   		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import math  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		 	 	 		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		 	 	 		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """
    np.random.seed(seed)
    rows, cols = 200, 5
    x = np.random.random(size=(rows, cols))* 200 - 100
    w = np.random.uniform(-1,1, size=cols)
    bias  = np.random.uniform(-2,2)
    y = np.dot(x,w) + bias
    return x,y

    # x = np.zeros((100, 2))
    # y = np.random.random(size=(100,)) * 200 - 100
    # Here's is an example of creating a Y from randomly generated  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # X with multiple columns  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # y = x[:,0] + np.sin(x[:,1]) + x[:,2]**2 + x[:,3]**3  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # return x, y

  		  	   		 	 	 		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    np.random.seed(seed)
    rows, cols = 200, 5
    x = np.random.random(size=(rows, cols)) * 200 - 100 # [0,200)-> [-100,100)
    y = np.zeros(rows)

    for c in range(cols):
        if c%3 == 0:
            y = y + np.cos(x[:, c])

        elif c%3 == 1:
            for i in range(rows):
                if x[i, c] > 0:
                    y[i] += -0.4 * (x[i, c]) ** 2
                else:
                    y[i] += 0.4 * (x[i, c]) ** 2

        else:
            for i in range(rows):
                if x[i, c] > 2:
                    y[i] += 0.06 * (x[i, c]) ** 3
                else:
                    y[i] += -0.02 * (x[i, c]) ** 3

    return x, y

    # x = np.zeros((100, 2))
    # y = np.random.random(size=(100,)) * 200 - 100
    # return x, y
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def author():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return "stwisha3"  # Change this to your user ID

def study_group():
    """
    :return: A comma-separated string of GT usernames of your study group members.
             If working alone, just return your username.
    :rtype: str
    """
    return "stwisha3"  # replace/add other usernames if you have a study group
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
# if __name__ == "__main__":
    # print("they call me Tim.")
