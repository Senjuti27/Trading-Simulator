""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import warnings  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  


class DTLearner(object):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		 	 	 		  		  		    	 		 		   		 		  
    your own correct DTLearner from Project 3.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type leaf_size: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = int(max(1, leaf_size))
        self.verbose = bool(verbose)
        self.tree = None


    def add_evidence(self, data_x, data_y):
        """
        Train the tree
        """
        X = np.asarray(data_x, dtype=float)
        y = np.asarray(data_y, dtype=float).reshape(-1)
        self.tree = self._build(X, y)

        # print("------------------------------")
        # for line in self.tree:
        #     print(line)


    def _build(self, X, y):
        """
        Recursively construct a subtree and return it as an (M, 4) ndarray.
        Stop if:
          - node has <= leaf_size rows
          - all y values are identical
          - all feature rows are identical (no further split possible)
        """
        # Stopping conditions
        if (
            X.shape[0] <= self.leaf_size          # too few samples
            or np.all(y == y[0])                  # all targets identical
            or np.all(X == X[0, :])               # all features rows identical
        ):
            """Create a leaf row."""
            leaf = np.array([[-1.0, float(np.mean(y)), np.nan, np.nan]], dtype=float)
            return leaf

        # Choose split feature by largest absolute correlation with y
        feat = self._best_feature_by_abs_corr(X, y)
        split = np.median(X[:, feat])

        # Partition
        left_mask = X[:, feat] <= split
        right_mask = ~left_mask

        # If split doesn't separate, make a leaf
        if not np.any(left_mask) or not np.any(right_mask):
            leaf = np.array([[-1.0, float(np.mean(y)), np.nan, np.nan]], dtype=float)
            return leaf

        # Recurse
        left_tree = self._build(X[left_mask], y[left_mask])
        right_tree = self._build(X[right_mask], y[right_mask])
        # Root row: [feat, split, 1, 1 + len(left)]
        root = np.array([[float(feat), float(split), 1.0, 1.0 + left_tree.shape[0]]], dtype=float)

        return np.vstack((root, left_tree, right_tree))

    def _best_feature_by_abs_corr(self, X, y):
        yc = y - y.mean()
        yden = np.sum(yc * yc)
        if yden == 0:
            return 0  # y constant; caller will stop soon

        best_j = 0
        best_score = -1.0

        for j in range(X.shape[1]):
            xj = X[:, j]
            xc = xj - xj.mean()
            den = np.sqrt(np.sum(xc * xc) * yden)
            score = 0.0 if den == 0 else abs(np.dot(xc, yc) / den)
            if score > best_score:
                best_score = score
                best_j = j
        return best_j

    def query(self, points):

        """Predict y for each row in points. Returns a 1-D float array (n_samples,)."""

        if self.tree is None:
            raise ValueError("Model not trained. Call add_evidence(...) first.")

        # Convert to float array and prep outputs
        X = np.asarray(points, dtype=float)
        T = self.tree
        preds = np.empty(X.shape[0], dtype=float)

        # Loop over each row in X
        for r in range(X.shape[0]):
            x = X[r]
            i = 0  # start at root row

            # Walk down the tree until we hit a leaf
            while True:
                feat = int(T[i, 0])  # feature index (or -1 for leaf)
                if feat == -1:  # leaf node
                    preds[r] = T[i, 1]  # store prediction
                    break

                split = T[i, 1]  # threshold value
                left_off = int(T[i, 2])  # usually 1
                right_off = int(T[i, 3])  # 1 + size(left_subtree)

                # Decide which child to jump to
                i = i + left_off if x[feat] <= split else i + right_off

        return preds

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "stwisha3"  # replace tb34 with your Georgia Tech username.

    def study_group(self):
        """
        :return: A comma-separated string of GT usernames of your study group members.
                 If working alone, just return your username.
        :rtype: str
        """
        return "stwisha3"  # replace/add other usernames if you have a study group

