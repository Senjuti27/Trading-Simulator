"""
Implements a randomized decision tree learner that builds trees using random feature selection
and median splits, returning classification labels used as the base model in the Random Forest.
"""

import numpy as np

class RTLearner(object):
    """
    Randomized decision tree learner.
    Each internal node: [feature_idx, split_value, left_offset, right_offset]
    Each leaf node:     [-1, prediction_value, nan, nan]
    """

    def __init__(self, leaf_size, verbose=False):
        """
        leaf_size : minimum number of rows required to continue splitting.
        verbose   : unused, kept only for compatibility.
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None


    def _build(self, X, y):
        """
        Recursively build a randomized decision tree.
        Returns an array with nodes:
            Internal node: [feature_index, split_value, left_offset, right_offset]
            Leaf node    : [-1, prediction_value, nan, nan]
        """
        # Leaf conditions
        if (
            X.shape[0] <= self.leaf_size
            or np.all(y == y[0])
            or np.all(X == X[0, :])
        ):
            # return np.array([[-1.0, float(np.mean(y)), np.nan, np.nan]], dtype=float)
            # >>> CHANGED: classification leaf uses MODE (most common y), not MEAN
            vals, counts = np.unique(y, return_counts=True)
            leaf_value = vals[np.argmax(counts)]
            return np.array([[-1.0, float(leaf_value), np.nan, np.nan]], dtype=float)

        # Pick a useful random split
        feat, split, valid = self._random_feature_and_split(X)
        if not valid:
            # return np.array([[-1.0, float(np.mean(y)), np.nan, np.nan]], dtype=float)
            # >>> CHANGED: same as above, use MODE for fallback leaf
            vals, counts = np.unique(y, return_counts=True)
            leaf_value = vals[np.argmax(counts)]
            return np.array([[-1.0, float(leaf_value), np.nan, np.nan]], dtype=float)

        # Partition the data
        left_mask = X[:, feat] <= split
        right_mask = ~left_mask

        # Recurse
        left_tree  = self._build(X[left_mask],  y[left_mask])
        right_tree = self._build(X[right_mask], y[right_mask])

        # Root node format: [feat, split, 1, 1 + len(left)]
        root = np.array([[float(feat), float(split), 1.0, 1.0 + left_tree.shape[0]]], dtype=float)
        return np.vstack((root, left_tree, right_tree))

    def _random_feature_and_split(self, X):
        """
        Try features in random order; for the first one that yields a valid split,
        return (feat_index, median_split, True). If none work, return (0, 0.0, False).
        """
        num_feat = X.shape[1]
        for j in np.random.permutation(num_feat):
            col = X[:, j]

            # If feature constant → skip
            if np.all(col == col[0]):
                continue

            split = float(np.median(col))
            left = col <= split

            if np.any(left) and np.any(~left):
                return int(j), split, True

        return 0, 0.0, False

    def add_evidence(self, data_x, data_y):
        """
        Train the decision tree.
        """
        X = np.asarray(data_x, dtype=float)
        y = np.asarray(data_y, dtype=float).reshape(-1)
        self.tree = self._build(X, y)

    def query(self, X):
        """Predict y for each row in X.
        Traverses the tree until hitting a leaf node.
        Returns a 1-D float array (n_samples,)."""

        if self.tree is None:
            raise ValueError("Model not trained. Call add_evidence(...) first.")

        X = np.asarray(X, dtype=float)
        T = self.tree
        preds = np.empty(X.shape[0], dtype=float)

        for r in range(X.shape[0]):
            i = 0  # start at root row

            while True:
                feat = int(T[i, 0])
                #Leaf node
                if feat == -1:
                    preds[r] = T[i, 1]
                    break

                split = T[i, 1]
                left_off  = int(T[i, 2])
                right_off = int(T[i, 3])

                # Decide whether to go left or right based on the split
                if X[r, feat] <= split:
                    i = i + left_off
                else:
                    i = i + right_off
        return preds

    def author(self):
        return "stwisha3"

    def study_group(self):
        return "stwisha3"
