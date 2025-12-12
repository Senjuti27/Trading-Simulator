import numpy as np

class RTLearner(object):
    """
    Node layout (each row):
      - Internal: [feature_idx, split_value, left_offset, right_offset]
      - Leaf    : [-1, prediction_value, nan, nan]
    """

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = int(max(1, leaf_size))
        self.verbose = bool(verbose)
        self.tree = None
        self._rng = np.random.default_rng()

    def add_evidence(self, data_x, data_y):
        """
        Train the tree.
        """
        X = np.asarray(data_x, dtype=float)
        y = np.asarray(data_y, dtype=float).reshape(-1)
        self.tree = self._build(X, y)

    def _build(self, X, y):
        """
        Recursively construct a subtree and return it as an (M, 4) ndarray.
        Stop if:
          - node has <= leaf_size rows
          - all y values are identical
          - all feature rows are identical (no further split possible)
        """
        # Stopping conditions → make a leaf
        if (
            X.shape[0] <= self.leaf_size
            or np.all(y == y[0])
            or np.all(X == X[0, :])
        ):
            return np.array([[-1.0, float(np.mean(y)), np.nan, np.nan]], dtype=float)

        # Pick a random feature that actually separates the data; else make a leaf
        feat, split, ok = self._random_feature_and_split(X)
        if not ok:
            return np.array([[-1.0, float(np.mean(y)), np.nan, np.nan]], dtype=float)

        # Partition
        col = X[:, feat]
        left_mask  = col <= split
        right_mask = ~left_mask

        # Recurse
        left_tree  = self._build(X[left_mask],  y[left_mask])
        right_tree = self._build(X[right_mask], y[right_mask])

        # Root row: [feat, split, 1, 1 + len(left)]
        root = np.array([[float(feat), float(split), 1.0, 1.0 + left_tree.shape[0]]], dtype=float)
        return np.vstack((root, left_tree, right_tree))

    def _random_feature_and_split(self, X):
        """
        Try features in random order; for the first one that yields a valid split,
        return (feat_index, median_split, True). If none work, return (0, 0.0, False).
        """
        d = X.shape[1]
        for j in self._rng.permutation(d):
            col = X[:, j]
            # skip constant columns
            if np.all(col == col[0]):
                continue
            split = float(np.median(col))
            left = col <= split
            if np.any(left) and np.any(~left):
                return int(j), split, True

        return 0, 0.0, False

    def query(self, points):
        """Predict y for each row in points. Returns a 1-D float array (n_samples,)."""
        if self.tree is None:
            raise ValueError("Model not trained. Call add_evidence(...) first.")

        X = np.asarray(points, dtype=float)
        T = self.tree
        preds = np.empty(X.shape[0], dtype=float)

        for r in range(X.shape[0]):
            i = 0  # start at root row
            while True:
                feat = int(T[i, 0])  # -1 → leaf
                if feat == -1:
                    preds[r] = T[i, 1]
                    break
                split = T[i, 1]
                left_off  = int(T[i, 2])
                right_off = int(T[i, 3])
                i = i + (left_off if X[r, feat] <= split else right_off)

        return preds

    def author(self):
        return "stwisha3"

    def study_group(self):
        return "stwisha3"
