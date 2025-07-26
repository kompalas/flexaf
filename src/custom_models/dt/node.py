class TreeNode:
    def __init__(self, is_leaf, prediction=None, feature_index=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction              # For leaf nodes
        self.feature_index = feature_index        # For decision nodes
        self.threshold = threshold                # For decision nodes
        self.left = left                          # Left subtree
        self.right = right                        # Right subtree

