from collections import Counter
from src.custom_models.dt.node import TreeNode
from src.utils import best_split


class CustomDecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, feature_usage_limit=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_usage_limit = feature_usage_limit
        self.root = None

    def fit(self, X, y):
        n_features = len(X[0])
        initial_feature_usage = {i: 0 for i in range(n_features)}
        self.root = build_tree(X, y, depth=0, max_depth=self.max_depth,
                               min_samples_split=self.min_samples_split,
                               feature_usage=initial_feature_usage,
                               feature_limit=self.feature_usage_limit)

    def predict_one(self, x, node):
        if node.is_leaf:
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return [self.predict_one(x, self.root) for x in X]


def build_tree(X, y, depth, max_depth, min_samples_split, feature_usage, feature_limit):
    # Stopping conditions
    if depth >= max_depth or len(set(y)) == 1 or len(X) < min_samples_split:
        prediction = Counter(y).most_common(1)[0][0]
        return TreeNode(is_leaf=True, prediction=prediction)

    # Restrict features based on usage
    features_to_consider = [f for f, count in feature_usage.items() if count < feature_limit]
    if not features_to_consider:
        prediction = Counter(y).most_common(1)[0][0]
        return TreeNode(is_leaf=True, prediction=prediction)
    
    # Find best split
    best_feature, best_threshold, split_data = best_split(X, y, features_to_consider)
    if best_feature is None:
        prediction = Counter(y).most_common(1)[0][0]
        return TreeNode(is_leaf=True, prediction=prediction)

    # Update usage for path-based constraint
    updated_usage = feature_usage.copy()
    updated_usage[best_feature] += 1

    X_left, y_left, X_right, y_right = split_data
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split, updated_usage, feature_limit)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split, updated_usage, feature_limit)
    
    return TreeNode(
        is_leaf=False,
        feature_index=best_feature,
        threshold=best_threshold,
        left=left_subtree,
        right=right_subtree
    )
