# Machine Learning Coding

## Real Problems

- mAP (object detection metric)
- self attention

### find local maxima with a decreasing window.

**Question.** Given integer array `rawData` and integer `localArea`, return the sorted 0-based indices `i` for which values decrease **strictly** as one moves away from `rawData[i]` up to `localArea` positions to the left and right. At an array boundary, use the available positions only.

**Answer.** Precompute the consecutive decreasing-run length in each direction. An index qualifies if both runs cover the required number of available neighbors. O(n) time and O(n) space.

```python
def solution(rawData, localArea):
    n = len(rawData)
    if n == 0:
        return []

    left = [0] * n
    right = [0] * n

    # Number of strictly decreasing steps while moving left from i.
    for i in range(1, n):
        if rawData[i] > rawData[i - 1]:
            left[i] = left[i - 1] + 1

    # Number of strictly decreasing steps while moving right from i.
    for i in range(n - 2, -1, -1):
        if rawData[i] > rawData[i + 1]:
            right[i] = right[i + 1] + 1

    answer = []
    for i in range(n):
        required_left = min(localArea, i)
        required_right = min(localArea, n - 1 - i)
        if left[i] >= required_left and right[i] >= required_right:
            answer.append(i)
    return answer


if __name__ == "__main__":
    print(solution([2, 10, 6, 3, 1, 2, 4, 6, 9, 5, 2], 2))
    # [1, 8]
```

**Edge cases.** Equal adjacent values break strict decrease. At index 0 or `n−1`, the missing side is vacuously satisfied, so an endpoint may qualify if its existing side decreases. If `localArea == 0`, every index qualifies.

### add parentheses to minimize an arithmetic expression.

**Question.** Given a string such as `"112+422"` consisting of two positive integers with no zero digits, insert exactly one pair of parentheses. The `+` must be inside, with at least one digit of each operand inside. Adjacent numeric factors multiply. Return the smallest possible integer value.

**Answer.** Enumerate where the opening parenthesis falls in the left operand and where the closing parenthesis falls in the right operand. Any prefix/suffix outside parentheses is a multiplicative factor; absent factors count as 1. Precompute the numeric values of all relevant substrings, then evaluate O(L×R) candidate expressions; overall O(n²) time for a length-n string (under the usual arithmetic-operation model).

```python
def solution(expr):
    left, right = expr.split("+")
    best = float("inf")

    left_outside = [int(left[:i]) if i else 1 for i in range(len(left))]
    left_inside = [int(left[i:]) for i in range(len(left))]
    right_inside = [int(right[:j]) for j in range(1, len(right) + 1)]
    right_outside = [
        int(right[j:]) if j < len(right) else 1
        for j in range(1, len(right) + 1)
    ]

    for i in range(len(left)):
        for j in range(len(right)):
            value = (
                left_outside[i] *
                (left_inside[i] + right_inside[j]) *
                right_outside[j]
            )
            best = min(best, value)

    return int(best)


if __name__ == "__main__":
    assert solution("112+422") == 108
    assert solution("55+55") == 110
    print(solution("112+422"), solution("55+55"))
```

**Correct minimal placement for `112+422`:** `1(12+42)2 = 108`. The assertion is for that placement. The screenshot also gives `"55+55" → 110` via `(55+55)`.

### implement k-means clustering.

**Question.** Given `data` (list of float feature vectors), `k` initial `centroids`, and `iterations`, repeatedly assign points to their nearest centroid, replace each centroid with its assigned points' mean, and return final integer cluster IDs. Empty clusters are retained at their previous locations in this reference implementation.

**Answer.** Use squared Euclidean distance because the square root does not affect the closest centroid. Reassign after the last centroid update.

```python
def solution(data, k, centroids, iterations):
    if not data:
        return []

    centers = [list(c) for c in centroids]
    dimension = len(data[0])
    assert len(centers) == k

    def assign(points, center_list):
        result = []
        for point in points:
            nearest = min(
                range(k),
                key=lambda j: sum(
                    (point[d] - center_list[j][d]) ** 2
                    for d in range(dimension)
                ),
            )
            result.append(nearest)
        return result

    for _ in range(iterations):
        labels = assign(data, centers)
        sums = [[0.0] * dimension for _ in range(k)]
        counts = [0] * k

        for point, label in zip(data, labels):
            counts[label] += 1
            for d in range(dimension):
                sums[label][d] += point[d]

        for j in range(k):
            if counts[j] > 0:
                centers[j] = [v / counts[j] for v in sums[j]]

    return assign(data, centers)


if __name__ == "__main__":
    points = [
        [2.0, 3.0], [-3.0, 2.0], [3.0, 3.0], [-3.0, 4.0],
        [-1.0, -1.0], [-2.0, -2.0], [-1.0, -2.0], [3.0, -2.0],
        [4.0, -5.0], [2.0, 2.0], [-3.0, 3.0], [3.0, -3.0],
    ]
    initial = [[2.0, 3.0], [-3.0, 4.0], [4.0, -5.0], [-1.0, -1.0]]
    print(solution(points, 4, initial, 10))
    # [0, 1, 0, 1, 3, 3, 3, 2, 2, 0, 1, 2]
```

**Complexity.** O(`iterations × n × k × d`). The cluster numbers are arbitrary labels; ties in this reference select the lower-index centroid.

### bagging / bootstrap decision trees.

**Question.** Given training features/labels, test features, and `n_estimators`, bootstrap a sample of size n **with replacement** for each classifier, fit it, predict the test data, and take a majority vote. On a tie, return the smaller class label. The screenshot requires sampling indices via `randint` and forbids using NumPy for the sampling step.

**Answer.** The following complete reference solution implements bootstrap voting and supplies a basic recursive decision-tree learner. The original CodeSignal scaffold appears to provide its own base estimators; if so, **keep those** and reuse only the bootstrap, fitting, and voting logic. This implementation does **not** re-seed the RNG inside the loop.

```python
import random


class BasicDecisionTree:
    def __init__(self, max_depth=8):
        self.max_depth = max_depth
        self.tree = None

    @staticmethod
    def _gini(labels):
        n = len(labels)
        if n == 0:
            return 0.0
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return 1.0 - sum((count / n) ** 2 for count in counts.values())

    def _build(self, X, y, depth):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        majority = min(counts, key=lambda label: (-counts[label], label))
        if len(counts) == 1 or depth >= self.max_depth:
            return {"label": majority}

        best_score = self._gini(y)
        best_split = None
        for feature in range(len(X[0])):
            values = sorted(set(row[feature] for row in X))
            for j in range(len(values) - 1):
                threshold = (values[j] + values[j + 1]) / 2.0
                left = [i for i, row in enumerate(X)
                        if row[feature] <= threshold]
                right = [i for i, row in enumerate(X)
                         if row[feature] > threshold]
                if not left or not right:
                    continue
                score = (
                    len(left) / len(y) * self._gini([y[i] for i in left]) +
                    len(right) / len(y) * self._gini([y[i] for i in right])
                )
                if score < best_score - 1e-12:
                    best_score = score
                    best_split = feature, threshold, left, right

        if best_split is None:
            return {"label": majority}
        feature, threshold, left, right = best_split
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build([X[i] for i in left], [y[i] for i in left], depth + 1),
            "right": self._build([X[i] for i in right], [y[i] for i in right], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def predict(self, X):
        result = []
        for x in X:
            node = self.tree
            while "feature" in node:
                branch = "left" if x[node["feature"]] <= node["threshold"] else "right"
                node = node[branch]
            result.append(node["label"])
        return result


class BaggingClassifier:
    def __init__(self, estimators):
        if not estimators:
            raise ValueError("At least one estimator is required")
        self.estimators = estimators

    @staticmethod
    def bootstrap(x_train, y_train):
        n = len(x_train)
        if n == 0:
            raise ValueError("Training set cannot be empty")
        # Sample n times WITH replacement. randint includes both endpoints.
        indices = [random.randint(0, n - 1) for _ in range(n)]
        return ([x_train[i] for i in indices],
                [y_train[i] for i in indices])

    def fit(self, x_train, y_train):
        for model in self.estimators:
            sample_x, sample_y = self.bootstrap(x_train, y_train)
            model.fit(sample_x, sample_y)
        return self

    def predict(self, x_test):
        predictions = [model.predict(x_test) for model in self.estimators]
        result = []
        for test_index in range(len(x_test)):
            counts = {}
            for pred in predictions:
                label = int(pred[test_index])
                counts[label] = counts.get(label, 0) + 1

            # Max vote count; on equal counts, choose smaller label.
            result.append(min(counts, key=lambda x: (-counts[x], x)))
        return result


def bagging_predict(x_train, y_train, x_test, estimators):
    model = BaggingClassifier(estimators)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def solution(x_train, y_train, x_test, n_estimators):
    estimators = [BasicDecisionTree() for _ in range(n_estimators)]
    return bagging_predict(x_train, y_train, x_test, estimators)


if __name__ == "__main__":
    random.seed(9)  # Demo only: defer to the CodeSignal scaffold's seed.
    X = [[0.0], [0.2], [0.4], [2.0], [2.2], [2.4]]
    y = [0, 0, 0, 1, 1, 1]
    print(solution(X, y, [[0.1], [2.1]], 11))
    # [0, 1]
```

**Using the provided CodeSignal skeleton:** The recollection specifically mentions filling in `bootstrap()`, `fit()`, and `predict()`. The reference class above separates those methods to match that workflow. Construct or use exactly `n_estimators` supplied base models and call the three methods while preserving the existing code and initial random seed. The sample standalone base learner uses Gini splits and a maximum tree depth of 8; the grader's base learners may differ, so exact labels for unseen inputs are not guaranteed to match that scaffold. The screenshot's model-specific implementation details are incomplete.

### implement a decision tree from scratch.

**Question.** Given `x_train`, `y_train`, and `x_test`, build a classifier whose internal nodes choose a feature/threshold, whose leaves predict a label, and whose splits maximize information gain based on entropy. Return predicted float class labels. No extra third-party libraries are allowed.

**Answer.** Below is a **self-contained reference** without imports. For continuous numerical features, it tests midpoints between sorted distinct training values. The helper `log2` uses a convergent series to honor the screenshot's “no importing libraries” restriction.

```python
def log2_no_import(x):
    if x <= 0:
        raise ValueError("log2 requires a positive number")
    exponent = 0
    while x < 1.0:
        x *= 2.0
        exponent -= 1
    while x >= 2.0:
        x /= 2.0
        exponent += 1

    t = (x - 1.0) / (x + 1.0)
    power = t
    series = 0.0
    for odd in range(1, 61, 2):
        series += power / odd
        power *= t * t
    return exponent + 2.0 * series / 0.6931471805599453


def entropy(labels):
    if not labels:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(labels)
    return -sum(
        (count / total) * log2_no_import(count / total)
        for count in counts.values()
    )


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def majority(labels):
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return min(counts, key=lambda label: (-counts[label], label))

    def fit(self, X, y):
        if not X:
            raise ValueError("Training set cannot be empty")
        self.root = self._grow(X, y, 0)
        return self

    def _grow(self, X, y, depth):
        prediction = self.majority(y)
        node = {"label": prediction}
        if (len(set(y)) == 1 or len(X) < 2 or
                (self.max_depth is not None and depth >= self.max_depth)):
            return node

        n = len(y)
        old_entropy = entropy(y)
        best_gain = 0.0
        best = None

        for feature in range(len(X[0])):
            values = sorted(set(row[feature] for row in X))
            for pos in range(len(values) - 1):
                threshold = (values[pos] + values[pos + 1]) / 2.0
                left_ids = [i for i in range(n) if X[i][feature] <= threshold]
                right_ids = [i for i in range(n) if X[i][feature] > threshold]
                if not left_ids or not right_ids:
                    continue

                left_y = [y[i] for i in left_ids]
                right_y = [y[i] for i in right_ids]
                gain = old_entropy - (
                    len(left_y) / n * entropy(left_y) +
                    len(right_y) / n * entropy(right_y)
                )
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (feature, threshold, left_ids, right_ids)

        if best is None:
            return node

        feature, threshold, left_ids, right_ids = best
        node["feature"] = feature
        node["threshold"] = threshold
        node["left"] = self._grow(
            [X[i] for i in left_ids], [y[i] for i in left_ids], depth + 1
        )
        node["right"] = self._grow(
            [X[i] for i in right_ids], [y[i] for i in right_ids], depth + 1
        )
        return node

    def predict_one(self, x):
        node = self.root
        while "feature" in node:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return float(node["label"])

    def predict(self, X):
        return [self.predict_one(x) for x in X]


def solution(x_train, y_train, x_test):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)


if __name__ == "__main__":
    print(solution([[0.0], [1.0], [2.0], [3.0]],
                   [0.0, 0.0, 1.0, 1.0], [[0.5], [2.5]]))
    # [0.0, 1.0]
```

**Information gain:** `IG = H(parent) − (n_left/n)H(left) − (n_right/n)H(right)`. Actual grader behavior can depend on its provided stopping, tie-breaking, and split conventions; those details are not fully visible in the screenshots.

### implement gradient descent for linear regression.

**Question.** Fit `y = b + Σ_j θ_j x_j` using mean squared error, random initial bias and coefficients, batch gradients, and a fixed number of gradient-descent iterations. Return predictions for `x_test`.

**Answer.** For `MSE = (1/n)Σ_i(y_i−ŷ_i)²`, the derivatives are

`d_b = (−2/n)Σ_i (y_i−ŷ_i)` and

`d_θ_j = (−2/n)Σ_i x_ij(y_i−ŷ_i)`.

```python
import random


def random_bias_theta(n_features):
    bias = random.random()
    theta = [random.random() for _ in range(n_features)]
    return bias, theta


def calculate_y(b, theta, x):
    return b + sum(weight * value for weight, value in zip(theta, x))


def calculate_d_b(Y, Y_hat):
    n = len(Y)
    return (-2.0 / n) * sum(y - yh for y, yh in zip(Y, Y_hat))


def calculate_d_theta(X, Y, Y_hat):
    n = len(Y)
    d = len(X[0])
    return [
        (-2.0 / n) * sum(
            X[i][j] * (Y[i] - Y_hat[i]) for i in range(n)
        )
        for j in range(d)
    ]


def solution(x_train, y_train, x_test, iterations=2000,
             learning_rate=0.01):
    if not x_train:
        raise ValueError("Training set cannot be empty")

    b, theta = random_bias_theta(len(x_train[0]))

    for _ in range(iterations):
        predicted = [calculate_y(b, theta, row) for row in x_train]
        grad_b = calculate_d_b(y_train, predicted)
        grad_theta = calculate_d_theta(x_train, y_train, predicted)

        # Simultaneous parameter update from the same old predictions.
        b -= learning_rate * grad_b
        theta = [
            theta[j] - learning_rate * grad_theta[j]
            for j in range(len(theta))
        ]

    return [calculate_y(b, theta, row) for row in x_test]


if __name__ == "__main__":
    random.seed(7)  # For demonstration only; honor the grader's supplied seed.
    print(solution([[0.0], [1.0], [2.0], [3.0]],
                   [1.0, 3.0, 5.0, 7.0], [[4.0]],
                   iterations=5000, learning_rate=0.05))
    # Approximately [9.0]
```

**Caution:** The screenshots do not show the full original caller's learning rate or iteration parameters. For exact CodeSignal results use the pre-existing values and preserve its random initialization / seed. Large feature magnitudes can require scaling or a smaller learning rate.

### implement Gaussian Naive Bayes.

**Question.** Complete a `NaiveBayesClassifier` with four missing methods: (1) per-class feature means and variances, (2) class priors, (3) Gaussian density, and (4) class prediction. The screenshot permits NumPy and `numpy.typing`, and its existing `fit` and `predict` call those methods.

**Answer.** Assume features are conditionally independent given class. Work in log space to prevent underflow. The standalone implementation below reproduces the method names in the visible scaffold.

```python
import numpy as np
import numpy.typing as npt


class NaiveBayesClassifier:
    def get_descriptives(
        self,
        x_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self.classes = np.unique(y_train)
        self.mean = np.array(
            [x_train[y_train == label].mean(axis=0) for label in self.classes]
        )
        self.variance = np.array(
            [x_train[y_train == label].var(axis=0) for label in self.classes]
        )
        self.variance = np.maximum(self.variance, 1e-9)
        return self.mean, self.variance

    def get_priors(
        self, y_train: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        self.prior = np.array(
            [(y_train == label).mean() for label in self.classes],
            dtype=float,
        )
        return self.prior

    def gaussian_density(
        self, class_idx: np.int64, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        mean = self.mean[class_idx]
        variance = self.variance[class_idx]
        return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(
            2.0 * np.pi * variance
        )

    def get_prediction(self, x: npt.NDArray[np.float64]) -> int:
        scores = []
        for class_idx in range(len(self.classes)):
            # Equivalent to log(prior) + sum(log(Gaussian density)).
            variance = self.variance[class_idx]
            mean = self.mean[class_idx]
            log_probability = (
                np.log(self.prior[class_idx])
                - 0.5 * np.sum(np.log(2.0 * np.pi * variance))
                - 0.5 * np.sum((x - mean) ** 2 / variance)
            )
            scores.append(log_probability)
        return int(self.classes[np.argmax(scores)])

    def fit(
        self,
        x_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.int64],
    ) -> None:
        self.get_descriptives(x_train, y_train)
        self.get_priors(y_train)

    def predict(self, x_test: npt.NDArray[np.float64]) -> list[int]:
        return [self.get_prediction(x) for x in x_test]


if __name__ == "__main__":
    X = np.array([[0., 0.], [0.2, 0.1], [4., 4.], [4.2, 3.9]])
    y = np.array([0, 0, 1, 1])
    model = NaiveBayesClassifier()
    model.fit(X, y)
    print(model.predict(np.array([[0.1, 0.1], [4.1, 4.1]])))
    # [0, 1]
```

**Gaussian density:** `p(x_j | c) = (1 / sqrt(2π σ²_cj)) exp(−(x_j−μ_cj)²/(2σ²_cj))`.

**MAP rule:** `argmax_c [log P(c) + Σ_j log p(x_j|c)]`. The small variance floor prevents division by zero. Because the screenshot shows class indices but not the grader's full test expectations, check whether labels are guaranteed to be contiguous integers; the code above also supports noncontiguous class labels.

## Python Basics for Interviews

- NumPy

- Pandas

- Data Cleaning

## Feature Preprocessing

## Train-Test Split Implementation

## Metric Implementation

## Loss Function Implementation

## Gradient Descent Implementation

## Linear Regression from Scratch

## Logistic Regression from Scratch

## K-Means from Scratch

## Mini-Batch Training

## Simple Neural Network Components

## Model Debugging

## Unit Tests and Sanity Checks

## Practice Task Tracker
