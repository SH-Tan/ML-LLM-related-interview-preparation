# Machine Learning Foundations

## Supervised Learning

## Unsupervised Learning

## Reinforcement Learning

## Data Splitting

## Training, Validation, and Test Sets

## Cross-Validation

## Bias and Variance

## Overfitting and Underfitting

### Why can training accuracy be high while test accuracy is low?

**Answer.** There is a generalization gap, often due to overfitting: the model learns training-specific noise or patterns. Other possible causes include train/test distribution shift, data leakage, inconsistent preprocessing, or validation mistakes. Check learning curves and evaluation protocol; consider regularization, early stopping, more representative data, augmentation, or simpler models.

## Regularization

### Which regularization methods push coefficients to zero, and why might many neural-network weights be zero?

**Answer.** Possibilities include explicit pruning, sparsity-inducing regularization (especially L1 with a proximal optimizer), quantization rounding, parameter masks, or numerical underflow. **Do not confuse zero ReLU activations with zero weights.** Ordinary L2 weight decay tends to shrink weights but does not normally make them exactly zero. A dense network trained with ordinary SGD should not be expected to acquire many exact zero weights by itself.

For a linear model with coefficient vector `w`:

- **L1 / Lasso:** `L_total = L_data + λ Σ_j |w_j|`. Its nonsmooth kink at zero encourages sparse solutions; coordinate descent or proximal soft-thresholding can set coefficients **exactly to zero**.
- **L2 / Ridge:** `L_total = L_data + λ Σ_j w_j²`. Smoothly shrinks coefficients, but does **not generally produce exact zeros**.
- **Elastic Net:** Combines L1 and L2; the **L1 component** can produce zeros.
- **L0-style penalties / explicit pruning:** Directly penalize or remove nonzero coefficients, creating sparsity.

**For a multiple-choice item asking which regularizer drives coefficients to zero, the standard answer is L1 (Lasso), and Elastic Net if listed.** Do not select ordinary L2 on that basis.

## Feature Engineering

## Feature Scaling

## Missing Values

## Class Imbalance

## Linear Regression

## Logistic Regression

### Which losses are suitable for binary classification with logistic regression?

**Question.** A dog/not-dog classifier uses logistic regression trained by gradient descent. Select appropriate losses among mean absolute error, hinge loss, mean squared error, cross-entropy, KL divergence, none.

**Answer (interpretation matters):** **Cross-entropy** is the canonical answer for *ordinary binary logistic regression*. **Hinge** is a valid margin-based classification loss, typically paired with an SVM-style score; **KL divergence** is a valid distributional classification objective and equals cross-entropy minus the fixed entropy of the target. Therefore, if the question broadly asks *which losses can be used for classification*, hinge, cross-entropy, and KL are defensible. If it asks for the *standard logistic-regression objective*, select cross-entropy. The appropriate choice depends on the objective specified in the question.

For `y ∈ {0,1}` and predicted probability `p`, binary cross-entropy is

`L(y,p) = −y log(p) − (1−y) log(1−p)`.

Hinge loss for `y ∈ {−1,+1}` and an unbounded decision score `z` is `max(0, 1−yz)`. KL divergence `D_KL(P||Q) = H(P,Q) − H(P)` is equivalent to cross-entropy for a fixed target distribution. MAE/MSE are usually regression losses and are not the preferred logistic classification objective.

## Decision Trees

## Ensemble Learning

### Which statements about ensembles are correct?

**Question.** A loan-default model performs only marginally better than chance. Which statements about using ensemble learning are appropriate?

1. Modern ensembles improve overall model interpretability.
2. Ensembles typically create overfitted models.
3. Ensembles can combine linear and nonlinear relationships.
4. Ensembles typically perform worse when a dataset has both linear and nonlinear relationships.

**Answer: Option 3.** Ensemble methods can combine complementary models. Bagging often reduces variance; boosting can reduce bias and improve fit; stacking can combine model types. The other statements are not reliable general properties. In particular, ensembles are often *harder* to interpret than a single simple model.

**Ensemble tradeoffs:**

| Aspect | Single simple model | Bagging / Random Forest | Boosting |
|---|---|---|---|
| Interpretability | Often easier to explain | Usually more complex; post-hoc feature importance can help | Usually more complex |
| Training cost | One model fit | Many model fits; can often run in parallel | Many stages, typically sequential |
| Memory / inference | Usually lower | Often higher because many models are stored/evaluated | Often higher, depending on ensemble size |
| Generalization | Sensitive to chosen model | Often lowers variance, especially for unstable trees | Can reduce bias and improve accuracy, but needs tuning |

Do not assert that an ensemble *always* prevents overfitting or *always* improves interpretability. More estimators can raise runtime and memory costs, while the effect on predictive performance depends on the data and base learners.

## Random Forests

### Why does bagging help prevent overfitting?

**Question.** Explain why bagging, especially with decision trees, can improve generalization.

**Answer.** Bagging primarily **reduces variance**. A single deep decision tree can change substantially when its training sample changes; it is a high-variance estimator. Bootstrap resampling makes multiple trees see slightly different training sets. Averaging their predictions (or voting for classification) smooths out individual trees' unstable errors.

For regression trees with variance `σ²` and pairwise error correlation `ρ`, the average of `B` similarly distributed trees has variance approximately

`Var(average) = σ² [ρ + (1−ρ)/B]`.

As `B` increases, the uncorrelated part of variance falls, while highly correlated errors remain. Random Forests further decorrelate trees by selecting random candidate feature subsets. **Bagging does not guarantee that bias decreases, nor does it make every ensemble immune to overfitting.**

Practical tradeoff: training `B` trees costs more compute and memory than training one, although the fits are usually parallelizable. Out-of-bag samples can estimate generalization without allocating a separate validation subset.

## Gradient Boosting

## Support Vector Machines

## Naive Bayes

## K-Means

## PCA

## Evaluation Metrics for Classification

### Define TP, FP, TN, FN and common classification metrics.

**Answer.** For positive class 1:

| Quantity | Actual | Predicted | Interpretation |
|---|---|---|---|
| TP | 1 | 1 | Correct positive |
| FP | 0 | 1 | False alarm |
| TN | 0 | 0 | Correct negative |
| FN | 1 | 0 | Missed positive |

| Metric | Formula |
|---|---|
| Accuracy | `(TP + TN) / (TP + TN + FP + FN)` |
| Precision | `TP / (TP + FP)` |
| Recall / Sensitivity / TPR | `TP / (TP + FN)` |
| Specificity / TNR | `TN / (TN + FP)` |
| False-positive rate | `FP / (FP + TN) = 1 − specificity` |
| False-negative rate | `FN / (FN + TP) = 1 − recall` |
| F1 score | `2TP / (2TP + FP + FN)` |
| Balanced accuracy | `(TPR + TNR) / 2` |

Accuracy can be misleading under heavy class imbalance. For example, predicting the majority class for every sample can yield high accuracy yet recall zero for the minority class.

### Which confusion matrices satisfy the specified thresholds?

**Question.** For binary classifiers, select every confusion matrix with recall **strictly greater than 90%** and false-positive rate **strictly less than 10%**. Rows are true labels 0/1; columns are predicted labels 0/1.

**Answer: Option 2 only.**

Recall = `TP / (TP + FN)`; false-positive rate = `FP / (FP + TN)`.

| Option | TN | FP | FN | TP | Recall | FPR | Pass? |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 71 | 12 | 8 | 94 | 92.16% | 14.46% | No: FPR too high |
| **2** | **72** | **7** | **8** | **98** | **92.45%** | **8.86%** | **Yes** |
| 3 | 82 | 7 | 19 | 77 | 80.21% | 7.87% | No: recall too low |
| 4 | 84 | 10 | 8 | 83 | 91.21% | 10.64% | No: FPR too high |

## Evaluation Metrics for Regression

## Model Calibration

## Data Leakage

## Experiment Design
