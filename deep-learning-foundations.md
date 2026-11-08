# Deep Learning Foundations

## Neural Network Basics

## Forward Propagation/Backpropagation

### Calculate two neural-network forward passes.

**Question.** Both examples use three inputs, two linear hidden units `f1`, `f2`, and one sigmoid output `f3`; all biases are zero. Calculate output to three decimal places.

**Example 1:** Inputs `(3, 4, 9)`. Input-to-hidden weights are `(0.3, 0.4, 0.2)` for `f1`, `(0.4, 0.2, 0.5)` for `f2`; both hidden-to-output weights are `0.1`.

```
f1 = 3(0.3) + 4(0.4) + 9(0.2) = 4.3
f2 = 3(0.4) + 4(0.2) + 9(0.5) = 6.5
z  = 0.1(4.3) + 0.1(6.5) = 1.08
f3 = sigmoid(1.08) = 1 / (1 + exp(-1.08)) = 0.746493983...
```

**Final answer: 0.746.**

**Example 2:** Inputs `(6, 7, 1)`. Input-to-hidden weights `(0.5, 0.1, 0.1)` for `f1`, `(0.1, 0.5, 0.2)` for `f2`; hidden-to-output weights `0.1` and `0.4`.

```
f1 = 6(0.5) + 7(0.1) + 1(0.1) = 3.8
f2 = 6(0.1) + 7(0.5) + 1(0.2) = 4.3
z  = 0.1(3.8) + 0.4(4.3) = 2.10
f3 = sigmoid(2.10) = 1 / (1 + exp(-2.10)) = 0.890903179...
```

**Final answer: 0.891.**

## Loss Functions

## Activation Functions

## Optimizers

### What does a gradient mean, and how does gradient descent update parameters?

**Question.** Explain gradients, their signs, the chain rule, and common derivatives used in ML.

**Answer.** The gradient `∇θ L` points in the local direction of **steepest increase** in a differentiable loss, so gradient descent takes a step in the opposite direction:

`θ_new = θ_old − η ∇θ L`, where `η > 0` is the learning rate.

For a sigmoid output `p = σ(z) = 1/(1+exp(−z))`:

`dσ/dz = p(1−p)`.

For binary cross-entropy `L = −y log(p) − (1−y)log(1−p)` combined with sigmoid:

`dL/dz = p−y`.

For a linear prediction `ŷ = w·x + b` and **one-example squared error** `L=(y−ŷ)²`:

`dL/dw_j = 2(ŷ−y)x_j`, and `dL/db = 2(ŷ−y)`.

The chain rule multiplies derivatives across nested functions, which is the foundation of backpropagation. A very large learning rate may cause divergence or oscillation; a very small one converges slowly. A zero gradient can mean a stationary point, saturation, or a disconnected/masked path—it does **not** automatically prove a global optimum.

## Initialization

## Batch Size and Epochs

## Normalization Layers

## Embeddings

## CNN Basics

## Attention Mechanism

## Transformer Basics

## Encoder and Decoder Architectures

## Pretraining and Fine-Tuning

## Training Stability

## Distributed Training Basics

## GPU and Memory Basics

## Common Interview Questions
