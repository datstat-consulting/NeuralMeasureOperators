# NeuralMeasureOperators

Finite computational representations of operators acting on functions, fields, and empirical measures.

NeuralMeasureOperators is a lightweight Python package for experimenting with the operator-theoretic view of neural computation. It treats arrays as finite representations of structured objects: coordinate fields, image fields, empirical measures, and kernel-induced feature measures.

The package provides scikit-learn-compatible components for:

- raw finite-field classifiers
- PCA field projections
- DCT low-frequency image-field projections
- empirical RBF class-kernel-mean classifiers
- Nyström low-rank RBF kernel approximations
- exact RBF kernel SVM baselines

The guiding principle is:

$$
\text{operator}
\;\longrightarrow\;
\text{represented function or measure}
\;\longrightarrow\;
\text{finite computable approximation}.
$$

NeuralMeasureOperators follows the decomposition

$$
T
\quad\leadsto\quad
T_h
\quad\leadsto\quad
\text{finite computation}.
$$

The continuous object is an operator on functions or measures. The implemented model is a finite representation of that operator, obtained through sampling, projection, truncation, empirical embedding, or kernel approximation.

A vector, image, graph signal, point cloud, or token sequence can be treated as a finite representation of a structured function or measure.

## Mathematical view

Let $(X,\mathcal{A},\mu)$ be a measure space and let

$$
u:X\to E
$$

be a feature field. A broad class of neural and kernel-based transformations can be written as

$$
(Tu)(y)
=
\Gamma\left(
y,
u(y),
\int_X K(y,x,u)\Psi(u(x))\,d\mu(x)
\right).
$$

Here $K$ is an interaction kernel, $\Psi$ transforms features before aggregation, $\Gamma$ updates the state at the output site, and $\mu$ determines how information is integrated over the domain.

A finite representation replaces the function or measure by samples, coefficients, basis projections, quadrature nodes, graph vertices, Fourier modes, or empirical measures.

For sampled sites $x_1,\dots,x_n$ with weights $w_i$,

$$
\mu_n=\sum_{i=1}^n w_i\delta_{x_i}.
$$

The corresponding finite approximation is

$$
(T_nu)(y)
=
\Gamma\left(
y,
u(y),
\sum_{i=1}^n
w_iK(y,x_i,u)\Psi(u(x_i))
\right).
$$

NeuralMeasureOperators contains small finite approximators used in this sense.

## Included components

### Raw finite field

A vector

$$
x\in\mathbb{R}^d
$$

is treated as a scalar field on the finite coordinate set

$$
X_d=\{1,\dots,d\}.
$$

A classifier can act directly on this finite field.

### PCA projection

PCA gives an empirical spectral projection

$$
P_mx
=
\sum_{j=1}^m
\langle x,v_j\rangle v_j.
$$

This is a finite-dimensional approximation of a sampled coordinate field or image field.

### DCT low-frequency projection

For image fields

$$
u:\{1,\dots,H\}\times\{1,\dots,W\}\to\mathbb{R},
$$

the DCT representation keeps low-frequency harmonic coefficients

$$
\widehat{u}_{ij}
=
\sum_{p,q}u_{pq}\phi_{ij}(p,q).
$$

The package includes `DCT2DLowFreq` for flattened image arrays.

### Empirical class-kernel mean

Each class $c$ is represented by an empirical measure

$$
\mu_c
=
\frac{1}{n_c}
\sum_{i:y_i=c}\delta_{x_i}.
$$

Using an RBF kernel

$$
k(x,z)
=
\exp\left(-\gamma\|x-z\|^2\right),
$$

the class score is

$$
s_c(x)
=
\frac{1}{n_c}
\sum_{i:y_i=c}
k(x,x_i).
$$

Prediction is

$$
\widehat{y}(x)
=
\operatorname*{arg\,max}_c s_c(x).
$$

This is a direct empirical-measure classifier: each class is represented by its kernel mean embedding.

### Nyström RBF approximation

The Nyström method approximates the empirical kernel operator by a finite-rank feature map

$$
k(x,z)\approx \phi_R(x)^\top\phi_R(z).
$$

This gives a lower-rank computable approximation of an RBF kernel machine.

## Installation

Install locally from the repository root:

```bash
pip install -e .
```

The install/distribution name is:

```text
neural-measure-operators
```

The Python import package is:

```python
import NeuralMeasureOperators
```

## Basic imports

```python
from NeuralMeasureOperators import (
    DCT2DLowFreq,
    RBFClassMeanClassifier,
    raw_logistic,
    pca_logistic,
    dct_logistic,
    kernel_mean,
    nystroem_ridge,
    exact_rbf_svm,
)
```

## Example: Breast Cancer coordinate field

The Breast Cancer dataset has 30 features. In this view, each sample is a scalar field over the finite coordinate domain

$$
X=\{1,\dots,30\}.
$$

The examples below compare several finite approximations of the same represented object.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from NeuralMeasureOperators import (
    raw_logistic,
    pca_logistic,
    kernel_mean,
    nystroem_ridge,
    exact_rbf_svm,
)

data = load_breast_cancer()

X = data.data.astype(float)
y = data.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=11,
)

models = {
    "raw finite field": raw_logistic(),
    "PCA 8 modes": pca_logistic(n_components=8),
    "kernel mean": kernel_mean(gamma="scale"),
    "Nystroem RBF rank 64": nystroem_ridge(n_components=64, gamma=0.10),
    "exact RBF SVM": exact_rbf_svm(C=10.0, gamma="scale"),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print(f"{name:24s} accuracy={acc:.4f} macro_f1={f1:.4f}")
```

Interpretation:

$$
x\in\mathbb{R}^{30}
$$

is a finite coordinate field.

- Raw logistic regression acts on the sampled coordinate field directly.
- PCA compresses the coordinate field into empirical spectral modes.
- The kernel mean classifier embeds class empirical measures in an RKHS.
- Nyström approximates the RBF kernel operator with finite-rank features.
- The exact RBF SVM evaluates the full empirical kernel over the training measure.

## Example: Digits image field

The Digits dataset has $8\times8$ grayscale images. Each sample is a scalar field

$$
u:\{1,\dots,8\}\times\{1,\dots,8\}\to\mathbb{R}.
$$

The flattened array has 64 entries, while `DCT2DLowFreq` treats each row as an $8\times8$ image field.

```python
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from NeuralMeasureOperators import (
    raw_logistic,
    pca_logistic,
    dct_logistic,
    kernel_mean,
    nystroem_ridge,
    exact_rbf_svm,
)

digits = load_digits()

X = digits.data.astype(float) / 16.0
y = digits.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=11,
)

models = {
    "raw 64-pixel field": raw_logistic(),
    "PCA 32 modes": pca_logistic(n_components=32),
    "DCT 32 modes": dct_logistic(n_components=32, image_shape=(8, 8)),
    "kernel mean": kernel_mean(gamma="scale"),
    "Nystroem RBF rank 128": nystroem_ridge(n_components=128, gamma=0.02),
    "exact RBF SVM": exact_rbf_svm(C=10.0, gamma="scale"),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print(f"{name:24s} accuracy={acc:.4f} macro_f1={f1:.4f}")
```

Interpretation:

$$
u\in\mathbb{R}^{8\times8}
$$

is a sampled image field.

- Raw logistic regression acts on all 64 grid values.
- PCA learns empirical image-field modes.
- DCT keeps fixed low-frequency harmonic modes.
- The kernel mean classifier embeds class empirical image measures.
- Nyström gives a low-rank approximation of the RBF kernel operator.
- The exact RBF SVM uses the full empirical kernel operator over the training measure.

## Direct use: DCT image-field representation

```python
from sklearn.datasets import load_digits

from NeuralMeasureOperators import DCT2DLowFreq

digits = load_digits()
X = digits.data.astype(float) / 16.0

dct = DCT2DLowFreq(image_shape=(8, 8), n_components=16)

Z = dct.fit_transform(X)
X_reconstructed = dct.inverse_transform(Z)

print("original shape:", X.shape)
print("DCT feature shape:", Z.shape)
print("reconstructed shape:", X_reconstructed.shape)
```

Mathematically, this computes

$$
u
\mapsto
(\widehat{u}_{i_1j_1},\dots,\widehat{u}_{i_mj_m}),
$$

where the retained indices are ordered from low to high frequency.

## Direct use: empirical kernel-mean classifier

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from NeuralMeasureOperators import RBFClassMeanClassifier

data = load_breast_cancer()

X = data.data.astype(float)
y = data.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=11,
)

clf = RBFClassMeanClassifier(gamma="scale")
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
scores = clf.decision_function(X_test)

print("accuracy:", accuracy_score(y_test, pred))
print("score matrix shape:", scores.shape)
```

The score matrix has shape

$$
n_{\text{test}}\times n_{\text{classes}}.
$$

Each entry is an empirical kernel integral against a class measure:

$$
s_c(x)
=
\int k(x,z)\,d\mu_c(z).
$$

## Package API

### Representations

```python
DCT2DLowFreq(image_shape=(8, 8), n_components=16)
RBFClassMeanClassifier(gamma="scale")
```

### Pipelines

```python
raw_logistic()
pca_logistic(n_components=8)
dct_logistic(n_components=32, image_shape=(8, 8))
kernel_mean(gamma="scale")
nystroem_ridge(n_components=128, gamma=0.02)
exact_rbf_svm(C=10.0, gamma="scale")
```

## License

MIT