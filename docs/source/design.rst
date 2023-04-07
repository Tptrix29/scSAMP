.. toctree::
    :maxdepth: 2
    :caption: Contents

Design
=========
Method design document.

Pipeline
------------
1. Preprocessing
2. Sampling
3. Model Training
4. Evaluation

Sampling Strategy
-------------------
7 strategies in toolkit.


1. Classical stratified sampling (Statistical)
2. Balanced stratified sampling (Statistical)
3. Stratified sampling with compactness factor (Bio)
4. Stratified sampling with complexity factor (Bio)
5. Stratified sampling with concave factor (Integrated)
6. Stratified sampling with convex factor (Integrated)
7. Stratified sampling with entropy-weighted factor (Integrated)


Balance Processing
+++++++++++++++++++++++++++++++

:Purpose:
    Avoid the sampling ration which is extremely large or small.

    Assure the stability of method.

There are $M$ clusters in dataset, each cluster's size ratio is $p_l,l=1, ..., M$.
Balance processing aimed at balancing $p_l$, the processing formula is:

$$
p_l^*=\frac{f_b(p_l)}{\sum_{l=1}^{M}{f_b(p_l)}}
$$
$f_b$ is balance function.

Given our purpose, this processing need to scale up the small ratio and scale down the large ratio.
So the balance function should satisfy these constrains:

- monotonically increasing in range (0, 1)
- symmetrical to point $(\frac{1}{2}, \frac{1}{2})$

Function choices：

1. Poly-nominal
$$
f_b(x) = (x-0.5)(ax^2-ax+1)+0.5, a\in(0, 4]
$$
$a$ is scale factor for balance curve, default value is 3.

Curve pattern:
   
   
2. Inverse sigmoid
3. Inverse tanh


Factor Calculation
+++++++++++++++++++++++++++++++

1. Compactness factor

Intra-cluster compactness: for each cluster

$$
Compactness_i = \frac{1}{m_i}\sum^{m_i}_{j=1}||person\_corr(avg_i, value_{ij})||, \in(0, 1)
$$
$$
CompactnessFactor_i = 1 - Compactness_i
$$

2. Complexity factor

Inter-cluster complexity: for each cluster

$$
Complexity_i = \sum^{n}_{j=1}MAX(person\_corr(avg_i, avg_j))
$$
$$
ComplexityFactor_i = Complexity_i
$$


Factor Integration
+++++++++++++++++++++++++++++++

:Purpose:
    Find function $f$ to integrate compactness factor and complexity factor.

1. Concave

2. Convex

3. Entropy weight

The factor with large entropy indicates small information and small weight.

For a scaled matrix $C_{n\times m}$:
Calculate entropy value:
$$
e_j = -\frac{1}{ln(n)}\sum_{i=1}^n c_{ij} * ln(c_{ij})~~~j=1, 2, ..., m
$$

Differential coefficient:
$$
d_j = 1-e_j
$$

Weight：
$$
w_i = \frac{d_j}{\sum d_j}
$$

Oversampling
+++++++++++++++++++++++++++++++

:Purpose:
    To deal with the issue that some cluster's size is not sufficient for expected sampling size, package applied **oversampling** method to generate auxiliary sample.
    The default oversampling method is **SMOTE**.

Sampling Size
-------------------
Users could use customized sampling size or automatically-calculated recommended size to generate sampled dataset.

Recommended Size Calculation
+++++++++++++++++++++++++++++++

Define a state of sampled dataset:
If we apply resampling/random sampling with replacement to shrunk dataset in *k* times, we will consider this dataset is **‘reliable’** enough as long as there are *r* cells from minimum cell type with *c* confidence.

:Conditions:
    *p*, relative abundance, equals to balanced factor

    *r*, success count, customized parameter

    *k*, size of shrunk dataset

    *c*, confidence, customized parameter, default 99%
	
In this scenario, sampling size conform to NB(Negative Binomial) distribution.

**Negative Binomial Distribution**

:Description:
    When applying independent binomial trails (success probability is $p$) and succeed $r$ times, the total trail count is $X$.

$$
X \sim NB(r, p)
$$

:Possibility mass function, pmf:
$$
f(k;r, p) = P(X=k) = \binom {k-1}{r-1}p^r(1-p)^{k-r}~~~k=r, r+1,...
$$

:Cumulative distribution function, cdf:
$$
F(k; r, p) = P(X\leq k) = \sum_{i=r}^{k} P(X=i)
$$

:Expectation & Variance:
$$
E(X) = \frac{r}{p} ~~~~~ Var(X) = \frac{r}{p^2}
$$



Strategy Choice
++++++++++++++++++

Extreme condition:

- Least: 1 cell from minimum cell type, default value
- Most: sampled dataset size = raw dataset size

Check threshold: 
When target dataset size >= 0.5 * total, degenerated to balanced stratified sampling

