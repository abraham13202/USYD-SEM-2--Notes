---
output:
  pdf_document: default
  html_document: default
editor_options: 
  markdown: 
    wrap: 72
---

# STAT5003 FINAL EXAM CHEAT SHEET

## SIDE 1: MODELS, ALGORITHMS & FORMULAS

### PERFORMANCE METRICS

**Total Error Decomposition:** Total Error = Bias² + Variance +
Irreducible Error

**Regression Metrics:** - MSE = (1/n) Σ(yᵢ - ŷᵢ)² - RSS = Σ(yᵢ - ŷᵢ)² -
R² = [Σ(yᵢ - ȳ)² - Σ(yᵢ - ŷᵢ)²] / Σ(yᵢ - ȳ)² - Adjusted R² = 1 -
[(1-R²)(n-1)/(n-p-1)] - Cₚ = (1/n)(RSS + 2dσ²) - BIC = (1/n)(RSS +
log(n)dσ²)

**Classification Metrics:** - Accuracy = (TP + TN) / Total - Precision =
TP / (TP + FP) - Recall (Sensitivity) = TP / (TP + FN) - Specificity =
TN / (TN + FP) - F1 Score = 2 × (Precision × Recall) / (Precision +
Recall) - Cohen's Kappa: κ = (p₀ - pₑ) / (1 - pₑ) - p₀ = observed
agreement, pₑ = expected agreement by chance - AUC-ROC: Area under ROC
curve (TPR vs FPR)

**For Imbalanced Data:** Use Precision, Recall, F1, Cohen's Kappa,
AUC-ROC (NOT accuracy!)

------------------------------------------------------------------------

### REGRESSION METHODS

**1. Linear Regression** - Model: Y = β₀ + β₁X₁ + ... + βₚXₚ + ε -
Minimize RSS (ordinary least squares) - Assumptions: linearity,
independence, homoscedasticity, normality of residuals

**2. Ridge Regression** - min_β [Σ(yᵢ - β₀ - Σβⱼxᵢⱼ)² subject to Σβⱼ² ≤
s] - L2 penalty, shrinks coefficients, keeps all variables - R:
`glmnet(alpha=0)`

**3. Lasso Regression** - min_β [Σ(yᵢ - β₀ - Σβⱼxᵢⱼ)² subject to Σ\|βⱼ\|
≤ s] - L1 penalty, feature selection (sets some β to 0) - R:
`glmnet(alpha=1)`

**4. Regression Trees** - Partition space into rectangular regions -
Minimize RSS in each region - Prone to overfitting

**5. Random Forests (Regression)** - Bagging with feature subsampling at
each split - Reduces variance, harder to interpret - OOB error for
validation

**6. Gradient Boosting Trees** - Fit trees sequentially to residuals -
Key hyperparameters: n_trees, learning rate (shrinkage), max_depth -
XGBoost: advanced implementation

------------------------------------------------------------------------

### CLASSIFICATION METHODS

**1. Logistic Regression** - Model: log(p/(1-p)) = Xβ - Predicted
probability: p = P(Y=1\|x) = 1/(1 + exp(-Xβ)) - Binary classification
using sigmoid function

**2. Linear Discriminant Analysis (LDA)** - Bayes theorem: P(Y=k\|X=x) =
[πₖfₖ(x)] / [Σπₗfₗ(x)] - Assumes Gaussian distributions with same
covariance - Finds linear decision boundaries

**3. k-Nearest Neighbors (kNN)** - Non-parametric: classify based on k
nearest training points - No training phase, flexible decision
boundary - Sensitive to k choice and feature scaling

**4. Support Vector Machines (SVM)** - Find hyperplane maximizing margin
between classes - Support vectors define the boundary - Can use kernels
(linear, RBF, polynomial) for non-linear boundaries - Hyperparameter C:
larger C = smaller margin, lower bias, higher variance

**5. Classification Trees** - Split based on Gini impurity or entropy -
Easy to interpret, prone to overfitting

**6. Random Forests (Classification)** - Bagging + random feature
selection at splits - Reduces variance, provides feature importance -
OOB error estimates test performance

**7. AdaBoost** - Sequential learning: weight misclassified samples
more - Combines weak learners into strong classifier - Sensitive to
noisy data and outliers

------------------------------------------------------------------------

### MODEL SELECTION & FEATURE SELECTION

**Best Subset Selection:** - Try all 2ᵖ models - Computationally
expensive

**Forward Selection:** - Start with null model - Add one variable at a
time (best improvement)

**Backward Selection:** - Start with full model - Remove one variable at
a time (least important)

**Selection Criteria:** - Cross-validation error (direct) - Adjusted R²,
AIC, BIC, Cₚ (indirect) - Lower BIC/AIC = better

------------------------------------------------------------------------

### CROSS-VALIDATION

**k-Fold CV:** 1. Split data into k folds 2. Train on k-1 folds,
validate on 1 fold 3. Repeat k times, average errors 4. Common: k=5 or
k=10

**Leave-One-Out CV (LOOCV):** - k = n (each observation is validation
set once) - Unbiased but high variance, computationally expensive

**Repeated k-Fold CV:** - Repeat k-fold CV with different random
splits - Reduces bias, provides variance estimate - Computationally
expensive

**CRITICAL: Proper CV Workflow** 1. Split into train/test FIRST 2. All
preprocessing (imputation, scaling, feature selection) on TRAINING data
only 3. Apply same transformations to test data 4. **Data leakage**:
using information from test set in training = overestimating performance

------------------------------------------------------------------------

## SIDE 2: UNSUPERVISED LEARNING, RESAMPLING & R CODE

### PRINCIPAL COMPONENTS ANALYSIS (PCA)

**Purpose:** Dimensionality reduction, find directions of maximum
variance

**Method:** 1. Standardize data (mean=0, sd=1) 2. Compute covariance
matrix 3. Find eigenvectors (principal components) and eigenvalues 4.
PC1 = direction of maximum variance 5. PC2 = direction of maximum
remaining variance (orthogonal to PC1)

**Properties:** - PCs are uncorrelated - Total variance preserved =
Σλᵢ - Proportion of variance explained by PCⱼ = λⱼ / Σλᵢ

**R Code:**

``` r
pca_result <- prcomp(data, scale=TRUE)
summary(pca_result)  # variance explained
biplot(pca_result)
```

------------------------------------------------------------------------

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Purpose:** Non-linear dimensionality reduction for visualization

**Key Differences from PCA:** - Non-linear (captures complex
structures) - Preserves local structure better than global - Stochastic
(different runs give different results) - Mainly for visualization
(2D/3D) - Cannot transform new data easily - Sensitive to perplexity
parameter

**When to use:** Visualizing high-dimensional data with complex
structure (e.g., image data, clusters)

------------------------------------------------------------------------

### CLUSTERING METHODS

**1. K-Means Clustering**

**Algorithm:** 1. Initialize: randomly assign observations to K clusters
2. Iterate until convergence: - Step 1: Compute cluster means
(centroids): x̄ⱼ = argmin_m Σ \|\|xᵢ - m\|\|² - Step 2: Assign
observations to nearest centroid: cluster(i) = argmin_k \|\|xᵢ - x̄ₖ\|\|²

**Properties:** - Need to specify K - Sensitive to initialization (run
multiple times) - Assumes spherical clusters - Fast and scalable

**R Code:**

``` r
kmeans_result <- kmeans(data, centers=K, nstart=25)
```

**2. Hierarchical Clustering**

**Algorithm:** 1. Start: each observation is its own cluster 2.
Repeatedly merge two closest clusters 3. Create dendrogram 4. Cut
dendrogram at desired height to get K clusters

**Linkage Methods:** - Complete: max distance between clusters - Single:
min distance between clusters - Average: average distance between
clusters - Centroid: distance between centroids

**R Code:**

``` r
hc <- hclust(dist(data), method="complete")
plot(hc)
clusters <- cutree(hc, k=K)
```

------------------------------------------------------------------------

### BOOTSTRAP

**Purpose:** Estimate sampling distribution and uncertainty

**Method:** 1. Resample n observations WITH replacement from original
data 2. Calculate statistic of interest on bootstrap sample 3. Repeat B
times (typically B = 1000-10000) 4. Use bootstrap distribution to
estimate SE, confidence intervals

**Formula:** θ̂\_boot = (1/B) Σθ̂ᵇ

**Properties:** - Non-parametric (no distributional assumptions) - Works
for complex statistics - Approximates sampling distribution

------------------------------------------------------------------------

### BAGGING (Bootstrap Aggregating)

**Method:** 1. Create B bootstrap samples 2. Train model on each
bootstrap sample 3. Average predictions: f̂\_bag(x) = (1/B) Σf̂ᵇ\*(x)

**Benefits:** - Reduces variance (especially for high-variance models
like trees) - Doesn't increase bias

**Out-of-Bag (OOB) Error:** - Each bootstrap sample excludes \~37% of
observations - Use excluded observations as validation set - Average OOB
predictions across all trees where observation was OOB - OOB error ≈
cross-validation error (no need for separate test set)

------------------------------------------------------------------------

### BOOSTING

**Key Hyperparameters:** 1. **Number of trees (B):** More trees = lower
bias, risk overfitting 2. **Learning rate (λ or shrinkage):** Smaller =
slower learning, need more trees, better generalization 3. **Max depth
(d):** Tree complexity, typically d=1 (stumps) to d=6

**Bias-Variance Trade-off:** - More trees, higher depth → lower bias,
higher variance - Lower learning rate, fewer trees → higher bias, lower
variance

**AdaBoost:** - Weight misclassified observations more heavily -
Sequentially build models focusing on "hard" cases

**Gradient Boosting:** - Fit trees to residuals sequentially - Update:
f̂ₘ(x) = f̂ₘ₋₁(x) + λ·tree_m(x)

------------------------------------------------------------------------

### DENSITY ESTIMATION

**1. Maximum Likelihood Estimation (MLE)** - Likelihood: L(θ\|x) =
Πf(xᵢ\|θ) - Log-likelihood: ℓ(θ\|x) = Σlog f(xᵢ\|θ) - MLE: θ̂ = argmax_θ
ℓ(θ\|x)

**2. Kernel Density Estimation (KDE)**

**Formula:** f̂(x) = (1/nh) Σ K((x - Xᵢ)/h)

**Kernel Properties:** - Non-negative: K(x) ≥ 0 - Symmetric: K(-x) =
K(x) - Unit measure: ∫K(x)dx = 1

**Bandwidth (h):** - Larger h = smoother, higher bias, lower variance -
Smaller h = rougher, lower bias, higher variance

------------------------------------------------------------------------

### MONTE CARLO METHODS

**Purpose:** Use simulation to estimate complex
distributions/expectations

**Basic Method:** 1. Generate N random samples from distribution 2.
Compute statistic on each sample 3. Law of Large Numbers: E[g(X)] ≈
(1/N) Σg(Xᵢ)

**Sampling Methods:** 1. **Inverse Transform:** If X\~f, generate
U\~Uniform(0,1), set X = F⁻¹(U) 2. **Acceptance-Rejection:** Sample from
proposal, accept/reject to get target distribution

**R Code Structure:**

``` r
n_sim <- 10000
results <- numeric(n_sim)
for(i in 1:n_sim) {
  # Generate random values
  sample_value <- rnorm(n, mean, sd)  # or other distribution
  # Compute statistic
  results[i] <- some_function(sample_value)
}
# Estimate probability/expectation
estimate <- mean(results > threshold)
```

------------------------------------------------------------------------

### MARKOV CHAIN MONTE CARLO (MCMC)

**Purpose:** Sample from complex distributions (especially in Bayesian
inference)

**Key Idea:** - Simulate Markov chain where stationary distribution =
target distribution - New point depends only on current point (Markov
property) - After burn-in period, samples approximate target
distribution

**Metropolis-Hastings Algorithm:** 1. Start at x₀ 2. Propose x\* from
proposal distribution q(x*\|xₜ) 3. Accept x* with probability α = min(1,
[π(x*)q(xₜ\|x*)] / [π(xₜ)q(x\*\|xₜ)]) 4. If rejected, stay at xₜ

------------------------------------------------------------------------

### LOCAL REGRESSION (SMOOTHING)

**Model:** Yᵢ = f(xᵢ) + εᵢ

**Loess (Locally Weighted Regression):** - Fit polynomial regression in
local neighborhood - Weight nearby points more heavily - Bandwidth
parameter controls smoothness

------------------------------------------------------------------------

### MISSING DATA STRATEGIES

**1. Complete Case Analysis** - Remove all observations with missing
values - Problems: loses information, may introduce bias

**2. Single Imputation** - Mean/median imputation (for numerical) - Mode
imputation (for categorical) - Problems: underestimates variance,
ignores uncertainty

**3. Multiple Imputation** - Create M imputed datasets - Analyze each
separately - Pool results accounting for uncertainty

**4. Model-Based Imputation** - Use predictive models (regression, kNN,
RF) - Can capture relationships between variables

**CRITICAL:** Impute ONLY using training data, then apply to test data

------------------------------------------------------------------------

### KEY R FUNCTIONS

**Model Fitting:** - Linear: `lm(y ~ x1 + x2, data)` - Logistic:
`glm(y ~ x, family=binomial, data)` - Ridge/Lasso:
`glmnet(X, y, alpha=0/1, lambda)` - Trees: `rpart(y ~ ., data)` or
`tree(y ~ ., data)` - Random Forest:
`randomForest(y ~ ., data, ntree, mtry)` - SVM:
`svm(y ~ ., data, kernel="linear"/"radial")` - kNN:
`knn(train, test, cl, k)` - LDA: `lda(y ~ ., data)`

**Cross-Validation:** - `caret::trainControl(method="cv", number=10)` -
`caret::train(y ~ ., data, method="rf", trControl)`

**Clustering:** - K-means: `kmeans(data, centers=k, nstart=25)` -
Hierarchical: `hclust(dist(data), method="complete")`

**Dimensionality Reduction:** - PCA: `prcomp(data, scale=TRUE)` - t-SNE:
`Rtsne::Rtsne(data, perplexity=30)`

**Model Evaluation:** - Confusion matrix:
`caret::confusionMatrix(predicted, actual)` - ROC curve:
`pROC::roc(actual, predicted_prob)`

------------------------------------------------------------------------

### COMMON PITFALLS & WORKFLOWS

**Data Leakage Scenarios:** 1. ❌ Imputing missing values using full
dataset (including test) 2. ❌ Feature selection on full dataset before
splitting 3. ❌ Scaling/normalizing using full dataset statistics 4. ❌
Using test data for any model tuning decisions

**Proper Workflow:**

```         
1. Split data → Train | Test
2. Exploratory analysis on TRAIN only
3. Preprocessing (impute, scale) on TRAIN
   - Save parameters (mean, sd, imputation values)
4. Apply same preprocessing to TEST using TRAIN parameters
5. Feature selection on TRAIN only
6. Model training with CV on TRAIN
7. Final evaluation on TEST (once!)
```

**Imbalanced Classes:** - Don't use accuracy! - Use: Precision, Recall,
F1, Kappa, AUC-ROC - Consider: SMOTE, class weights, stratified sampling

**Model Comparison:** - Use same CV splits for fair comparison -
Consider computational cost vs performance gain - Interpretability vs
accuracy trade-off

------------------------------------------------------------------------

### QUICK DECISION GUIDE

**Regression vs Classification?** - Continuous outcome → Regression -
Categorical outcome → Classification

**Linear vs Non-linear?** - Linear relationship → Linear models, LDA -
Non-linear → Trees, SVM with kernels, kNN

**Interpretability needed?** - High: Linear regression, logistic, trees,
LDA - Low: Random forests, boosting, SVM, neural nets

**High-dimensional data (p \>\> n)?** - Ridge/Lasso, PCA then model,
elastic net

**Feature selection needed?** - Lasso, stepwise selection, random forest
importance

**Small dataset?** - Avoid complex models (overfitting risk) - Use CV
carefully, consider simpler models

**Large dataset?** - Can use more complex models - Computational
efficiency matters

------------------------------------------------------------------------

### EXAM-SPECIFIC TIPS

**Multiple Choice:** - Need BOTH correct answers - Wrong answer deducts
1 mark - If unsure between 3+ options, leave blank (0) vs risk (-1 or
-2) - Look for: "supervised" vs "unsupervised", "reduces bias" vs
"reduces variance"

**Extended Answer:** - Workflow critique: look for data leakage, wrong
metrics for imbalanced data, improper CV - Algorithm comparison:
bias-variance, interpretability, computational cost, assumptions - Monte
Carlo: n_sim (line 8), distribution (line 18), proper update formulas -
Always explain WHY, not just WHAT

**Common MC Topics:** - Supervised (RF, LDA, SVM, Logistic, Lasso) vs
Unsupervised (K-means, PCA, hierarchical) - Kernel properties:
symmetric, non-negative, integrates to 1 - Data leakage scenarios - SVM
properties: support vectors, margin, hyperplane - Indirect error
measures: Cp, BIC, Adj-R² (NOT F1, accuracy)
