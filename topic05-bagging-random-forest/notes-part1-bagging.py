import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Topic 5 — Part 1: Bagging

    An **ensemble** is a group of models whose combined answer beats any single member. Part 1 covers the first and simplest ensemble idea: resample the training set (bootstrap), fit one model per resample, average the predictions (bagging). Part 2 builds random forests on top of this.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why averaging models works at all

    Two old results give the intuition:

    **Condorcet's jury theorem (1785).** With $N$ jurors voting independently, each correct with probability $p$, the probability $\mu$ that the majority is correct is

    $$\mu = \sum_{i=m}^{N} \binom{N}{i} p^i (1-p)^{N-i}, \qquad m = \lfloor N/2 \rfloor + 1$$

    If $p > 0.5$ then $\mu > p$, and $\mu \to 1$ as $N \to \infty$. The catch is the direction flips: if $p < 0.5$, more jurors make the verdict *worse*, tending to 0. So an ensemble amplifies whatever the base learner is — it needs to be better than a coin flip, and the votes need to be independent.

    **Wisdom of the crowd (Galton, 1906).** 800 fairgoers guessed the weight of a bull; the true weight was 1198 lb and the *average* guess was 1197 lb. No individual was right. The errors were roughly independent and cancelled.

    Both conditions — better than chance, and errors that are independent — are exactly what ensembles in ML try to engineer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bootstrapping

    **Bootstrapping** is the statistical trick underneath bagging. Given a sample $X$ of size $N$, build a new sample by drawing $N$ elements from $X$ **uniformly at random with replacement**. Each draw picks any element with probability $1/N$, and because you put the ball back, a bootstrap sample contains duplicates and misses some originals. Repeat $M$ times to get $X_1, \ldots, X_M$.

    Why bother: it lets you estimate the sampling distribution of *any* statistic (mean, median, a model's score) without assuming a parametric form, using nothing but the data you already have. Small dataset, and you want a confidence interval? Bootstrap it.

    We'll use the telecom churn dataset and the `Customer service calls` feature — the one that showed up as a strong churn signal back in Topic 1.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt

    sns.set()
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'retina'

    DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    telecom_data = pd.read_csv(DATA_PATH + "telecom_churn.csv")

    telecom_data.loc[telecom_data["Churn"] == False, "Customer service calls"].hist(label="Loyal")
    telecom_data.loc[telecom_data["Churn"] == True, "Customer service calls"].hist(label="Churn")
    plt.xlabel("Number of calls")
    plt.ylabel("Number of customers")
    plt.legend();
    return np, telecom_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Loyal customers call support less than customers who eventually leave. The group means are the obvious next question, but with a small dataset a single point estimate is unconvincing — so bootstrap 1000 resamples per group, take the mean of each, and read off the 2.5th and 97.5th percentiles of those 1000 means. That's a 95% **confidence interval** built from nothing but resampling.
    """)
    return


@app.cell
def _(np, telecom_data):
    def get_bootstrap_samples(data, n_samples):
        """n_samples bootstrap resamples of `data`, each the same length as `data`."""
        indices = np.random.randint(0, len(data), (n_samples, len(data)))
        return data[indices]


    def stat_intervals(stat, alpha):
        """Percentile interval estimate at level 1 - alpha."""
        return np.percentile(stat, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])


    loyal_calls = telecom_data.loc[telecom_data["Churn"] == False, "Customer service calls"].values
    churn_calls = telecom_data.loc[telecom_data["Churn"] == True, "Customer service calls"].values

    np.random.seed(0)
    loyal_mean_scores = [np.mean(s) for s in get_bootstrap_samples(loyal_calls, 1000)]
    churn_mean_scores = [np.mean(s) for s in get_bootstrap_samples(churn_calls, 1000)]

    print("Service calls from loyal: mean interval", stat_intervals(loyal_mean_scores, 0.05))
    print("Service calls from churn: mean interval", stat_intervals(churn_mean_scores, 0.05))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two intervals don't overlap — the churn group really does call more. The loyal interval is also narrower, which makes sense: those customers call 0, 1 or 2 times, while frustrated customers keep calling until they quit, so that group's values are spread wider.

    **Trap:** a 95% CI does *not* mean "95% of the values live in here", nor "95% probability the true mean is in here". It's a statement about the procedure: intervals built this way cover the true parameter 95% of the time across repeated sampling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bagging = Bootstrap AGGregatING

    Breiman, 1994. Draw $M$ bootstrap samples from the training set $X$, fit a separate base model $a_i(x)$ on each, then combine:

    $$a(x) = \frac{1}{M} \sum_{i=1}^{M} a_i(x)$$

    For regression that's an average; for classification it's a vote (or averaged probabilities).

    ### Why it works: the $1/n$ variance proof

    Take base regressors $b_1, \ldots, b_n$, a true function $y(x)$, and per-model errors $\varepsilon_i(x) = b_i(x) - y(x)$. The mean error of a single model is

    $$E_1 = \frac{1}{n} \mathbb{E}_x \Big[\sum_{i=1}^{n} \varepsilon_i^2(x)\Big]$$

    Assume the errors are **unbiased and uncorrelated**:

    $$\mathbb{E}_x[\varepsilon_i(x)] = 0, \qquad \mathbb{E}_x[\varepsilon_i(x)\varepsilon_j(x)] = 0 \;\; (i \neq j)$$

    Then for the averaged model $a(x) = \frac{1}{n}\sum_i b_i(x)$:

    $$E_n = \mathbb{E}_x\Big[\big(\frac{1}{n}\sum_{i=1}^n \varepsilon_i\big)^2\Big] = \frac{1}{n^2} \mathbb{E}_x\Big[\sum_{i=1}^n \varepsilon_i^2(x) + \sum_{i \neq j} \varepsilon_i(x)\varepsilon_j(x)\Big] = \frac{1}{n} E_1$$

    The cross terms vanish by the uncorrelated assumption, and the error drops by a factor of $n$.

    **Read this against the bias–variance decomposition** from Topic 4:

    $$\mathrm{Err}(\vec{x}) = \mathrm{Bias}(\hat{f})^2 + \mathrm{Var}(\hat{f}) + \sigma^2$$

    Bagging attacks the **variance** term only. Bias stays put and $\sigma^2$ (irreducible noise) is untouchable. Hence the rule of thumb: **bag models that are low-bias and high-variance** — deep, unpruned decision trees are the canonical choice. Bagging a high-bias model (a shallow stump, a linear regression) just gives you the same biased answer with less jitter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the assumption breaks

    The $1/n$ result is the best case and it rests on errors being uncorrelated — which is far too optimistic in practice. Models trained on bootstrap samples of the *same* data see mostly the same rows and make mostly the same mistakes, so real variance reduction is well short of $1/n$. Killing that correlation is precisely what random forests add in Part 2 (random feature subsets per split).

    Numbers from sklearn's bias–variance example, tree vs. bagged trees on the same problem:

    | | Err | Bias² | Var | σ² |
    |---|---|---|---|---|
    | single tree | 0.0255 | 0.0003 | 0.0152 | 0.0098 |
    | bagging | 0.0196 | 0.0004 | 0.0092 | 0.0098 |

    Bias and noise are flat; variance nearly halves. Exactly as advertised.

    **Practical notes:**
    - Bagging shines on **small datasets** — dropping a few rows produces genuinely different trees.
    - On large datasets the bootstrap samples are near-identical, so the models correlate; sample *smaller* subsets to force diversity.
    - sklearn ships `BaggingClassifier` / `BaggingRegressor` as meta-estimators; almost any base estimator plugs in.
    - Outliers land in only some bootstrap samples, so they get partially voted away for free.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Out-of-bag error

    Each bootstrap sample misses some rows. Those rows are a ready-made validation set for that particular tree — free, no cross-validation, no hold-out split.

    **How much is left out?** With $\ell$ examples, each draw picks a given example with probability $1/\ell$, so it *misses* it with probability $1 - 1/\ell$. Over $\ell$ independent draws:

    $$P(\text{example never drawn}) = \left(1 - \frac{1}{\ell}\right)^{\ell} \xrightarrow[\ell \to \infty]{} \frac{1}{e} \approx 0.37$$

    So each base model trains on roughly **63%** of the data and can be validated on the remaining **37%** — its *out-of-bag* rows.
    """)
    return


@app.cell
def _(np):
    # the 63/37 split is just the second remarkable limit showing up
    for l in [10, 100, 1000, 100000]:
        print(l, (1 - 1 / l) ** l)
    print("1/e =", 1 / np.e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Computing OOB error:**

    1. Collect every instance that was out-of-bag for at least one tree — that's the OOB dataset.
    2. For one such instance, find all trees that never saw it during training.
    3. Take the majority vote of just those trees and compare it to the true label.
    4. Average the errors over the whole OOB dataset.

    Because every prediction comes only from trees that never trained on that row, the estimate is essentially unbiased — it plays the same role as a hold-out score, but costs nothing extra. In sklearn this is `oob_score=True`.

    **Caveat:** OOB uses fewer trees per prediction than the full forest (~37% of them), so it tends to be slightly pessimistic, and it needs enough trees to be stable. It's a great cheap check, not a replacement for CV when you're doing serious model selection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Ensembles need better-than-chance, weakly-correlated members.** Condorcet cuts both ways: below $p=0.5$, more voters make it worse.
    - **Bootstrap = resample $N$ rows with replacement.** It buys you confidence intervals for any statistic without distributional assumptions.
    - **Bagging = bootstrap + aggregate.** Fit one model per resample, average or vote.
    - **Bagging reduces variance, not bias.** Theoretical best case is a $1/n$ error drop, and only if errors are uncorrelated.
    - **So bag high-variance, low-bias base models** — deep trees. Bagging a stump or a linear model is a waste.
    - **Real gains fall short of $1/n$** because bootstrap samples overlap heavily and errors correlate. Random forests fix that by decorrelating splits.
    - **Small data benefits most.** On big data, shrink the sample size to keep the base models diverse.
    - **Each model sees ~63% of rows, misses ~37%** — from $(1 - 1/\ell)^\ell \to 1/e$.
    - **OOB error is free validation.** Score each row using only the trees that never trained on it; no CV split needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Bagging is the entry-level ensemble: draw $M$ bootstrap samples (size-$N$ resamples with replacement) from the training set, fit an independent base model on each, and average or vote their predictions. Bootstrapping alone is already useful — resampling the churn dataset 1000 times gives non-overlapping 95% confidence intervals for the mean number of support calls in the loyal and churned groups, confirming that frustrated customers call more. Applied to models, averaging $n$ unbiased, uncorrelated learners cuts mean squared error to $E_1/n$, which in bias–variance terms means bagging attacks the **variance** term while leaving bias and irreducible noise alone — so the base learner should be low-bias and high-variance, i.e. a deep tree. Real gains are smaller than the theory because bootstrap samples overlap and their errors correlate, the motivation for random forests in Part 2. A useful side effect: since $(1-1/\ell)^\ell \to 1/e$, each model trains on ~63% of rows and misses ~37%, and scoring each row on only the models that never saw it gives an out-of-bag error — an unbiased validation estimate for free, no cross-validation required.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **ensemble** = a set of models whose combined prediction beats any single member
    - **Condorcet's jury theorem** = majority of independent voters, each correct with $p > 0.5$, tends to correct as $N \to \infty$ (and to wrong if $p < 0.5$)
    - **wisdom of the crowd** = averaged independent guesses beat almost every individual guess
    - **bootstrapping** = resampling $N$ elements from a size-$N$ sample uniformly with replacement
        - **bootstrap sample** = one such resample; contains duplicates, misses ~37% of originals
    - **confidence interval** = percentile range of a statistic across bootstrap resamples; a statement about the procedure's coverage, not about where the values lie
    - **bagging** = bootstrap aggregating: one base model per bootstrap sample, predictions averaged or voted
        - **base model / base algorithm** = the individual learner being bagged (typically a deep tree)
    - **`BaggingClassifier` / `BaggingRegressor`** = sklearn meta-estimators wrapping any base estimator
    - **bias-variance decomposition** = $\mathrm{Err} = \mathrm{Bias}^2 + \mathrm{Var} + \sigma^2$
        - **bias** = systematic error of the model class; bagging does not reduce it
        - **variance** = sensitivity of the fit to the particular training set; the term bagging reduces
        - **irreducible error ($\sigma^2$)** = noise floor no model can beat
    - **uncorrelated errors** = the assumption $\mathbb{E}[\varepsilon_i \varepsilon_j] = 0$ that gives the $1/n$ error reduction; false in practice
    - **out-of-bag (OOB) instances** = the ~37% of rows excluded from a given bootstrap sample
    - **OOB error** = error from predicting each row using only the models that never trained on it
    - **`oob_score=True`** = sklearn flag that computes OOB error for you
    """)
    return


if __name__ == "__main__":
    app.run()
