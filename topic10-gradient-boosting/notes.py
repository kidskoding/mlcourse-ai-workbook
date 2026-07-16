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
    # Topic 10: Gradient Boosting

    **Gradient boosting (GBM)** builds a strong model as a *sum of many weak ones*, where each new weak model is fit to the errors the ensemble has made so far. On tabular data — anything that isn't images, audio, or extremely sparse text — it is usually the algorithm to beat, and it is almost always a component of whatever wins a Kaggle competition.

    The key idea to hold onto: GBM is **not one algorithm**, it's a *recipe*. Pick a differentiable loss, pick a family of base learners (almost always shallow trees), and GBM turns "minimize this loss" into a sequence of ordinary regression fits. Change the loss and you change the whole character of the model — mean vs. median vs. quantile, regression vs. classification — with no change to the machinery underneath.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where it came from: weak learners and AdaBoost

    Boosting started as a theoretical question: can you combine many **weak learners** — models barely better than a coin flip — into one arbitrarily accurate strong learner? Schapire proved yes, and **AdaBoost** was the first practical answer.

    AdaBoost is greedy and reweights the *data*: train a stump (a depth-1 tree), find which points it got wrong, crank up their weights, train the next stump on the reweighted data, repeat. The final prediction is a weighted vote of the stumps:

    - init $w_i = 1/\ell$
    - for $t = 1 \dots T$: train $b_t$, get its weighted error $\epsilon_t$, set $\alpha_t = \tfrac12 \ln\frac{1-\epsilon_t}{\epsilon_t}$, update $w_i \leftarrow w_i e^{-\alpha_t y_i b_t(x_i)}$ and renormalize
    - return $\sum_t \alpha_t b_t$

    Note what $\alpha_t$ does: an accurate stump ($\epsilon_t$ near 0) gets a big vote, a coin-flip stump gets roughly none.

    **The trap AdaBoost walks into:** growing the weight of every misclassified point means **outliers eventually dominate**. A permanently-wrong noisy label gets an exponentially growing weight until the whole ensemble bends around it. AdaBoost is unstable on noisy data, and for years nobody could explain *why* it worked when it worked.

    Friedman (1999) resolved this by re-deriving boosting as **gradient descent in function space**. That reframing came with a bonus: AdaBoost turns out to be just GBM with one particular loss ($e^{-yf}$). It stopped being a clever heuristic and became a statistical method you can extend.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The core trick: gradient descent, but on functions

    Standard setup. Given $\{(x_i, y_i)\}_{i=1}^n$, find $\hat{f}$ minimizing expected loss:

    $$\hat{f}(x) = \arg\min_{f} \; \mathbb{E}_{x,y}\big[L(y, f(x))\big]$$

    The only requirement on $L$ is that it be **differentiable in $f$**. No assumptions about the shape of the true dependence or the distribution of $y$.

    The problem: the space of all functions $f$ is infinite-dimensional. The usual escape is to restrict to a parametric family $f(x, \theta)$ and gradient-descend on $\theta$ — that's how you fit a linear model or a net. Boosting takes a **different escape**: keep the search in function space, but express the answer as a *sum of increments*, each increment itself a function:

    $$\hat{f}(x) = \sum_{i=0}^{M} \hat{f}_i(x)$$

    At step $t$ you want the increment $\rho \cdot h(x,\theta)$ that most reduces the loss:

    $$(\rho_t, \theta_t) = \arg\min_{\rho, \theta} \; \mathbb{E}_{x,y}\big[L\big(y, \hat{f}(x) + \rho \cdot h(x, \theta)\big)\big]$$

    Solving that directly, for an arbitrary $L$ and an arbitrary model family $h$, is hopeless. **Here is the whole trick.** In ordinary gradient descent you step along $-\nabla L$. You can still *evaluate* that gradient — pointwise, at each training point:

    $$r_{it} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f(x) = \hat{f}(x)}$$

    Those $n$ numbers are the direction you'd like to move in, but they're just values at the training points, not a function. So **fit a model to them** — by plain least squares, regardless of what $L$ was:

    $$\theta_t = \arg\min_{\theta} \sum_{i=1}^{n} \big(r_{it} - h(x_i, \theta)\big)^2$$

    That's it. The base learner's job is never to predict $y$. Its job is to **predict the gradient**, so that adding it takes a step downhill. This is why any differentiable loss plugs into the same code path, and why the base learner is always a *regressor* even when you're doing classification.

    The $r_{it}$ are called **pseudo-residuals**. For L2 loss they literally *are* the residuals $y - \hat{f}$ — which is why the "fit the next tree to the leftover error" folk explanation of boosting is correct, but only for that one loss.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Friedman's classic GBM

    Assemble the pieces. You must supply: the data; the number of iterations $M$; a differentiable loss $L$; a base-learner family $h(x,\theta)$ and its hyperparameters (for trees: depth).

    1. **Initialize with a constant**: $\hat{f}_0 = \gamma$, where $\gamma = \arg\min_\gamma \sum_i L(y_i, \gamma)$ — the single best guess under this loss (for L2 that's the mean, for L1 the median; for many losses there's no formula, so you line-search).
    2. For $t = 1 \dots M$:
    3. Compute pseudo-residuals $r_{it} = -\big[\partial L(y_i, f(x_i)) / \partial f(x_i)\big]_{f = \hat{f}}$
    4. Fit base learner $h_t(x)$ by **regression onto $\{(x_i, r_{it})\}$**
    5. Line-search the step size against the *original* loss: $\rho_t = \arg\min_\rho \sum_i L\big(y_i, \hat{f}(x_i) + \rho \cdot h_t(x_i)\big)$
    6. Set $\hat{f}_t(x) = \rho_t \cdot h_t(x)$ and update $\hat{f} \leftarrow \hat{f} + \hat{f}_t$
    7. Final model: $\hat{f}(x) = \sum_{i=0}^{M} \hat{f}_i(x)$

    Steps 3–5 are the loop: **evaluate gradient → fit a tree to it → line search → add**. Note step 5 uses $L$ itself, not its gradient; the gradient only ever tells you the *direction*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step by step on a toy problem

    Restore a noisy cosine: $y = \cos(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, 1/5)$, $x \in [-5, 5]$, 300 points, depth-2 trees, $M = 3$.

    With L2 loss everything simplifies: $\gamma$ is the mean of $y$, the gradient is exactly the residual $r = y - \hat{f}$, and all $\rho_t = 1$. So a from-scratch GBM is about ten lines — worth writing once so the loop stops being magic.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.RandomState(17)
    n = 300
    X = np.linspace(-5, 5, n).reshape(-1, 1)
    y = np.cos(X).ravel() + rng.normal(0, 0.2, n)
    return DecisionTreeRegressor, X, n, np, plt, rng, y


@app.cell
def _(DecisionTreeRegressor, X, np, plt, y):
    # Friedman's GBM with L2 loss, from scratch: yields (approximation, tree fit) per iteration
    def gbm_l2(X, y, M=3, depth=2, lr=1.0):
        f = np.full(len(y), y.mean())  # step 1: gamma = mean under L2
        for _ in range(M):
            r = y - f  # step 3: pseudo-residuals == residuals for L2
            h = DecisionTreeRegressor(max_depth=depth).fit(
                X, r
            )  # step 4: regress onto the gradient
            f = f + lr * h.predict(X)  # steps 5-6: rho = 1 for L2
            yield f.copy(), h.predict(X)

    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    for t, (f, tree_pred) in enumerate(gbm_l2(X, y, M=3)):
        axes[0, t].scatter(X, y, s=5, alpha=0.3)
        axes[0, t].plot(X, f, "b", lw=2)
        axes[0, t].set_title(f"approximation, iteration {t + 1}")
        axes[1, t].plot(X, tree_pred, "g", lw=2)
        axes[1, t].set_title(f"tree fit to pseudo-residuals, iteration {t + 1}")
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Watch the first iteration: the fit only picks up the **left** branch of the cosine. A depth-2 tree doesn't have enough splits to build both bumps at once, so it spends them where the error is biggest and ignores the rest. The right branch shows up only at iteration 2. That's boosting's character in miniature — **each tree is a greedy, myopic patch on the largest remaining error**, and the ensemble is correct only in aggregate.

    Also note the staircase: sums of trees are step functions. GBM will **never** be smooth on a smooth target. It approximates, it doesn't interpolate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss functions I: regression

    Switching the loss is how you choose *which property of the conditional distribution $(y \mid x)$ you want back*. That's the real decision; everything else is plumbing.

    - **L2 / Gaussian**, $L = (y-f)^2$. Gradient is $y - f$. Recovers the **conditional mean**. The default when you have no reason to want otherwise.
    - **L1 / Laplacian**, $L = |y-f|$. Gradient is $\mathrm{sign}(y-f)$ — so the tree sees only $\pm 1$ and *the magnitude of the error is ignored*. Recovers the **conditional median**, and is therefore **robust to outliers**: a point that's wrong by 1000 pulls exactly as hard as one wrong by 1.
    - **Quantile / $L_q$**, an asymmetric L1 with $\alpha \in (0,1)$:

    $$L(y,f) = \begin{cases}(1-\alpha)\,|y-f|, & y - f \le 0\\ \alpha\,|y-f|, & y - f > 0\end{cases}$$

      Recovers the **conditional $\alpha$-quantile**. Gradient is $\alpha \,\mathbb{I}(y_i > \hat f) - (1-\alpha)\,\mathbb{I}(y_i \le \hat f)$ — again only two values. Initialize at the sample $\alpha$-quantile.
    - **Huber**: L2 near zero, L1 past a threshold. Best of both — quadratic sensitivity where it matters, robustness where the outliers live. Often the best fit available if you tune the threshold, but **library support is patchy** (h2o has it, XGBoost doesn't).

    Fitting the 75%-quantile to the cosine gives the same shape as L2, just shifted up by about 0.135. **The trap:** push to the 90%-quantile and there simply aren't enough points above the line to fit — the effective sample goes lopsided and the estimate falls apart. Extreme quantiles need data you probably don't have.
    """)
    return


@app.cell
def _(X, plt, y):
    from sklearn.ensemble import GradientBoostingRegressor

    # same data, four losses -> four different things recovered
    fits = {
        "L2 (mean)": GradientBoostingRegressor(
            loss="squared_error", max_depth=2, n_estimators=100
        ),
        "L1 (median)": GradientBoostingRegressor(
            loss="absolute_error", max_depth=2, n_estimators=100
        ),
        "quantile a=0.75": GradientBoostingRegressor(
            loss="quantile", alpha=0.75, max_depth=2, n_estimators=100
        ),
        "huber": GradientBoostingRegressor(loss="huber", max_depth=2, n_estimators=100),
    }

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, s=5, alpha=0.2, label="data")
    for name, model in fits.items():
        plt.plot(X, model.fit(X, y).predict(X), lw=2, label=name)
    plt.legend()
    plt.title("same GBM, same data, different loss")
    return (GradientBoostingRegressor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss functions II: classification

    For binary classification, encode $y \in \{-1, 1\}$ (**not** 0/1 — the losses below are written in terms of the **margin** $y \cdot f$). You *could* run L2 regression against $\pm 1$, but it's wrong: it penalizes predictions for being *too correct*, since overshooting past $y=1$ raises the squared error.

    Margin-based losses instead:

    - **Logistic / Bernoulli**, $L = \log(1 + e^{-2yf})$, gradient $r_i = \dfrac{2 y_i}{1 + e^{2 y_i \hat f(x_i)}}$. It decays but never hits zero, so it keeps mildly penalizing *already-correct* points. That's a feature: it keeps pushing the classes further apart instead of stopping the moment the sign is right.
    - **Exponential / AdaBoost loss**, $L = e^{-yf}$. GBM with this loss *is* classic AdaBoost. Same shape as logistic, but the penalty for a wrong prediction explodes exponentially — which is exactly the outlier sensitivity AdaBoost is known for. **Prefer logistic unless you have a reason.**

    Note $\hat{f}$ here is a real-valued score, not a probability; you squash it afterwards. And initialization gets awkward: the classes are imbalanced, and there's no closed form for $\gamma$ under logistic loss, so you **line-search it** — the cell below does exactly that. Expect it to come out **negative**: when in doubt it pays to lean toward the majority class. No formula gives you the exact value; only the search does.
    """)
    return


@app.cell
def _(X, n, np, plt, rng):
    from sklearn.ensemble import GradientBoostingClassifier
    from scipy.optimize import minimize_scalar

    # classification toy data: sign of the noisy cosine, y in {-1, 1}
    y_cls = np.sign(np.cos(X).ravel() + rng.normal(0, 0.5, n))
    print("class balance:", np.round(np.bincount((y_cls > 0).astype(int)) / n, 2))

    # line-search the constant initialization gamma under logistic loss (no analytic formula)
    logistic = lambda g: np.log(1 + np.exp(-2 * y_cls * g)).sum()
    print("optimal f0 =", round(minimize_scalar(logistic).x, 3))

    clf = GradientBoostingClassifier(max_depth=2, n_estimators=100).fit(X, y_cls)
    plt.figure(figsize=(10, 4))
    plt.scatter(
        X, y_cls + rng.normal(0, 0.05, n), s=5, alpha=0.3
    )  # jitter for visibility
    plt.plot(X, clf.decision_function(X), "b", lw=2, label="decision function f(x)")
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weights: the cheap alternative to a custom loss

    Real problems want opinions the standard losses don't have. Weight large moves more in a financial series; care more about churn from high-LTV customers.

    The heroic path is to invent your own loss, derive its gradient (and Hessian, if you want it to train fast), and verify it has the properties you claimed. **The high-probability outcome of that path is a subtle mistake and a lost week.**

    The lazy path, which people forget exists: **weight the observations**. Any $w_i \ge 0$ with $\sum_i w_i > 0$. It drops straight through the math — scale the loss, and the gradient scales with it:

    $$L_w(y,f) = w \cdot L(y,f), \qquad r_{it} = -w_i \left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f = \hat{f}}$$

    Class-balance weights are the trivial case. Anything beyond that is creativity.

    **The caveat, and it's real:** with arbitrary weights you no longer know your model's statistical properties. And weighting is *not* a substitute for changing the loss — e.g. L1 weighted by $|y|$ is **not** L2, because the gradient still ignores the prediction $\hat{f}(x)$ entirely. Weights are powerful and risky. **Rule of thumb: try the simple loss plus weights before you write a custom gradient.**
    """)
    return


@app.cell
def _(GradientBoostingRegressor, X, np, plt, y):
    # strongly asymmetric weights: ignore x <= 0, care about x > 0
    w = np.where(X.ravel() <= 0, 0.1, 0.1 + np.abs(np.cos(X.ravel())))

    plain = GradientBoostingRegressor(max_depth=2, n_estimators=30).fit(X, y)
    weighted = GradientBoostingRegressor(max_depth=2, n_estimators=30).fit(
        X, y, sample_weight=w
    )

    plt.figure(figsize=(10, 4))
    plt.scatter(X, y, s=5, alpha=0.2)
    plt.plot(X, plain.predict(X), "b", lw=2, label="unweighted")
    plt.plot(X, weighted.predict(X), "r", lw=2, label="weighted (right side favored)")
    plt.legend()
    plt.title("weights buy detail on the right, spend it from the left")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What was deliberately left out — and why $M=3$

    Everything above used $M = 3$, and that was hiding something. Regularization, stochasticity, and hyperparameter tuning are the entire reason GBM is usable in practice, because **GBM will happily overfit to death**. Each tree fits the previous ensemble's errors, and past a point those "errors" are just noise — which it will then fit perfectly.

    The standard defenses:

    - **Learning rate / shrinkage** — multiply each increment by $\nu \approx 0.01\!-\!0.1$. Take many small steps instead of few big ones. This is the single most important knob, and it trades off against $M$: a lower $\nu$ needs more trees.
    - **Shallow trees** — depth 2–6. The weak learner should stay weak.
    - **Stochasticity** — subsample rows/columns per tree.
    - **Early stopping** — pick $M$ by validation score, not by faith.
    """)
    return


@app.cell
def _(GradientBoostingRegressor, X, plt, y):
    # same data, M=3 vs M=300 at full learning rate: the second one fits the noise
    plt.figure(figsize=(10, 4))
    plt.scatter(X, y, s=5, alpha=0.2)
    for M, style in [(3, "b"), (300, "r")]:
        m = GradientBoostingRegressor(
            max_depth=2, n_estimators=M, learning_rate=1.0
        ).fit(X, y)
        plt.plot(X, m.predict(X), style, lw=1.5, label=f"M={M}, lr=1.0")

    # the fix: many small steps, not few big ones
    m = GradientBoostingRegressor(
        max_depth=2, n_estimators=300, learning_rate=0.05
    ).fit(X, y)
    plt.plot(X, m.predict(X), "g", lw=2, label="M=300, lr=0.05 (shrinkage)")
    plt.legend()
    plt.title("overfitting, and what shrinkage does about it")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **GBM is a methodology, not an algorithm.** Differentiable loss + any base learner + "fit the next model to the gradient" = a family of methods, not one method.
    - **The base learner regresses onto the gradient, never onto $y$.** This is why one code path handles every loss, and why base learners are regressors even in classification.
    - **Pseudo-residuals equal true residuals only for L2.** The "each tree fixes the leftover error" story is a special case, not the definition.
    - **The loss picks what you recover:** L2 → mean, L1 → median, quantile → quantile, Huber → robust mean. Choose it before you touch hyperparameters.
    - **Use margin losses ($y \in \{-1,1\}$) for classification.** Logistic by default; the exponential loss is AdaBoost and is outlier-hungry.
    - **Reach for weights before a custom loss.** Far cheaper and far less error-prone — but they invalidate your statistical guarantees and can't emulate a different loss.
    - **Boosting overfits.** Shrinkage ($\nu = 0.01\!-\!0.1$) plus early stopping plus shallow trees is the standard defense; $\nu$ and $M$ trade off against each other.
    - **Sums of trees are step functions.** No smooth fits, no extrapolation.
    - **GBM is the default for tabular data**, and a component of essentially every serious stack — but not for images, audio, or very sparse data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Boosting began with the question of whether many weak learners can be combined into a strong one; AdaBoost answered yes by iteratively reweighting misclassified points, but it was unstable on outliers and poorly understood. Friedman's GBM reframed it as gradient descent in *function space*: represent the model as a sum of increments, evaluate the loss gradient pointwise at the training data to get **pseudo-residuals**, fit a base learner to those by ordinary least squares, line-search a step size, and add. That single loop accommodates any differentiable loss — L2 for the conditional mean, L1 for the median, quantile loss for a quantile, Huber for robustness, logistic or exponential (= AdaBoost) for margin-based classification — and observation **weights** provide a cheap, low-risk substitute for hand-deriving a bespoke loss. The article's $M=3$ examples deliberately dodge the practical half of GBM: with enough iterations at full step size it overfits noise aggressively, so real usage depends on shrinkage, shallow trees, subsampling, and early stopping. The through-line: **choose the loss to name what you want back from the data, then let the same machinery go get it.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **boosting** = build a strong model as a sum of many weak ones, each fit to what the ensemble got wrong
    - **weak learner** = a model only slightly better than random guessing
    - **AdaBoost** = the original boosting algorithm; reweights misclassified points each round
        - **stump** = a depth-1 decision tree, AdaBoost's usual base learner
    - **GBM** = Gradient Boosting Machine; boosting recast as gradient descent in function space
        - **pseudo-residual** = negative loss gradient evaluated at a training point; the target the next base learner regresses onto
        - **base learner** = the weak model fit at each iteration (usually a shallow regression tree)
        - **initial approximation** = the constant $\gamma$ minimizing the loss; where boosting starts
        - **line search** = numerically finding the step size $\rho_t$ against the original loss
    - **functional gradient descent** = optimizing over functions by adding increments along the negative gradient
    - **loss function** = what you minimize; must be differentiable in $f$; determines which property of $(y \mid x)$ you recover
    - **L2 / Gaussian loss** = $(y-f)^2$; recovers the conditional mean; its gradient is the plain residual
    - **L1 / Laplacian loss** = $|y-f|$; recovers the conditional median; robust to outliers
    - **quantile loss** = asymmetric L1 with weight $\alpha$; recovers the conditional $\alpha$-quantile
    - **Huber loss** = L2 near zero, L1 beyond a threshold; robust but thinly supported in libraries
    - **margin** = $y \cdot f$ with $y \in \{-1,1\}$; the quantity classification losses are written in terms of
    - **logistic / Bernoulli loss** = $\log(1+e^{-2yf})$; default for binary classification; still penalizes correct-but-close points
    - **exponential / AdaBoost loss** = $e^{-yf}$; GBM with this loss reproduces AdaBoost; punishes errors exponentially
    - **weights** = per-observation multipliers $w_i \ge 0$ scaling the loss and its gradient; a cheap alternative to a custom loss
    - **shrinkage / learning rate** = scale each increment by $\nu \approx 0.01\!-\!0.1$; the main regularizer, trades off against $M$
    - **early stopping** = choose the number of iterations $M$ by validation score
    - **XGBoost** = an efficient implementation of classic GBM plus heuristics, not a new algorithm
    """)
    return


if __name__ == "__main__":
    app.run()
