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
    # Topic 4 — Part 3: Regularization in Logistic Regression, Worked Example

    Part 1 claimed that polynomial features let a *linear* model draw *nonlinear* boundaries. This part makes that claim visible, and then shows the price: once the feature space is big enough to bend the boundary anywhere, the only thing standing between you and a memorized training set is the regularization strength `C`. We fit the same model three times at three values of `C`, watch it go from useless to sensible to insane, and then pick `C` properly with cross-validation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    The data is the microchip QA set from Andrew Ng's ML course: 118 chips, two quality-control test scores, and a binary label for whether the chip shipped. The features are already **centered** (each column has had its mean subtracted), so `(0, 0)` is the "average" chip — worth noting, because regularization penalizes weights and is therefore not scale-invariant.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import PolynomialFeatures

    DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"

    data = pd.read_csv(
        DATA_PATH + "microchip_tests.txt",
        header=None,
        names=("test1", "test2", "released"),
    )
    data.info()
    data.head()
    return (
        LogisticRegression,
        LogisticRegressionCV,
        PolynomialFeatures,
        StratifiedKFold,
        data,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Split into a plain NumPy design matrix `X` and target `y`. Two features only — that is the whole point, since a 2-D feature space is one we can actually draw.
    """)
    return


@app.cell
def _(data, plt):
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Released")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="orange", label="Faulty")
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")
    plt.title("Microchip tests")
    plt.legend()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the scatter before writing any model code. The good chips sit in a blob near the middle, the faulty ones ring the outside. **No straight line separates these classes.** Plain logistic regression on `test1`/`test2` is dead on arrival — which is exactly the situation polynomial features exist for.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Polynomial features: buying nonlinearity with dimensions

    For two variables $x_1, x_2$, the degree-$d$ terms are

    $$\{x_1^i x_2^j\}_{\,i+j \le d},\qquad i, j \in \mathbb{N}$$

    so degree 3 gives $1, x_1, x_2, x_1^2, x_1x_2, x_2^2, x_1^3, x_1^2x_2, x_1x_2^2, x_2^3$. The model stays linear **in the weights** — it is still logistic regression — but the boundary in the original $(x_1, x_2)$ plane is now a curve.

    The catch: the count of these terms grows like $\binom{d+n}{n}$ in the number of original variables $n$. Two variables at degree 7 is a harmless 36 columns; 100 variables at degree 10 is astronomical and would never be worth building. The trap is that this blow-up is not just a compute cost — every extra column is another dimension in which the model can contort itself to fit noise. Which is the whole story of the rest of this notebook.
    """)
    return


@app.cell
def _(PolynomialFeatures, X):
    poly = PolynomialFeatures(degree=7)
    X_poly = poly.fit_transform(X)
    X_poly.shape  # (118, 36) -- 36 features from 2, and only 118 rows to constrain them
    return X_poly, poly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    36 features against 118 rows. That ratio is the danger sign: there is more than enough capacity here to draw a boundary that wraps individually around single points.

    To see it, a helper that evaluates the classifier on a fine grid over the plane and contours where its prediction flips. Note it takes the `poly_featurizer` — grid points are raw 2-D, so they need the same transform the model was trained on. **Applying the identical transform to train and to anything you predict on is not optional**; skip it and the model silently scores garbage.
    """)
    return


@app.cell
def _(np, plt):
    def plot_boundary(clf, X, grid_step=0.01, poly_featurizer=None):
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step)
        )
        Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
        plt.contour(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.Paired)

    return (plot_boundary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same model at three values of C

    Nothing changes below except `C`. Same data, same 36 features, same solver. Watch the boundary and the training accuracy together:

    - **`C = 0.01`** — regularization crushes the weights toward zero. The boundary is barely a shape and training accuracy is ~0.63, hardly better than guessing the majority class. This is **underfitting**.
    - **`C = 1`** — a smooth, roughly elliptical boundary that matches the shape your eye saw in the scatter. Training accuracy ~0.83.
    - **`C = 10000`** — regularization is effectively off. The boundary grows tendrils to swallow individual points. Training accuracy climbs to ~0.88. This is **overfitting**.

    The lesson is in the *gap between* the second and third: rising training accuracy bought a wildly more fragile boundary. `C=1` gives up ~5 accuracy points on data it has already seen, and would almost certainly win on data it hasn't. **Training accuracy going up is not evidence the model got better** — that is the trap this whole example exists to name.
    """)
    return


@app.cell
def _(LogisticRegression, X, X_poly, plot_boundary, plt, poly, y):
    for C, grid_step in [(1e-2, 0.01), (1, 0.005), (1e4, 0.005)]:
        logit = LogisticRegression(C=C, random_state=17)
        logit.fit(X_poly, y)

        plt.figure()
        plot_boundary(logit, X, grid_step=grid_step, poly_featurizer=poly)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="Released")
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c="orange", label="Faulty")
        plt.xlabel("Test 1")
        plt.ylabel("Test 2")
        plt.title("Logit with C=%s" % C)
        plt.legend()
        print(f"C={C:>8}  accuracy on training set:", round(logit.score(X_poly, y), 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why C does that

    Logistic regression with $L_2$ regularization minimizes

    $$J(X, y, w) = \mathcal{L} + \frac{1}{C}\|w\|^2$$

    where $\mathcal{L}$ is the logistic loss summed over the dataset, and $C$ is sklearn's **inverse** regularization strength. The two terms compete for the objective's attention, and $C$ sets the exchange rate:

    - **Small `C`** ⇒ the penalty $\frac{1}{C}\|w\|^2$ dominates. The cheapest way to shrink $J$ is to drive weights toward zero, so the model is barely punished for being wrong. Underfit.
    - **Large `C`** ⇒ the penalty is negligible and $\mathcal{L}$ is everything. Weights are free to blow up, and the model becomes terrified of a single training mistake. Overfit.
    - So **`C` is effectively a model-capacity dial**: bigger `C` = more complex relationships recoverable.

    The critical structural point: **`C` cannot be learned the way $w$ is.** Minimizing $J$ over $w$ tells you nothing about which $C$ to use — larger `C` always fits the training set at least as well, so "optimize $J$" would just pick $C = \infty$. It is a **hyperparameter**, chosen from *outside* the training loop, on held-out data. Same structure as `max_depth` in a decision tree back in Topic 3: the training procedure cannot pick it, because the training objective always prefers the more flexible model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tuning C with cross-validation

    So we search. `LogisticRegressionCV` does grid search + cross-validation in one object, specialized to logistic regression (it exploits warm starts along the `C` path, so it's much faster than the generic route). For an arbitrary estimator you'd reach for `GridSearchCV` / `RandomizedSearchCV`, or something smarter like `hyperopt`.

    `StratifiedKFold` is the right splitter here, not plain `KFold`: it keeps each fold's class balance equal to the full set's. With 118 rows a naive random split can hand you a fold that's lopsided, and the resulting score is noise.
    """)
    return


@app.cell
def _(LogisticRegressionCV, StratifiedKFold, X_poly, np, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    c_values = np.logspace(
        -2, 3, 500
    )  # log scale: C matters multiplicatively, not additively

    logit_searcher = LogisticRegressionCV(Cs=c_values, cv=skf, n_jobs=-1)
    logit_searcher.fit(X_poly, y)

    logit_searcher.C_
    return c_values, logit_searcher


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two details worth internalizing. **Search `C` on a log scale** (`np.logspace`) — the difference between 0.01 and 0.1 is the same *kind* of change as between 10 and 100, so a linear grid wastes nearly every point at the useless high end. And `scores_` is a dict keyed by class label, holding a (folds × Cs) array of validation accuracies; averaging over axis 0 collapses the folds and leaves accuracy as a function of `C` — a **validation curve**.
    """)
    return


@app.cell
def _(c_values, logit_searcher, np, plt):
    mean_cv = np.mean(logit_searcher.scores_[1], axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(c_values, mean_cv)
    ax1.set_xlabel("C")
    ax1.set_ylabel("Mean CV-accuracy")
    ax1.set_title("Full range")

    ax2.plot(c_values, mean_cv)
    ax2.set_xlabel("C")
    ax2.set_ylabel("Mean CV-accuracy")
    ax2.set_xlim((0, 10))
    ax2.set_title("Zoomed on the useful region")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The full-range plot is mostly a flat line — because past `C ≈ 10` almost nothing happens, which is itself the finding: **the curve is a plateau, not a peak.** The search reports `C_ ≈ 161.6`, but the zoom shows accuracy already at ~0.822 by `C ≈ 10` and drifting sideways after. Don't read the argmax as a meaningful number; on 118 rows that winning `C` beats `C ≈ 10` by 0.0004 accuracy — fold-assignment luck, nothing more. Prefer the smallest `C` on the plateau — same CV score, simpler boundary, better odds on new data.

    These are **validation curves**. We drew this one by hand from `scores_`; sklearn ships `validation_curve` / `learning_curve` to build them directly, which later topics use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Polynomial features make a linear model nonlinear** — the boundary curves in the original space while the model stays linear in its weights.
    - **Feature count explodes with degree**, and each extra dimension is extra room to fit noise. Degree 7 on 2 variables = 36 features against 118 rows.
    - **`C` is inverse regularization strength**, minimizing $J = \mathcal{L} + \frac{1}{C}\|w\|^2$. Small `C` = strong penalty = underfit; large `C` = free weights = overfit.
    - **`C` is a capacity dial**, the direct analogue of a tree's `max_depth`.
    - **`C` cannot be learned by the training objective** — it always prefers more flexibility. It is a hyperparameter, tuned on held-out data.
    - **Rising training accuracy is not progress.** `C=1e4` scored higher on training than `C=1` while being obviously worse.
    - **Search `C` on a log scale**, and **stratify the folds** on small or imbalanced data.
    - **Read the validation curve, not the argmax.** On a plateau, take the smallest `C` that reaches it.
    - **Transform predict-time inputs exactly as you transformed the training data** — the featurizer travels with the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    This part turns the abstract regularization story into a picture: on 118 microchips whose classes are visibly not linearly separable, degree-7 polynomial features give logistic regression 36 columns and more than enough capacity to bend a boundary anywhere, so `C` alone decides what comes out. Fitting the identical model at `C = 0.01, 1, 10^4` walks the boundary from a shapeless underfit (63% train accuracy) through a smooth, believable ellipse (83%) to a tendril-covered overfit that wraps individual points (88%) — proving that training accuracy climbing while the model gets worse is the normal case, not an anomaly. The objective $J = \mathcal{L} + \frac{1}{C}\|w\|^2$ explains why: `C` sets the exchange rate between fitting and weight-shrinking, and because the objective always rewards more flexibility, `C` can never be learned from the training set — it is a hyperparameter, like a tree's `max_depth`, and must be chosen outside the fit. `LogisticRegressionCV` with stratified folds over a log-spaced grid does that choosing, and plotting mean CV accuracy against `C` yields a validation curve whose broad plateau is the real answer: pick the smallest `C` that reaches it and ignore the precise argmax.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **regularization** = penalizing large weights in the loss to limit model complexity
    - **`C`** = sklearn's inverse regularization strength; large `C` = weak penalty = more capacity
    - **$L_2$ penalty** = the $\|w\|^2$ term added to the loss
    - **logistic loss** = the $\mathcal{L}$ term; error summed over the dataset
    - **underfitting** = model too constrained to capture real structure (`C` too small)
    - **overfitting** = model fits training noise, generalizes badly (`C` too large)
    - **model capacity** = how complex a relationship the model can represent
    - **polynomial features** = products $x_1^i x_2^j$ added as new columns to make a linear model nonlinear
        - **`PolynomialFeatures`** = sklearn transformer that builds them up to a given `degree`
        - **degree** = highest total exponent $i + j$ of the generated terms
    - **centered features** = columns with their mean subtracted, so 0 is the average
    - **hyperparameter** = a knob set outside the training loop, tuned on held-out data (`C`, `max_depth`)
    - **cross-validation** = scoring on held-out folds to estimate performance on new data
        - **`StratifiedKFold`** = fold splitter preserving class proportions in each fold
    - **`LogisticRegression`** = the linear classifier being fit
    - **`LogisticRegressionCV`** = grid search + cross-validation specialized for logistic regression
        - **`Cs`** = the grid of `C` values it searches
        - **`C_`** = the best `C` it selected
        - **`scores_`** = per-class (folds × Cs) array of validation accuracies
    - **`GridSearchCV`** = generic exhaustive hyperparameter search for any estimator
    - **validation curve** = plot of validation score against one hyperparameter
    - **`np.logspace`** = log-spaced grid, the right shape for searching `C`
    - **decision boundary** = the surface where the classifier's predicted class flips
    """)
    return


if __name__ == "__main__":
    app.run()
