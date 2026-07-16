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
    # Topic 4 — Part 5: Validation and Learning Curves

    Once you know how to cross-validate and regularize, the real question shows up: **the model is not good enough — now what?** More complexity? Less? More features? More rows? Guessing here is expensive. Two diagnostic plots answer it almost mechanically:

    - **Validation curve** — score vs. *model complexity* (here: the regularization strength).
    - **Learning curve** — score vs. *training set size*.

    Read them and the next move is usually obvious. Same telecom churn data as Topic 1.
    """)
    return


@app.cell
def _():
    import warnings
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import learning_curve, validation_curve
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = "retina"
    warnings.filterwarnings("ignore")

    DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    data = pd.read_csv(DATA_URL + "telecom_churn.csv").drop("State", axis=1)
    data["International plan"] = data["International plan"].map({"Yes": 1, "No": 0})
    data["Voice mail plan"] = data["Voice mail plan"].map({"Yes": 1, "No": 0})

    y = data["Churn"].astype("int").values
    X = data.drop("Churn", axis=1).values
    return (
        Pipeline,
        PolynomialFeatures,
        SGDClassifier,
        StandardScaler,
        X,
        learning_curve,
        np,
        plt,
        validation_curve,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model under test: SGD-trained linear SVM in a pipeline

    `SGDClassifier(loss="hinge")` is a linear SVM fit by stochastic gradient descent (Topic 8 covers SGD properly). The knob we vary is `alpha`, the L2 regularization strength: **big alpha = simple model, small alpha = complex model**.

    Two settings stay pinned for the rest of the notebook so both experiments describe the *same* model:

    - `max_iter=5` — SGD gets 5 passes over the data and no more. That cap matters for how the plots look, and it gets called out again below.
    - `degree=2` polynomial features.

    Everything lives in a `Pipeline` so the scaler and the polynomial expansion are refit *inside each CV fold*. That is not cosmetic — scaling on the full dataset before CV leaks test statistics into training and quietly inflates your scores. The pipeline is what makes `param_name="sgd_model__alpha"` (the `step__param` syntax) work.

    The sweep runs `alpha` from `1e-4` to `10` over 20 log-spaced points, and each learning curve below pins `alpha` to one of those 20 points (`1e-4`, `0.0785`, `10`) so the two plots line up on the same x-values. Lining up the *alphas* is not quite the same as lining up the *scores*, though: `learning_curve`'s largest slice is slightly smaller than the training set `validation_curve` uses, and at `alpha=10` that difference turns out to matter. That discrepancy gets measured rather than hand-waved when we reach it.
    """)
    return


@app.cell
def _(
    Pipeline,
    PolynomialFeatures,
    SGDClassifier,
    StandardScaler,
    X,
    np,
    validation_curve,
    y,
):
    alphas = np.logspace(-4, 1, 20)

    sgd_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
            (
                "sgd_model",
                SGDClassifier(loss="hinge", n_jobs=-1, random_state=17, max_iter=5),
            ),
        ]
    )

    val_train, val_test = validation_curve(
        estimator=sgd_pipe,
        X=X,
        y=y,
        param_name="sgd_model__alpha",
        param_range=alphas,
        cv=5,
        scoring="roc_auc",
    )
    return alphas, sgd_pipe, val_test, val_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validation curve: score vs. complexity

    `validation_curve` refits the model once per `alpha` per fold and returns two arrays of shape `(n_params, n_folds)` — train scores and held-out scores. Plot the mean per alpha with a ±1 std band so the fold-to-fold noise is visible; a gap smaller than the bands is not a gap.

    `alphas` is log-spaced, so the x-axis has to be log-scaled too. On a linear axis 15 of the 20 points would pile up against the left edge and the shape would be unreadable.
    """)
    return


@app.cell
def _(alphas, plt, val_test, val_train):
    def plot_with_err(x, data, **kwargs):
        mu, std = data.mean(1), data.std(1)
        lines = plt.plot(x, mu, "-", **kwargs)
        plt.fill_between(
            x,
            mu - std,
            mu + std,
            edgecolor="none",
            facecolor=lines[0].get_color(),
            alpha=0.2,
        )

    plot_with_err(alphas, val_train, label="training scores")
    plot_with_err(alphas, val_test, label="validation scores")
    plt.xscale("log")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("ROC AUC")
    plt.legend()
    plt.grid(True)
    return (plot_with_err,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How to read it

    Two things carry information, and they are independent: the **height** of the pair (fit quality) and the **gap** between them (overfitting). Walk the axis from right to left:

    - **Right end — large alpha, the simple end.** At `alpha=10` both curves have collapsed to roughly 0.64 train / 0.62 validation. Both scores bad *and* glued together is the signature of **underfitting**: the penalty is so heavy the model can barely use its features. Buying more rows here is money burned — there is no capacity to spend them on. Do **not** try to read a gap off this end, though. The bands are enormous (±0.09 train, ±0.10 validation) because the five folds disagree wildly — validation runs 0.47 to 0.75 across them. The rule from the section above applies to the notebook's own plot: whatever separates the two means here is far inside the bands, so it is not a gap.
    - **The peak.** Validation tops out near `alpha≈0.14` at ROC-AUC ≈ 0.884. That is the alpha to ship. Tune to the *validation* peak, never the training one.
    - **Left end — small alpha, the complex end.** Both curves sag again: at `alpha=1e-4`, train ≈ 0.873 and validation ≈ 0.844.

    That left end deserves a note, because it is **not** the textbook overfitting picture and this notebook should not pretend otherwise. The textbook says the training curve shoots toward 1.0 while validation dives, opening a wide gap. Here the gap between the means never exceeds ≈ 0.03 anywhere on the sweep — it runs about 0.015 to 0.030, with no trend that tracks complexity. More to the point, it is nowhere large enough to trust: the validation band is ±0.019 at its tightest (mid-sweep) and ±0.12 at its widest (heavy-penalty end), so the gap is always comparable to or smaller than the fold noise. The reason is `max_iter=5`: SGD is stopped after five passes, long before it can memorize the training set, so the extra capacity that small alpha unlocks is capacity the optimizer never gets to exploit. Both scores just get worse together.

    So on this sweep the gap never rises above the noise and the story is almost entirely about height. Take the general lesson (gap = overfitting, height = fit quality) but read it off the plot you actually have, not the one from the textbook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning curve: score vs. amount of data

    Now fix the complexity and vary the training set size instead — same pipeline, same `max_iter=5`, only `alpha` and the row count change. This answers a question with an actual price tag: *is it worth paying annotators to double the dataset?* You cannot buy the extra data to find out, so you simulate the direction — train on 5%, 10%, … 100% of what you already have and look at the trend.

    `learning_curve` returns `(train_sizes, train_scores, test_scores)`. The helper below wraps it so we can flip `alpha` and compare. Both parameters are required — there is no sensible default complexity, which is the entire point of the exercise.
    """)
    return


@app.cell
def _(
    Pipeline,
    PolynomialFeatures,
    SGDClassifier,
    StandardScaler,
    X,
    learning_curve,
    np,
    plot_with_err,
    plt,
    y,
):
    def plot_learning_curve(degree, alpha):
        train_sizes = np.linspace(0.05, 1, 20)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree)),
                (
                    "sgd_model",
                    SGDClassifier(
                        loss="hinge",
                        n_jobs=-1,
                        random_state=17,
                        alpha=alpha,
                        max_iter=5,
                    ),
                ),
            ]
        )
        N_train, val_train, val_test = learning_curve(
            pipe, X, y, train_sizes=train_sizes, cv=5, scoring="roc_auc"
        )
        plot_with_err(N_train, val_train, label="training scores")
        plot_with_err(N_train, val_test, label="validation scores")
        plt.xlabel("Training Set Size")
        plt.ylabel("AUC")
        plt.legend()
        plt.grid(True)

    return (plot_learning_curve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Heavy regularization (`alpha=10`): converged and mediocre

    This fixes `alpha` at the right-hand end of the validation curve's grid. On the first slice (~133 rows) the two curves are far apart — train ≈ 0.88 against validation ≈ 0.60 — because a handful of rows is easy to fit and tells you nothing. Feed it more and they **converge fast** — the gap falls from ≈ 0.28 to ≈ 0.02 — onto a low, *genuinely* noisy band: validation wanders between 0.56 and 0.72 for the whole right half of the axis and finishes at ≈ 0.69. It is a plateau in the sense that matters (no upward trend), not a flat line.

    That is the whole message: more rows will not move the validation score, because the model is capacity-bound, not data-bound. It has already learned everything a penalty this heavy will let it learn, and it is stuck well below the ≈ 0.884 the peak alpha reaches. This is the business-relevant case — you could 10x the dataset and gain nothing. "Collect data once, reuse forever" is not a strategy if the curves have already met.
    """)
    return


@app.cell
def _(plot_learning_curve):
    plot_learning_curve(degree=2, alpha=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why this curve ends at 0.69 but the validation curve reads 0.62

    Those two numbers are the same model at the same `alpha` with the same `random_state=17`, so they ought to agree, and they do not. Worth chasing, because the answer is the lesson of this whole notebook rather than a bug.

    `learning_curve`'s largest slice is 2666 rows. `validation_curve` hands each fold its full training split, which is 2666 rows for three folds and **2667** for the other two. One row of difference. Fit both and compare per fold: the three folds that saw identical data return identical scores, and the two folds that got one extra row collapse from ≈ 0.65 / 0.71 to ≈ 0.47 / 0.54 — dragging the mean down by 0.07.

    A single training row does not really carry 0.2 AUC of information. What it does is nudge SGD onto a different path, and at `max_iter=5` with `alpha=10` the model is nowhere near converged, so the path is what decides the answer. This is exactly what the ±0.10 band was warning about: at this alpha neither 0.62 nor 0.69 is a fact about the model, they are two draws from the same noisy process. At `alpha=0.0785` the two plots *do* land on the same number — the curve below ends at 0.8836 against the validation curve's 0.8836 — but that is a fact about that one alpha, not about the sweep. The one-row mechanism never stops running; it only stays invisible where the folds happen to agree. At `alpha=1e-4` the same two folds move again and the learning curve ends ≈ 0.015 below the validation curve's point, which the `alpha=1e-4` section picks up.
    """)
    return


@app.cell
def _(X, np, sgd_pipe, y):
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    def folds_at(alpha, cap):
        """Per-fold validation AUC at `alpha`, training on at most `cap` rows."""
        scores = []
        for train_idx, test_idx in StratifiedKFold(5).split(X, y):
            pipe = clone(sgd_pipe).set_params(sgd_model__alpha=alpha)
            pipe.fit(X[train_idx[:_cap]], y[train_idx[:_cap]])
            scores.append(
                roc_auc_score(y[test_idx], pipe.decision_function(X[test_idx]))
            )
        return np.array(scores)

    for _cap, _label in [
        (2667, "2667 rows (validation_curve)"),
        (2666, "2666 rows (learning_curve)"),
    ]:
        _s = folds_at(10, _cap)
        print(
            f"{_label:30s} folds={np.round(_s, 3)}  mean={_s.mean():.3f}  std={_s.std():.3f}"
        )
    return (folds_at,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Moderate regularization (`alpha=0.0785`): the good model, and already fed

    `alpha=0.0785` is the grid point just left of the validation peak, and it is effectively tied with it (validation ≈ 0.8836 against the peak's ≈ 0.8837 at `alpha≈0.14`). The picture is much healthier: validation climbs steeply from ≈ 0.72 at 133 rows to ≈ 0.88 by the right edge, while the gap collapses from ≈ 0.27 to ≈ 0.02. It lands where the validation curve promised for this alpha — ≈ 0.884 on both plots. That agreement is what the `alpha=10` case could not give us, and the difference is that here the folds actually agree with each other.

    Note what the curve does **not** do: it does not keep climbing. Validation is essentially flat past ~1,900 rows, wobbling around 0.88 for the last third of the axis. So the honest verdict is the unglamorous one — this is the model to ship, and buying more rows would buy approximately nothing. Extra data pays only when the curves are *still far apart and validation is still visibly rising at the right edge*, and none of the three alphas in this notebook show that. This dataset is tapped out around 0.88 AUC.
    """)
    return


@app.cell
def _(plot_learning_curve):
    plot_learning_curve(degree=2, alpha=0.0785)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Barely any regularization (`alpha=1e-4`): just a worse model

    This is the left-hand endpoint of the validation curve, and the `alpha=10` caveat comes along with it in smaller print. The learning curve settles at validation ≈ 0.83 while the validation curve reads ≈ 0.84 at this same alpha — the 2666-vs-2667 mechanism again, with the same two folds rerouted by one extra row. The cell below measures it: 0.829 on 2666 rows against 0.844 on 2667. Read ≈ 0.83 off this plot and ≈ 0.84 off the sweep; the 0.015 between them is that one row, not a second effect. It is small here only because the folds at this alpha disagree less than at `alpha=10` (±0.03 rather than ±0.10).

    Calling this alpha "overfit" would be wrong, and it is worth being precise about why: overfitting means the training score stays good while validation falls well short of it. That is not what happens. Validation settles around 0.83 — clearly below `alpha=0.0785`'s ≈ 0.88 — but the training score sags too, ending near 0.87. The final gap (≈ 0.043) is double `alpha=0.0785`'s ≈ 0.021, which sounds like something until you put it against the ±0.03 fold noise at this alpha: both gaps are inside it, so neither is a gap you can read.

    Both curves dropping together is the *underfitting* signature, which is a strange thing to say about the least-regularized model in the notebook. The culprit is `max_iter=5` again: five SGD passes are not enough to converge a model this unconstrained, so removing the penalty just hands the optimizer a harder problem it has no budget to solve. Left long enough to converge it would eventually memorize and split the curves apart.

    The transferable point stands either way: **more complexity is not monotonically better.** Overshooting is a real risk, not a theoretical one — you just have to name the failure by what the plot shows, not by which end of the axis you are standing on.
    """)
    return


@app.cell
def _(plot_learning_curve):
    plot_learning_curve(degree=2, alpha=1e-4)
    return


@app.cell
def _(folds_at, np):
    for _cap, _label in [
        (2667, "2667 rows (validation_curve)"),
        (2666, "2666 rows (learning_curve)"),
    ]:
        _s = folds_at(0.0001, _cap)
        print(
            f"{_label:30s} folds={np.round(_s, 3)}  mean={_s.mean():.3f}  std={_s.std():.3f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Training error alone means nothing.** Only the gap between train and validation, and the level of the validation curve, carry information.
    - **Validation curve = complexity axis.** Curves together and low → underfitting; curves far apart with train high → overfitting; tune to the validation peak (`alpha≈0.14`, ROC-AUC ≈ 0.884 here).
    - **Learning curve = data axis.** Curves converged and flat → more data is wasted money, change the model. Curves still far apart *and* validation still rising at the right edge → more data pays.
    - **Neither good case here asks for more data.** `alpha=10` plateaus low, `alpha=0.0785` plateaus high; neither trends upward at the right edge. Flat is the common answer in practice.
    - **The two plots split the decision cleanly.** "Do I need a better model or more rows?" is answered by which plot is flat.
    - **Log-spaced sweep → log-scaled axis.** Plot `logspace` points on a linear axis and the shape you are trying to read gets squashed into the left margin.
    - **Sweep the range you will actually use.** Every learning-curve alpha here is one of the 20 points on the validation curve's grid, which is what lets the two plots be read against each other.
    - **Wrap preprocessing in a `Pipeline`.** Scaler and polynomial features must be refit per fold, or CV scores are optimistic through leakage.
    - **Plot the ±1 std band.** A gap inside the fold noise is not a gap — and on this sweep the gap (≈ 0.015–0.03) never escapes the noise, so the whole story is height.
    - **Matching alphas does not guarantee matching scores.** At `alpha=10` the learning curve ends at ≈ 0.69 while the validation curve reads ≈ 0.62, because `learning_curve` caps at 2666 rows and one extra row flips two folds. The same row moves `alpha=1e-4` by ≈ 0.015 and `alpha=0.0785` not at all. Where the bands are wide, the number you get is a draw, not a measurement.
    - **Name the diagnosis from the plot, not from the axis.** At `alpha=1e-4` both curves sag together — that reads as underfitting even though it is the *least* regularized model, because `max_iter=5` stops SGD before it can memorize.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    With cross-validation and regularization in hand, the practical skill is diagnosing *why* a model underperforms, and validation/learning curves are the two-plot answer. The validation curve sweeps complexity (`alpha` from `1e-4` to `10` on an SGD-trained linear SVM over degree-2 polynomial features, `max_iter=5`, ROC-AUC over 5 folds): scores rise to a peak of ≈ 0.884 near `alpha≈0.14` and collapse to ≈ 0.62 at `alpha=10`, where both curves sit low and together — textbook underfitting. The complex end is less textbook: the train/validation gap never exceeds ≈ 0.03 anywhere on the sweep (≈ 0.015–0.03) and stays inside the fold-to-fold noise throughout, because `max_iter=5` stops SGD before it can memorize, so small alpha degrades both curves rather than splitting them apart. The learning curves fix alpha and sweep training-set size: at `alpha=10` the curves converge quickly to a mediocre, visibly noisy band that wanders 0.56–0.72 and ends near 0.69 (capacity-bound — extra rows change nothing), at `alpha=0.0785` validation climbs to ≈ 0.88 and flattens past ~1,900 rows (good model, already fed), and at `alpha=1e-4` both curves settle lower, ≈ 0.87 train against ≈ 0.83 validation — a worse model, not an overfit one (its ≈ 0.83 sits ≈ 0.015 under the validation curve's ≈ 0.84 at that alpha, the same one-row fold shift in milder form). The `alpha=10` endpoint (≈ 0.69) does not match its validation-curve point (≈ 0.62), and chasing that down is instructive rather than alarming: `learning_curve` caps training at 2666 rows while two folds of `validation_curve` get 2667, and at this alpha a single row reroutes unconverged SGD enough to swing those folds by 0.2 AUC — precisely the instability the ±0.10 band was reporting. Everything runs inside a `Pipeline` so scaling and feature expansion refit per fold and the scores stay honest.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **validation curve** = score on train and validation vs. a model-complexity hyperparameter
    - **learning curve** = score on train and validation vs. training-set size
    - **underfitting** = train and validation scores both poor and close together; too little usable capacity
    - **overfitting** = train score good but validation much worse; the model fit noise
    - **regularization strength (`alpha`)** = L2 penalty weight in `SGDClassifier`; larger = simpler model
    - **`SGDClassifier`** = linear model trained by stochastic gradient descent
        - **hinge loss** = the `loss="hinge"` setting, making it a linear SVM
        - **`max_iter`** = cap on passes over the training data, pinned to 5 here
    - **`Pipeline`** = chained preprocessing + estimator refit as a unit inside each CV fold
        - **`step__param`** = the naming syntax for addressing a pipeline step's hyperparameter, e.g. `sgd_model__alpha`
    - **data leakage** = fitting preprocessing on data the fold is meant to be tested on, inflating CV scores
    - **`PolynomialFeatures`** = expansion adding powers and interaction terms, here degree 2
    - **`StandardScaler`** = zero-mean/unit-variance rescaling of each feature
    - **ROC-AUC** = ranking-quality metric used here as the score
    - **plateau** = flat region of a learning curve; the point where extra data stops paying
    - **capacity-bound** = performance limited by the model itself, so more rows do not help
    - **`np.logspace`** = evenly spaced points on a log scale; pair with `plt.xscale("log")`
    """)
    return


if __name__ == "__main__":
    app.run()
