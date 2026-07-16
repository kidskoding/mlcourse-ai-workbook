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
    # Topic 4 — Part 4: When Logistic Regression Is Good and When It Is Not

    Two demos that bracket the honest answer to "should I use a linear model?". First IMDB sentiment classification, where logistic regression on raw word counts beats a random forest with a fraction of the compute. Then XOR, a 200-point toy set where a line simply cannot work. The rule that falls out: linear models win when the feature space is huge, sparse, and the signal is roughly additive; they lose when the true boundary needs interaction between features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Everything below runs off these imports plus the IMDB dataset (25k labeled train reviews, 25k test, balanced 50/50 pos/neg). The tarball is ~80MB and is not mirrored in the mlcourse.ai data repo, so the loader downloads and extracts it into `data/` at the root of this workbook.
    """)
    return


@app.cell
def _():
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'retina'

    from sklearn.datasets import load_files
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    return CountVectorizer, LogisticRegression, load_files, np, os, plt


@app.cell
def _(os):
    import tarfile
    from io import BytesIO

    import requests

    URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


    def load_imdb_dataset(extract_path, overwrite=False):
        if os.path.isfile(os.path.join(extract_path, "aclImdb", "README")) and not overwrite:
            print("IMDB dataset is already in place.")
            return
        print("Downloading the dataset from: ", URL)
        response = requests.get(URL)
        tar = tarfile.open(mode="r:gz", fileobj=BytesIO(response.content))
        tar.extractall(extract_path)


    # Local to this workbook: the ~80MB tarball is not in the mlcourse.ai data repo,
    # so it lands in <repo root>/data/ rather than anywhere upstream. ~80MB, don't commit it.
    DATA_PATH = "../data/"
    os.makedirs(DATA_PATH, exist_ok=True)
    load_imdb_dataset(extract_path=DATA_PATH)
    return (DATA_PATH,)


@app.cell
def _(DATA_PATH, load_files, np, os):
    PATH_TO_IMDB = DATA_PATH + "aclImdb"

    reviews_train = load_files(os.path.join(PATH_TO_IMDB, "train"), categories=["pos", "neg"])
    text_train, y_train = reviews_train.data, reviews_train.target

    reviews_test = load_files(os.path.join(PATH_TO_IMDB, "test"), categories=["pos", "neg"])
    text_test, y_test = reviews_test.data, reviews_test.target

    print(len(text_train), np.bincount(y_train))  # 25000, [12500 12500]
    print(len(text_test), np.bincount(y_test))
    print(text_train[1][:120], y_train[1])        # a bad review, label 0
    return text_test, text_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bag of words: turning text into a design matrix

    There is no feature matrix $X$ handed to you here — only raw strings. The **bag of words** model builds one: collect every distinct token across the whole **corpus** (all reviews) into a vocabulary, then represent each review as a vector counting how many times each vocabulary word appears in it. Word order is thrown away entirely; "not good" and "good not" are identical. That sounds fatal but for sentiment it mostly isn't, because individual words like *terrible* and *awful* carry the signal.

    `CountVectorizer` does the whole thing. Fitting it on 25k reviews yields ~75k vocabulary entries.
    """)
    return


@app.cell
def _(CountVectorizer, text_train):
    cv = CountVectorizer()
    cv.fit(text_train)

    len(cv.vocabulary_)  # ~74849
    return (cv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Peek at the tokens and the sloppiness is visible immediately — `'0000000000001'`, `'00am'`, `'003830'`. A naive tokenizer keeps every numeric junk string, every misspelling, every inflection (`pinch`/`pinches`/`pinching` are three separate features). Real text pipelines add lowercasing, stemming/lemmatization, stopword removal, and min/max document-frequency cutoffs. We skip all of it here and the model still works — which is itself the point about how forgiving a big sparse linear model is.
    """)
    return


@app.cell
def _(cv):
    print(cv.get_feature_names_out()[:20])
    print(cv.get_feature_names_out()[50000:50020])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `transform` encodes the texts into a **sparse matrix**: 25000 x 74849, but only ~3.4M cells are non-zero (~0.2% density). Storing this dense would be gigabytes of mostly zeros; CSR format stores only the non-zeros. This is the shape of problem linear models were made for.

    Note the test set is transformed with the *same fitted* vectorizer — never refit on test, or the vocabulary leaks.
    """)
    return


@app.cell
def _(cv, text_test, text_train):
    X_train = cv.transform(text_train)
    X_test = cv.transform(text_test)

    X_train  # <25000x74849 sparse matrix, ~3.4M stored elements>
    return X_test, X_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sanity check on one short review: the row has exactly as many non-zero columns as it has distinct words, and each index points back into the vocabulary.
    """)
    return


@app.cell
def _(X_train, text_train):
    print(text_train[19726])  # b'This movie is terrible but it has some good effects.'
    print(X_train[19726].nonzero()[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logistic regression on the counts

    Fit it straight on the sparse matrix — a few seconds for 75k features.
    """)
    return


@app.cell
def _(LogisticRegression, X_train, y_train):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    logit = LogisticRegression(solver="lbfgs", n_jobs=-1, random_state=7, max_iter=500)
    logit.fit(X_train, y_train)
    return (logit,)


@app.cell
def _(X_test, X_train, logit, y_test, y_train):
    round(logit.score(X_train, y_train), 3), round(logit.score(X_test, y_test), 3)
    # ~(0.998, 0.867)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    99.8% train vs 86.7% test — a textbook **overfit** gap. With 75k features and 25k samples the model has enough capacity to memorize idiosyncratic words, so the default regularization ($C=1$) is far too weak. That gap is the thing to fix next.

    First, though, the payoff of a linear model on text: the coefficients *are* the explanation. One weight per word, sign = direction of sentiment, magnitude = strength. No feature-importance heuristics needed.
    """)
    return


@app.cell
def _(cv, logit, np, plt):
    def visualize_coefficients(classifier, feature_names, n_top_features=25):
        coef = classifier.coef_.ravel()
        positive_coefficients = np.argsort(coef)[-n_top_features:]
        negative_coefficients = np.argsort(coef)[:n_top_features]
        interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])

        plt.figure(figsize=(15, 5))
        colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
        plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(
            np.arange(1, 1 + 2 * n_top_features),
            feature_names[interesting_coefficients],
            rotation=60,
            ha="right",
        )


    visualize_coefficients(logit, cv.get_feature_names_out());
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tuning C with a pipeline (and why the pipeline matters)

    To tune the regularization strength $C$ we need cross-validation, and CV is exactly where vectorizing-then-splitting goes wrong: if `CountVectorizer` is fit on all the data before splitting, each validation fold's word statistics have already been seen by the transformer. That is **data leakage**, and it inflates the CV score.

    `make_pipeline` fixes it by construction — the pipeline is a single estimator, so `GridSearchCV` refits `CountVectorizer` inside every fold on that fold's training part only. Rule of thumb: any transformer that *learns something from the data* (vocabularies, means, scalers) belongs inside a pipeline, not in a preprocessing cell above the split.
    """)
    return


@app.cell
def _(
    CountVectorizer,
    LogisticRegression,
    text_test,
    text_train,
    y_test,
    y_train,
):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    from sklearn.pipeline import make_pipeline

    text_pipe_logit = make_pipeline(
        CountVectorizer(),
        # n_jobs=1 here: it conflicts with GridSearchCV's own n_jobs=-1 below
        LogisticRegression(solver="lbfgs", n_jobs=1, random_state=7, max_iter=500),
    )

    text_pipe_logit.fit(text_train, y_train)
    print(text_pipe_logit.score(text_test, y_test))  # ~0.867, matches the manual fit
    return (text_pipe_logit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Search $C$ over a log scale — regularization strength is multiplicative, so `np.logspace` is the right grid, never `np.linspace`. Note `C` is *inverse* regularization: small $C$ = strong penalty.
    """)
    return


@app.cell
def _(np, text_pipe_logit, text_train, y_train):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    from sklearn.model_selection import GridSearchCV

    param_grid_logit = {"logisticregression__C": np.logspace(-5, 0, 6)}
    grid_logit = GridSearchCV(
        text_pipe_logit, param_grid_logit, return_train_score=True, cv=3, n_jobs=-1
    )
    grid_logit.fit(text_train, y_train)

    grid_logit.best_params_, grid_logit.best_score_  # C=0.1, CV ~0.885
    return (grid_logit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plotting train vs validation score across the grid is the standard **validation curve** read: at tiny $C$ both curves sit low (underfit — the penalty crushes all the weights); as $C$ grows the train curve marches toward 1.0 while validation peaks at $C=0.1$. The peak is the bias–variance sweet spot.
    """)
    return


@app.cell
def _(grid_logit, plt):
    def plot_grid_scores(grid, param_name):
        plt.plot(
            grid.param_grid[param_name],
            grid.cv_results_["mean_train_score"],
            color="green",
            label="train",
        )
        plt.plot(
            grid.param_grid[param_name],
            grid.cv_results_["mean_test_score"],
            color="red",
            label="test",
        )
        plt.legend();


    plot_grid_scores(grid_logit, "logisticregression__C")
    return


@app.cell
def _(grid_logit, text_test, y_test):
    grid_logit.score(text_test, y_test)  # ~0.879, up from 0.867
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tuning $C$ bought ~1.2 points of test accuracy for one line of grid. Cheap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The random forest comparison

    Same features, a 200-tree forest, and it loses — ~0.855 vs the tuned logit's ~0.879, while burning substantially more CPU time to fit. This is not a fluke of this dataset. Trees split on one feature at a time, so on a 75k-column sparse matrix where each individual word is weakly informative and the evidence needs to be *summed* across hundreds of words, a tree has to burn depth reconstructing what a linear model gets for free in a single dot product. Meanwhile most randomly-sampled feature subsets at a split contain nothing but zeros.

    **Rule of thumb: high-dimensional sparse data (text, one-hot IDs, click logs) → reach for the linear model first.**
    """)
    return


@app.cell
def _(X_test, X_train, y_test, y_train):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=17)
    forest.fit(X_train, y_train)
    round(forest.score(X_test, y_test), 3)  # ~0.855
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The XOR problem: where linear models actually break

    Now the other side. A linear classifier can only ever draw a **hyperplane** (a line in 2D). **XOR** — exclusive OR, true when exactly one input is true — is the canonical function no line can express. Geometrically it's two diagonally-opposed pairs of point clouds: one class in the upper-right and lower-left quadrants, the other class in the remaining two.

    The generative rule below is literally `(x1 > 0) XOR (x2 > 0)`. Neither $x_1$ nor $x_2$ alone tells you anything about the label — the signal lives entirely in their **interaction**, and a linear model has no term for that.
    """)
    return


@app.cell
def _(np, plt):
    rng = np.random.RandomState(0)
    X = rng.randn(200, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

    plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired);
    return X, y


@app.cell
def _(LogisticRegression, X, np, plt, y):
    def plot_boundary(clf, X, y, plot_title):
        xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
        clf.fit(X, y)
        Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
        Z = Z.reshape(xx.shape)

        image = plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
            cmap=plt.cm.PuOr_r,
        )
        plt.contour(xx, yy, Z, levels=[0], linewidths=2)
        plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)
        plt.xticks(())
        plt.yticks(())
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.axis([-3, 3, -3, 3])
        plt.colorbar(image)
        plt.title(plot_title, fontsize=12);


    plot_boundary(
        LogisticRegression(solver="lbfgs", max_iter=500), X, y, "Logistic Regression, XOR problem"
    )
    return (plot_boundary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The probability surface comes out essentially flat — the model finds no line better than a coin flip, which is the correct answer to an impossible question.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fix: give it the interaction term

    The model isn't the problem, the **feature space** is. `PolynomialFeatures(degree=2)` maps $(x_1, x_2)$ into the 6-dimensional space

    $$1,\; x_1,\; x_2,\; x_1^2,\; x_1 x_2,\; x_2^2$$

    and that $x_1 x_2$ term is exactly the missing piece — it's positive in the two quadrants of one class and negative in the two of the other, so a single sign test on it solves XOR. Logistic regression still fits a plain hyperplane, just in 6D; **projected back down to $(x_1, x_2)$ that hyperplane appears as a nonlinear boundary.**

    The general lesson: "linear model" constrains the boundary only relative to *the features you hand it*. Feature engineering is how you buy nonlinearity without leaving linear-model land.
    """)
    return


@app.cell
def _(LogisticRegression, X, plot_boundary, y):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures

    logit_pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2)),
            ("logit", LogisticRegression(solver="lbfgs", max_iter=500)),
        ]
    )

    plot_boundary(logit_pipe, X, y, "Logistic Regression + quadratic features. XOR problem")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The catch: explicit polynomial expansion is combinatorial. Degree $d$ on $n$ features costs roughly $\binom{n+d}{d}$ columns — fine for $n=2$, hopeless for $n=75{,}000$. This is precisely the gap the **kernel trick** (SVM) fills: it computes only the inner products between objects in the high-dimensional space via a kernel function, never materializing the features at all.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Bag of words is a legitimate first move on text.** Vocabulary of every token, one count per column, word order discarded. Crude tokenization and all, it gets ~87% on IMDB sentiment.
    - **Sparse + high-dimensional = the linear model's home turf.** 75k features, ~0.2% dense, and logistic regression beats a 200-tree random forest (~0.879 vs ~0.855) in a fraction of the time. Trees split one feature at a time and can't cheaply sum weak evidence across thousands of columns.
    - **Linear models on text are self-explaining.** One coefficient per word — sign and magnitude read directly as sentiment direction and strength. No post-hoc importance machinery.
    - **Default `C=1` overfits wide sparse data.** 0.998 train vs 0.867 test is the tell. Tuning $C$ to 0.1 recovered ~1.2 points of test accuracy.
    - **Put learned transformers inside a Pipeline.** Fitting `CountVectorizer` before the CV split leaks fold vocabulary and inflates scores. The pipeline refits it per fold automatically.
    - **Search regularization on a log scale.** `np.logspace(-5, 0, 6)`, never `linspace` — $C$ acts multiplicatively.
    - **XOR is the canonical linear failure.** When the label depends on an *interaction* and neither raw feature is informative alone, no useful hyperplane exists and the model flatlines.
    - **"Linear" is relative to your features.** Add $x_1 x_2$ via `PolynomialFeatures` and the same logistic regression solves XOR — a hyperplane in 6D projects down as a curve in 2D.
    - **Explicit polynomials don't scale; kernels do.** Degree-$d$ expansion is combinatorial in feature count. The kernel trick gets the same high-dimensional geometry via pairwise kernel values only.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    This part answers "when is logistic regression the right tool?" with one win and one loss. The win is IMDB sentiment: bag-of-words via `CountVectorizer` produces a 25000 x 74849 sparse matrix, logistic regression fits it in seconds, and after tuning $C$ inside a `Pipeline` + `GridSearchCV` (log-scale grid, best $C = 0.1$) it reaches ~0.879 test accuracy — beating a 200-tree random forest on identical features while being both faster and directly interpretable through its per-word coefficients. The loss is XOR: 200 points where the class depends purely on the interaction of two features, so the only hyperplane available is a coin flip and the probability surface flatlines around $0.5$. Feeding it `PolynomialFeatures(degree=2)` adds the $x_1 x_2$ term and the same estimator solves the problem cleanly — the boundary is still a hyperplane, just in 6D, and appears curved once projected back. Together: linear models excel on wide sparse additive signal and fail on unrepresented interactions, and the fix for the failure is usually the feature space, not the estimator — though explicit expansion is combinatorial, which is why kernels exist.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **bag of words** = text → vector of per-word counts, word order discarded
        - **token** = one vocabulary entry produced by the tokenizer (not always a real word)
    - **corpus** = the full set of documents the vocabulary is built from
    - **`CountVectorizer`** = sklearn transformer that fits a vocabulary and emits count vectors
    - **sparse matrix** = matrix stored as only its non-zero entries (CSR here)
    - **overfitting** = high train score, low test score; model memorized noise
    - **`C`** = logistic regression's inverse regularization strength; small `C` = strong penalty
    - **`GridSearchCV`** = exhaustive cross-validated search over a hyperparameter grid
    - **validation curve** = train vs validation score plotted across one hyperparameter
    - **`Pipeline`** = chained transformers + estimator treated as a single estimator
        - **`make_pipeline`** = shorthand constructor that auto-names the steps
    - **data leakage** = fitting on information the model shouldn't have yet (e.g. vectorizing before the CV split)
    - **`np.logspace`** = log-scaled grid; the correct spacing for regularization parameters
    - **hyperplane** = the flat decision surface a linear model is limited to
    - **XOR** = exclusive OR; the classic non-linearly-separable toy problem
    - **interaction** = signal carried by a combination of features rather than any one alone
    - **`PolynomialFeatures`** = transformer adding powers and cross-products of the inputs
    - **kernel trick** = obtaining high-dimensional geometry via pairwise kernel values without building the features
    """)
    return


if __name__ == "__main__":
    app.run()
