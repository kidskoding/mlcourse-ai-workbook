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
    # Topic 3: Classification, Decision Trees and k Nearest Neighbors

    Two of the most interpretable classifiers in ML, plus the machinery for tuning any model: hold-out sets, cross-validation, and grid search. Both algorithms are cheap to fit and easy to explain, which makes them the right *first* thing to try on a new dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where this sits: supervised learning

    Mitchell's definition: a program learns from **experience E** at **tasks T** measured by **performance P** if P improves with E. Unpacking it for our case:

    - **T** = the task. Here: **classification** (predict a category) and **regression** (predict a number). Others exist — clustering, anomaly detection.
    - **E** = the data. **Supervised learning** means each instance carries a **target** (the known answer); the model learns the feature → target mapping. Without targets you're in unsupervised land (Topic 7).
    - **P** = the metric. For now, **accuracy** = fraction of correct predictions on held-out data.

    Same client table, two different tasks: "did they default? (1/0)" is classification; "how many days late were they?" is regression. The target's type picks the task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision trees

    A decision tree is a flowchart: each internal node asks a yes/no question about one feature, each leaf hands back an answer. Any bank's hand-written credit-scoring rulebook *is* a decision tree — the ML version just learns the rules from data instead of from an expert's intuition.

    The whole game is **which question to ask first**. Think "20 Questions": you don't open with "Is it Angelina Jolie?" — a "no" barely narrows anything. You open with "Is it a woman?", which halves the field. A good split is one that removes the most **uncertainty**. To turn that instinct into an algorithm we need to measure uncertainty.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Entropy and information gain

    **Shannon entropy** over $N$ states with probabilities $p_i$:

    $$S = -\sum_{i=1}^{N} p_i \log_2 p_i$$

    Read it as degree of chaos: max entropy = a 50/50 coin flip (S=1 bit for two classes), zero entropy = you already know the answer. A tree's job is to drive entropy in its leaves to zero.

    **Information gain** of a split $Q$ that cuts $N$ objects into $q$ groups of size $N_i$:

    $$IG(Q) = S_0 - \sum_{i=1}^{q} \frac{N_i}{N} S_i$$

    i.e. entropy before, minus the *weighted average* entropy after. The weighting matters — a split that yields one pure leaf of 2 objects and one messy leaf of 198 is not a good split.

    Toy case: 9 blue + 11 yellow balls gives $S_0 \approx 1$. Splitting at $x \le 12$ gives a left group of 13 (8 blue, 5 yellow, $S_1 \approx 0.96$) and a right group of 7 (1 blue, 6 yellow, $S_2 \approx 0.6$). Both groups are less chaotic than the parent, so the split gained information — the right group especially.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The tree-building algorithm

    ID3, C4.5, CART all share one greedy recipe: **at every node, pick the split with the highest information gain, then recurse.** Stop when entropy hits zero or a stopping rule fires.

    Greedy = locally optimal at each step, with no guarantee of a globally optimal tree. That's a real limitation, not a footnote (finding the optimal tree is NP-complete), but it works well enough in practice and it's fast.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The shape of every tree learner, as pseudocode:

    ```
    build(L):
        create node t
        if the stopping criterion is True:
            assign a predictive model to t      # e.g. majority class
        else:
            find the best binary split  L = L_left + L_right
            t.left  = build(L_left)
            t.right = build(L_right)
        return t
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Other split criteria

    Entropy is one heuristic among several. For classification:

    - **Gini uncertainty (impurity)**: $G = 1 - \sum_k p_k^2$ — chance that two objects drawn from the node disagree on class. sklearn's default.
    - **Misclassification error**: $E = 1 - \max_k p_k$ — almost never used in practice; it's too flat to discriminate between decent splits.

    For binary classification these become $S = -p_+\log_2 p_+ - (1-p_+)\log_2(1-p_+)$ and $G = 2p_+(1-p_+)$. Plot them and the point lands: **entropy ≈ 2×Gini**. Same shape, same peak at $p_+ = 0.5$, same zeros at the pure ends. Rule of thumb: don't agonize over `criterion="gini"` vs `"entropy"` — they pick nearly the same trees. Spend the tuning budget on `max_depth`.
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt

    sns.set()
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'retina'
    return np, pd, plt


@app.cell
def _(np, plt):
    plt.figure(figsize=(6, 4))
    _xx = np.linspace(0, 1, 50)
    plt.plot(_xx, [2 * x * (1 - x) for x in _xx], label="gini")
    plt.plot(_xx, [4 * x * (1 - x) for x in _xx], label="2*gini")
    plt.plot(
        _xx, [-x * np.log2(x) - (1 - x) * np.log2(1 - x) for x in _xx], label="entropy"
    )
    plt.plot(_xx, [1 - max(x, 1 - x) for x in _xx], label="missclass")
    plt.plot(_xx, [2 - 2 * max(x, 1 - x) for x in _xx], label="2*missclass")
    plt.xlabel("p+")
    plt.ylabel("criterion")
    plt.title("Criteria of quality as a function of p+ (binary classification)")
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seeing the boundary: synthetic two-class data

    Two Gaussian blobs with different means. The classification problem *is* the problem of drawing a good boundary between them: a straight line is probably too simple, a curve that snakes around every single red dot is too complex and will fail on new data. Somewhere in between is the sweet spot — this tension is the whole of Topics 3–5.
    """)
    return


@app.cell
def _(np, plt):
    np.random.seed(17)
    train_data = np.random.normal(size=(100, 2))
    train_labels = np.zeros(100)

    # second class: same spread, shifted mean
    train_data = np.r_[train_data, np.random.normal(size=(100, 2), loc=2)]
    train_labels = np.r_[train_labels, np.ones(100)]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        train_data[:, 0],
        train_data[:, 1],
        c=train_labels,
        s=100,
        cmap="autumn",
        edgecolors="black",
        linewidth=1.5,
    )
    plt.plot(range(-2, 5), range(4, -3, -1))  # a candidate linear boundary
    return train_data, train_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now fit a depth-3 tree and paint what it predicts everywhere on the plane. The result is the signature of a decision tree: the boundary is **axis-parallel rectangles**, never a diagonal. Depth 3 → up to $2^3 = 8$ leaves → up to 8 rectangles, each predicting the majority class of the training points inside it.

    `plot_tree` renders the fitted tree itself. Reading it: the root holds all 200 samples, 100/100, so $S = 1$ — maximal chaos, and the node is drawn white. Each split drops entropy in both children; color darkens toward whichever class dominates. A path from root to leaf is a human-readable rule.
    """)
    return


@app.cell
def _(np, plt, train_data, train_labels):
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    def get_grid(data):
        """Dense grid over the data's bounding box, for painting decision regions."""
        x_min, x_max = (_data[:, 0].min() - 1, _data[:, 0].max() + 1)
        y_min, y_max = (_data[:, 1].min() - 1, _data[:, 1].max() + 1)
        return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=17)
    clf_tree.fit(train_data, train_labels)
    _xx, _yy = get_grid(train_data)
    _predicted = clf_tree.predict(np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
    plt.pcolormesh(_xx, _yy, _predicted, cmap="autumn")
    plt.scatter(
        train_data[:, 0],
        train_data[:, 1],
        c=train_labels,
        s=100,
        cmap="autumn",
        edgecolors="black",
        linewidth=1.5,
    )
    return DecisionTreeClassifier, clf_tree, get_grid, plot_tree


@app.cell
def _(clf_tree, plot_tree, plt):
    plt.figure(figsize=(16, 8))
    plot_tree(clf_tree, feature_names=["x1", "x2"], filled=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How trees handle numeric features

    A numeric feature like `Age` has a huge number of possible thresholds ("Age < 17", "Age < 22.87", …). Checking all of them at every node would be ruinous. The heuristic: **sort the feature's values and only consider thresholds where the target actually switches.** Everywhere else, a split can't separate anything it couldn't separate at the neighbouring value.

    Fit an unconstrained tree on Age alone and the chosen thresholds come out as the midpoints of exactly those switch points — e.g. 43.5 sits between the 38-year-old who defaulted and the 49-year-old who didn't.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, pd, plot_tree, plt):
    _data = pd.DataFrame(
        {
            "Age": [17, 64, 18, 20, 38, 49, 55, 25, 29, 31, 33],
            "Loan Default": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        }
    )
    print(_data.sort_values("Age"))
    age_tree = DecisionTreeClassifier(random_state=17)
    age_tree.fit(_data["Age"].values.reshape(-1, 1), _data["Loan Default"].values)
    plt.figure(figsize=(14, 7))
    plot_tree(age_tree, feature_names=["Age"], filled=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add `Salary` and the tree now has two features to pick thresholds from. Sorted by age the target flips 5 times; sorted by salary, 7 times. The tree splits on whichever candidate threshold gives the best Gini gain — and again lands on midpoints between switching neighbours (95 sits between 88 and 102, 30.5 between the pair around it).

    **Conclusion:** sort, only test the switch points, and when there are many numeric features, keep only the top-N thresholds by gain (fit a depth-1 stump, score each candidate, keep the winners). That's how a tree stays cheap on wide numeric data.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, pd, plot_tree, plt):
    data2 = pd.DataFrame(
        {
            "Age": [17, 64, 18, 20, 38, 49, 55, 25, 29, 31, 33],
            "Salary": [25, 80, 22, 36, 37, 59, 74, 70, 33, 102, 88],
            "Loan Default": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        }
    )

    age_sal_tree = DecisionTreeClassifier(random_state=17)
    age_sal_tree.fit(data2[["Age", "Salary"]].values, data2["Loan Default"].values)

    plt.figure(figsize=(16, 8))
    plot_tree(age_sal_tree, feature_names=["Age", "Salary"], filled=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overfitting and the parameters that stop it

    You *can* grow a tree until every leaf holds one instance. Don't. Down at the bottom the tree starts splitting on noise — "all four clients who wore green trousers defaulted" is true of the training set and worthless everywhere else. That's **overfitting**: fitting the training data so tightly that you learn its accidents instead of its structure.

    Two legitimate exceptions to "don't grow it fully":
    - **Random forest** grows many full-depth trees and averages them (Topic 5) — the averaging cancels the individual overfitting.
    - **Pruning** grows the full tree, then walks bottom-up removing nodes that don't pay for themselves under cross-validation.

    The knobs on `sklearn.tree.DecisionTreeClassifier`:

    - **`max_depth`** — hard cap on depth. The first thing to tune, and usually the one that matters.
    - **`max_features`** — how many features to consider per split. Necessary when the feature count is large (searching all of them at every node is expensive), and it injects useful randomness.
    - **`min_samples_leaf`** — floor on samples per leaf. Directly kills the "4 clients in green trousers" leaves.

    Set these by cross-validation, not by taste.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trees for regression

    Same greedy construction, different criterion. Instead of entropy, minimize **variance** within the leaf:

    $$D = \frac{1}{\ell}\sum_{i=1}^{\ell}\left(y_i - \frac{1}{\ell}\sum_{j=1}^{\ell} y_j\right)^2$$

    That is: find splits after which the target values inside each leaf are all roughly equal, then predict the leaf's mean.

    The consequence is visible below — a regression tree approximates a smooth function with a **piecewise-constant staircase**. It can only ever emit one value per leaf. This is also why trees **cannot extrapolate**: outside the training range there are no new leaves, so predictions flatline at the boundary value.
    """)
    return


@app.cell
def _(np, plt):
    from sklearn.tree import DecisionTreeRegressor

    n_train, n_test, noise = (150, 1000, 0.1)

    def _f(x):
        x = x.ravel()
        return np.exp(-(x**2)) + 1.5 * np.exp(-((x - 2) ** 2))

    def generate(n_samples, noise):
        _X = np.random.rand(n_samples) * 10 - 5
        _X = np.sort(_X).ravel()
        _y = (
            np.exp(-(_X**2))
            + 1.5 * np.exp(-((_X - 2) ** 2))
            + np.random.normal(0.0, noise, n_samples)
        )
        return (_X.reshape((n_samples, 1)), _y)

    X_train, y_train = generate(n_samples=n_train, noise=noise)
    X_test, y_test = generate(n_samples=n_test, noise=noise)
    reg_tree = DecisionTreeRegressor(max_depth=5, random_state=17)
    reg_tree.fit(X_train, y_train)
    reg_tree_pred = reg_tree.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, _f(X_test), "b")
    plt.scatter(X_train, y_train, c="b", s=20)
    plt.plot(X_test, reg_tree_pred, "g", lw=2)
    plt.xlim([-5, 5])
    plt.title(
        "Decision tree regressor, MSE = %.2f"
        % (np.sum((y_test - reg_tree_pred) ** 2) / n_test)
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## k Nearest Neighbors

    k-NN's premise, the **compactness hypothesis**: *if your distance measure is any good, similar objects share a class.* You look like your neighbours.

    To classify one test point:
    1. Compute its distance to every training point.
    2. Keep the **k** closest.
    3. Predict the majority class among those k. (For regression: return their mean.)

    Note what's missing — there is no training step. k-NN is a **lazy learner**: it memorizes the training set and defers all work to prediction time. Fitting is free; predicting is expensive.

    Three things decide whether it works:

    - **k** — small k overfits and chases outliers; large k oversmooths and blurs interpretability.
    - **the distance metric** — Euclidean, Manhattan, cosine, Minkowski, Hamming. Euclidean-by-default is often an unexamined choice.
    - **neighbour weights** — `uniform`, or `distance` (closer neighbours count more).

    **The trap: scale your features.** Almost every metric is scale-sensitive, so a `salary` in the tens of thousands will drown out an `age` under 100 — the model silently becomes "nearest by salary". Always put a `StandardScaler` in front of k-NN.

    Key `KNeighborsClassifier` parameters: `n_neighbors`, `weights`, `metric`, and `algorithm` (`brute` = full scan; `ball_tree`/`kd_tree` = precomputed structures that speed up the search; `auto` = let sklearn choose). In practice on big data people reach for approximate neighbour search (e.g. Spotify's Annoy). k-NN is also the standard trick for building meta-features in Kaggle stacking, and it's the seed of most recommender systems.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing parameters: hold-out and cross-validation

    A model's only job is to **generalize** to unseen data. You can't test on data you don't have, so sacrifice some of what you do have:

    - **Hold-out set** — carve off 20–40%, train on the rest, score once on the held-out part. Cheap, but the estimate rides on one arbitrary split.
    - **k-fold cross-validation** — chop the training data into K folds; K times, train on K−1 folds and score on the remaining one; average the K scores. A better estimate of new-data performance, at K× the compute.

    The discipline that makes it all work: **the hold-out set never participates in tuning.** Tune with cross-validation on the training portion; touch the hold-out once, at the end, to get an honest number. Peek at it repeatedly and you've just overfit to it too.

    **`GridSearchCV`** automates the loop: for every combination in the parameter grid, run K-fold CV, then keep the best combo (`best_params_`, `best_score_`) and refit on all training data (`best_estimator_`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Application: churn prediction

    The Topic 1 telecom dataset. Encode the binary plan columns as integers with `factorize`, put `State` aside (train without it first, see later whether it helps), and split 70/30 into train and hold-out.

    Both models get deliberately arbitrary parameters at first — the point is to see what untuned models give, then to see how much tuning buys. Note the asymmetry in setup: the tree eats raw features happily, while k-NN needs `StandardScaler` fitted **on the training data only** and then applied to the hold-out. Fitting the scaler on everything leaks hold-out information into training.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, pd):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    df = pd.read_csv(DATA_PATH + "telecom_churn.csv")
    df["International plan"] = pd.factorize(df["International plan"])[0]
    df["Voice mail plan"] = pd.factorize(df["Voice mail plan"])[0]
    df["Churn"] = df["Churn"].astype("int")
    states = df["State"]
    _y = df["Churn"]
    df.drop(["State", "Churn"], axis=1, inplace=True)
    X_train_1, X_holdout, y_train_1, y_holdout = train_test_split(
        df.values, _y, test_size=0.3, random_state=17
    )
    tree = DecisionTreeClassifier(max_depth=5, random_state=17)
    knn = KNeighborsClassifier(n_neighbors=10)
    tree.fit(X_train_1, y_train_1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_1)
    X_holdout_scaled = scaler.transform(X_holdout)
    knn.fit(X_train_scaled, y_train_1)
    print(accuracy_score(y_holdout, tree.predict(X_holdout)))
    print(accuracy_score(y_holdout, knn.predict(X_holdout_scaled)))
    return (
        GridSearchCV,
        KNeighborsClassifier,
        StandardScaler,
        X_holdout,
        X_train_1,
        accuracy_score,
        cross_val_score,
        train_test_split,
        tree,
        y_holdout,
        y_train_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Untuned: tree ~94%, k-NN ~89%. Now tune the tree over `max_depth` × `max_features` with 5-fold CV — 150 combinations, 750 fits.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    X_holdout,
    X_train_1,
    accuracy_score,
    tree,
    y_holdout,
    y_train_1,
):
    _tree_params = {"max_depth": range(1, 11), "max_features": range(4, 19)}
    _tree_grid = GridSearchCV(tree, _tree_params, cv=5, n_jobs=-1, verbose=True)
    _tree_grid.fit(X_train_1, y_train_1)
    print(_tree_grid.best_params_)
    print(_tree_grid.best_score_)
    print(accuracy_score(y_holdout, _tree_grid.predict(X_holdout)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For k-NN, wrap the scaler and the model in a **`Pipeline`**. This is not cosmetic: inside cross-validation the scaler must be refit on each training fold. Scale first and *then* cross-validate and every fold's scaler has already seen its own validation fold — leakage, and an optimistic score. A pipeline is one estimator, so `GridSearchCV` refits the whole chain per fold. Grid keys use the `step__param` form (`knn__n_neighbors`).
    """)
    return


@app.cell
def _(
    GridSearchCV,
    KNeighborsClassifier,
    StandardScaler,
    X_holdout,
    X_train_1,
    accuracy_score,
    y_holdout,
    y_train_1,
):
    from sklearn.pipeline import Pipeline

    _knn_pipe = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_jobs=-1))]
    )
    knn_params = {"knn__n_neighbors": range(1, 10)}
    knn_grid = GridSearchCV(_knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)
    knn_grid.fit(X_train_1, y_train_1)
    print(knn_grid.best_params_, knn_grid.best_score_)
    print(accuracy_score(y_holdout, knn_grid.predict(X_holdout)))
    return (Pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The tree wins here: **94.2% CV / 94.6% hold-out** vs k-NN's ~88.6%/89%. And a random forest — a bunch of trees voting, Topic 5 — only reaches ~95.1%/95.4% for far more compute. Worth remembering against the Topic 1 baselines: the naive "always loyal" constant already scores 85.5%, so the tree's real contribution is the ~9 points on top of it.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    X_holdout,
    X_train_1,
    accuracy_score,
    cross_val_score,
    np,
    y_holdout,
    y_train_1,
):
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=17)
    print(np.mean(cross_val_score(forest, X_train_1, y_train_1, cv=5)))
    forest_params = {"max_depth": range(6, 12), "max_features": range(4, 19)}  # ~0.949
    forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=-1, verbose=True)
    forest_grid.fit(X_train_1, y_train_1)
    print(forest_grid.best_params_, forest_grid.best_score_)
    print(
        accuracy_score(y_holdout, forest_grid.predict(X_holdout))
    )  # ~0.951 (CV)  # ~0.954
    return (RandomForestClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where each method falls apart

    Neither algorithm is universally good. The failure modes are structural, and knowing them is the actual skill.

    ### Trees choke on diagonal boundaries

    Points labelled by $\text{sign}(x_1 - x_2)$: the true boundary is the line $x_1 = x_2$. Trivial. But a tree can only cut with axis-parallel hyperplanes, so it approximates the diagonal with a monstrous staircase of tiny rectangles and grows very deep doing it. Worse, it has learned nothing about the space beyond the 30×30 training square. 1-NN handles the same task more gracefully — though a linear classifier (Topic 4) is what actually belongs here.
    """)
    return


@app.cell
def _(DecisionTreeClassifier, KNeighborsClassifier, get_grid, np, plt):
    def form_linearly_separable_data(n=500, x1_min=0, x1_max=30, x2_min=0, x2_max=30):
        _data, target = ([], [])
        for i in range(n):
            x1 = np.random.randint(x1_min, x1_max)
            x2 = np.random.randint(x2_min, x2_max)
            if np.abs(x1 - x2) > 0.5:
                _data.append([x1, x2])
                target.append(np.sign(x1 - x2))
        return (np.array(_data), np.array(target))

    _X, _y = form_linearly_separable_data()
    fig, _axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, model, title in [
        (
            _axes[0],
            DecisionTreeClassifier(random_state=17).fit(_X, _y),
            "Easy task. Decision tree complexifies everything",
        ),
        (
            _axes[1],
            KNeighborsClassifier(n_neighbors=1).fit(_X, _y),
            "Easy task, kNN. Not bad",
        ),
    ]:
        _xx, _yy = get_grid(_X)
        _predicted = model.predict(np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
        ax.pcolormesh(_xx, _yy, _predicted, cmap="autumn")
        ax.scatter(
            _X[:, 0],
            _X[:, 1],
            c=_y,
            s=100,
            cmap="autumn",
            edgecolors="black",
            linewidth=1.5,
        )
        ax.set_title(title)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MNIST digits: k-NN's home turf

    sklearn's built-in handwritten digits: 8×8 intensity matrices flattened to 64-length vectors. Here every feature is on the *same* scale and pixel-space distance is genuinely meaningful — exactly the regime where the compactness hypothesis holds and k-NN shines.

    Untuned, the gap is brutal: **k-NN ~97.6% vs tree ~66.7%**. Tuning the tree over a wide grid drags it to ~85.6% CV (~84.3% hold-out), still nowhere close. And 1-NN reaches ~98.7% CV. Even a random forest — which beats k-NN on most datasets — loses here, at ~97.7% CV.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    KNeighborsClassifier,
    Pipeline,
    StandardScaler,
    accuracy_score,
    plt,
    train_test_split,
):
    from sklearn.datasets import load_digits

    _data = load_digits()
    _X, _y = (_data.data, _data.target)
    _f, _axes = plt.subplots(1, 4, sharey=True, figsize=(16, 6))
    for i in range(4):
        _axes[i].imshow(_X[i, :].reshape([8, 8]), cmap="Greys")
    X_train_2, X_holdout_1, y_train_2, y_holdout_1 = train_test_split(
        _X, _y, test_size=0.3, random_state=17
    )
    tree_1 = DecisionTreeClassifier(max_depth=5, random_state=17)
    _knn_pipe = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=10))]
    )
    tree_1.fit(X_train_2, y_train_2)
    _knn_pipe.fit(X_train_2, y_train_2)
    print(
        accuracy_score(y_holdout_1, _knn_pipe.predict(X_holdout_1)),
        accuracy_score(y_holdout_1, tree_1.predict(X_holdout_1)),
    )
    return X_holdout_1, X_train_2, tree_1, y_holdout_1, y_train_2


@app.cell
def _(
    GridSearchCV,
    KNeighborsClassifier,
    RandomForestClassifier,
    X_holdout_1,
    X_train_2,
    accuracy_score,
    cross_val_score,
    np,
    tree_1,
    y_holdout_1,
    y_train_2,
):
    _tree_params = {
        "max_depth": [1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 64],
        "max_features": [1, 2, 3, 5, 10, 20, 30, 50, 64],
    }
    _tree_grid = GridSearchCV(tree_1, _tree_params, cv=5, n_jobs=-1, verbose=True)
    _tree_grid.fit(X_train_2, y_train_2)
    print(_tree_grid.best_params_, _tree_grid.best_score_)
    print(accuracy_score(y_holdout_1, _tree_grid.predict(X_holdout_1)))
    print(
        np.mean(
            cross_val_score(
                KNeighborsClassifier(n_neighbors=1), X_train_2, y_train_2, cv=5
            )
        )
    )
    print(
        np.mean(
            cross_val_score(
                RandomForestClassifier(random_state=17), X_train_2, y_train_2, cv=5
            )
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### k-NN chokes when one feature matters and 99 don't

    Flip it around. Build 100 features where **feature 1 is proportional to the target** and the other 99 are pure noise. A tree of depth **1** solves this perfectly — CV and hold-out both 1.0 — because information gain instantly identifies the one feature that matters.

    k-NN fails no matter how you vary k (the **validation curve** below climbs to a peak of only ~0.91 CV around k=200, then falls off either side — never close to the tree's 1.0). Euclidean distance sums over all 100 dimensions, so 99 noise features swamp the one signal feature. Distance has no notion of feature importance; it treats every dimension as equally worth listening to.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    KNeighborsClassifier,
    Pipeline,
    StandardScaler,
    accuracy_score,
    cross_val_score,
    np,
    plt,
    train_test_split,
):
    def form_noisy_data(n_obj=1000, n_feat=100, random_seed=17):
        np.random.seed(random_seed)
        _y = np.random.choice([-1, 1], size=n_obj)
        x1 = 0.3 * _y
        x_other = np.random.random(size=[n_obj, n_feat - 1])
        return (np.hstack([x1.reshape([n_obj, 1]), x_other]), _y)

    _X, _y = form_noisy_data()
    X_train_3, X_holdout_2, y_train_3, y_holdout_2 = train_test_split(
        _X, _y, test_size=0.3, random_state=17
    )
    cv_scores, holdout_scores = ([], [])
    n_neighb = [1, 2, 3, 5] + list(range(50, 550, 50))
    for k in n_neighb:
        _knn_pipe = Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))]
        )
        cv_scores.append(
            np.mean(cross_val_score(_knn_pipe, X_train_3, y_train_3, cv=5))
        )
        _knn_pipe.fit(X_train_3, y_train_3)
        holdout_scores.append(
            accuracy_score(y_holdout_2, _knn_pipe.predict(X_holdout_2))
        )
    plt.plot(n_neighb, cv_scores, label="CV")
    plt.plot(n_neighb, holdout_scores, label="holdout")
    plt.title("Easy task. kNN fails")
    plt.legend()
    plt.show()
    tree_2 = DecisionTreeClassifier(random_state=17, max_depth=1)
    tree_cv_score = np.mean(cross_val_score(tree_2, X_train_3, y_train_3, cv=5))
    tree_2.fit(X_train_3, y_train_3)
    tree_holdout_score = accuracy_score(y_holdout_2, tree_2.predict(X_holdout_2))
    print(tree_cv_score, tree_holdout_score)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pros and cons

    **Decision trees — pros:** interpretable (rules read as plain English: "if age < 25 and likes motorcycles, deny"); both model and individual prediction are visualizable; fast to train and predict; few parameters; handles numeric and categorical features.

    **Decision trees — cons:** **unstable** — perturb the training set slightly and the whole tree can change, which quietly undermines the interpretability you bought it for; boundaries are axis-parallel hyperplanes only, so diagonals cost you; needs explicit overfitting control (depth/leaf size/pruning); optimal tree search is NP-complete so you're stuck with greedy heuristics; awkward with missing values; **interpolates but cannot extrapolate** — predictions are constant outside the training range.

    **k-NN — pros:** trivial to implement, well studied; a solid first solution for classification, regression, and recommendations; adaptable via a well-chosen metric or kernel (tune the similarity measure and k-NN becomes surprisingly strong); interpretable for small k.

    **k-NN — cons:** slow at prediction time on real datasets, where k is often 100–150; with many variables it's hard to know which features to down-weight; entirely dependent on the distance metric, and the Euclidean default is usually unjustified; no theory for choosing k, only grid search; small k is outlier-sensitive and overfits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **A tree asks the question that removes the most uncertainty.** Entropy measures uncertainty; information gain = entropy before − weighted entropy after; the algorithm greedily maximizes it and recurses.
    - **Gini vs entropy barely matters** — entropy ≈ 2×Gini in shape. Tune `max_depth` instead. Misclassification error is a trap; it's too flat to discriminate.
    - **Trees draw axis-parallel boxes.** Great for rule-shaped data, terrible for diagonals, and they cannot extrapolate past the training range. Regression trees output a staircase.
    - **Numeric features are handled by sorting and only testing thresholds where the target flips.** That's the whole trick.
    - **A fully grown tree overfits.** Cap it with `max_depth` / `min_samples_leaf`, or prune, or hide it inside a random forest.
    - **Always scale features before k-NN.** Unscaled, the largest-magnitude feature silently becomes the whole metric.
    - **k-NN is a lazy learner**: no training, all cost at prediction. It lives or dies by the distance metric — 99 noise features will drown 1 signal feature under Euclidean distance, where a depth-1 tree is perfect.
    - **Never tune on the hold-out set.** Cross-validate on the training portion, touch the hold-out exactly once.
    - **Put preprocessing in a `Pipeline` before cross-validating**, or the scaler leaks validation-fold statistics into training and inflates your score.
    - **Try the simple models first.** On churn, the tree beat k-NN (94.6% vs 89%) and a random forest added ~1 point for far more compute. On MNIST, k-NN beat both. Neither is universally better — that's the point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    This topic pairs two interpretable classifiers with the tooling to evaluate any model. Decision trees split greedily on whichever feature maximizes information gain (entropy-based) or Gini reduction, handle numeric features by testing only the thresholds where the target switches, and carve the space into axis-parallel rectangles — fast, readable, but unstable, unable to extrapolate, and prone to overfitting unless capped by `max_depth`/`min_samples_leaf` or pruned. k-NN skips training entirely, classifying by majority vote among the k closest training points, which demands feature scaling and a well-chosen metric since Euclidean distance weights every dimension equally. Model selection runs on hold-out sets and k-fold cross-validation, automated by `GridSearchCV` and kept honest by `Pipeline` (which prevents preprocessing leakage) and by never tuning against the hold-out. The empirical lesson is that neither method dominates: the tree wins on churn, k-NN wins decisively on MNIST, and each has a synthetic case that destroys it — so start with cheap models, understand their failure modes, and make anything more complex justify itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **classification** = predicting a categorical target
    - **regression** = predicting a numerical target
    - **supervised learning** = learning from instances that carry known targets
    - **accuracy** = fraction of correct predictions
    - **generalization** = performing well on data not seen during training
    - **overfitting** = fitting the training set's accidents rather than its structure
    - **decision tree** = classifier of nested yes/no feature tests ending in leaf predictions
        - **leaf** = a terminal node; predicts the majority class (or mean target) of its samples
        - **split** = a binary test on one feature at a node
        - **`max_depth`** = cap on tree depth; the main anti-overfitting knob
        - **`max_features`** = number of features considered per split
        - **`min_samples_leaf`** = minimum samples allowed in a leaf
    - **entropy** = $-\sum p_i \log_2 p_i$; degree of chaos in a node (1 bit = 50/50, 0 = pure)
    - **information gain** = entropy before a split minus the weighted entropy after
    - **Gini uncertainty** = $1 - \sum_k p_k^2$; sklearn's default criterion, ≈ half of entropy
    - **misclassification error** = $1 - \max_k p_k$; too flat to be useful in practice
    - **variance criterion** = the regression-tree split criterion; minimizes spread of targets within a leaf
    - **pruning** = growing the full tree, then removing nodes that don't pay off under cross-validation
    - **greedy algorithm** = takes the locally best split at each step, with no global optimality guarantee
    - **compactness hypothesis** = under a good metric, similar objects share a class
    - **k-NN** = classify by majority vote of the k nearest training points
        - **k / `n_neighbors`** = how many neighbours vote; small overfits, large oversmooths
        - **distance metric** = how similarity is measured (Euclidean, Manhattan, cosine, Minkowski, Hamming)
        - **weights** = `uniform`, or `distance` to let closer neighbours count more
    - **lazy learner** = does no work at fit time, all work at prediction time
    - **feature scaling** = normalizing feature ranges so no one feature dominates a distance
    - **`StandardScaler`** = scales each feature to zero mean and unit variance
    - **hold-out set** = data reserved out of training, scored exactly once at the end
    - **cross-validation** = train/score K times on rotating folds, then average the scores
    - **`GridSearchCV`** = exhaustive parameter search scored by cross-validation
    - **`Pipeline`** = chained preprocessing + model treated as one estimator, preventing leakage in CV
    - **data leakage** = letting information from validation/test data influence training
    - **validation curve** = a plot of accuracy against one hyperparameter's value
    - **random forest** = an ensemble of full-depth trees whose votes are averaged (Topic 5)
    """)
    return


if __name__ == "__main__":
    app.run()
