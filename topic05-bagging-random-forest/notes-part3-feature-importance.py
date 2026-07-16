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
    # Topic 5 — Part 3: Feature Importance

    A forest of 100 trees is not something you can read. But you can still ask it *which inputs mattered*, and get a defensible answer — this is the main reason random forests survive in places where "explain the model" is a requirement.

    Two ways to answer, and they are not the same computation:

    - **Permutation importance** (Breiman's original): shuffle a feature, see how much accuracy drops.
    - **Mean decrease in impurity (MDI)**: sum the impurity gains at every split that used the feature. This is what `feature_importances_` gives you in sklearn.

    Everything below is how each number is actually produced, and where each one lies to you.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The intuition

    In the toy credit-scoring tree, *Age* splits at the root and *Income* only shows up deep down. That ordering isn't arbitrary — a tree picks the split with the largest information gain first, so **the closer a feature sits to the root, averaged across trees, the more it matters**.

    That's the whole recipe: a feature's importance is the gain it produced, accumulated over every node it split on, over every tree. Averaging across many trees is what turns a single tree's noisy, unstable choice of root feature into a stable ranking.

    Breiman's alternative framing skips gains entirely and asks a counterfactual: *if this feature carried no information, how much worse would the forest be?* You simulate "no information" by shuffling the column, which destroys its association with the target while keeping its marginal distribution intact.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Permutation importance

    Bagging hands you the evaluation set for free: each tree's **out-of-bag (OOB) sample** — the ~37% of rows that tree never saw. Measure the tree's accuracy on its OOB rows, shuffle column $X_j$ among those rows, measure again. The drop is that tree's vote on $X_j$.

    Let $\bar{B}^{(t)}$ be the OOB sample of tree $t$, for $t \in \{1, \dots, N\}$:

    $$PI^{(t)}(X_j) = \frac{\sum_{i \in \bar{B}^{(t)}} I(y_i = \hat{y}_i^{(t)})}{|\bar{B}^{(t)}|} - \frac{\sum_{i \in \bar{B}^{(t)}} I(y_i = \hat{y}_{i,\pi_j}^{(t)})}{|\bar{B}^{(t)}|}$$

    where $\hat{y}_i^{(t)}$ is the prediction before permuting $X_j$, $\hat{y}_{i,\pi_j}^{(t)}$ the prediction after, and $I(\cdot)$ the indicator. By definition $PI^{(t)} = 0$ if tree $t$ never used $X_j$ — a tree that ignores a feature can't lose accuracy when you scramble it.

    Aggregate over the forest:

    - **not normalized:** $PI(X_j) = \frac{1}{N}\sum_{t=1}^{N} PI^{(t)}(X_j)$
    - **normalized** by the standard deviation of the per-tree differences: $z_j = \frac{PI(X_j)}{\hat{\sigma}/\sqrt{N}}$

    The normalized form is the more honest one for comparing features: a big mean drop with wild variance across trees is not the same evidence as a small, consistent drop.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Working it by hand, once

    Ten instances, targets in $\{N, P\}$, five trees. Bootstrapping leaves each tree its own OOB set — say tree 1's is instances $\{2, 4, 5, 6\}$. We want $PI$ for feature $X_2$.

    1. Predict tree 1 on its 4 OOB rows → 3 of 4 correct → **accuracy 0.75**.
    2. Shuffle the $X_2$ *column only* among those 4 rows, re-predict → 2 of 4 correct → **accuracy 0.50**.
    3. Difference $0.75 - 0.50 = 0.25$ — tree 1's vote for $X_2$.

    Repeat for all five trees, average the differences, optionally divide by the standard error. That's the entire algorithm. The one subtlety worth internalizing: you permute *within* the OOB sample, and you permute *one column*, leaving every other feature aligned to its row. Shuffling whole rows would prove nothing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What sklearn actually computes (MDI)

    `feature_importances_` is **not** permutation importance. It's mean decrease in impurity: how much each split using $X_j$ purified its children, weighted by how many samples flowed through.

    For every node $i$ in tree $t$, the reduction in impurity is

    $$RI_i^{(t)} = w_i^{(t)} \cdot I_i^{(t)} - w_{LEFT_i}^{(t)} \cdot I_{LEFT_i}^{(t)} - w_{RIGHT_i}^{(t)} \cdot I_{RIGHT_i}^{(t)}$$

    where $w$ is the weighted fraction of samples reaching a node and $I$ is its impurity (Gini or entropy for classification, MSE for regression). Leaves contribute $RI = 0$ — nothing splits there.

    Per tree, a feature's importance is its *share* of the total reduction, which is why the numbers sum to 1:

    $$FI_j^{(t)} = \frac{\sum_{i \,:\, \text{node } i \text{ splits on } j} RI_i^{(t)}}{\sum_{i \in \text{all nodes}} RI_i^{(t)}}$$

    Then average over the ensemble: $FI_j = \frac{1}{N}\sum_{t=1}^{N} FI_j^{(t)}$.

    **Gini impurity** runs 0 (pure node, one class) to ~1 (mixed). A big drop means the split carved out homogeneous children — the feature did real work.

    Two properties that are easy to miss and matter later: MDI is computed **on training data** (permutation importance uses held-out OOB rows), and it is **free** — the gains are already recorded during fitting, no extra passes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reproducing `feature_importances_` on Iris

    Small enough to check by hand: 3 trees, depth ≤ 3, fixed seed. Target collapsed to one-vs-all (Virginica = 1) so the trees stay readable.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import tree
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()
    data = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    target = pd.Series(iris["target"]).map({0: 0, 1: 0, 2: 1})  # Virginica one-vs-all

    rfc = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=17)
    rfc.fit(data, target)

    tree_list = rfc.estimators_  # the fitted trees live here
    return iris, np, pd, plt, rfc, tree, tree_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plotting the trees exposes the raw material of the calculation: each node prints its `samples`, its `gini`, and the feature it split on. Everything MDI needs is on the picture.
    """)
    return


@app.cell
def _(iris, plt, tree, tree_list):
    for t in tree_list:
        plt.figure(figsize=(14, 8))
        tree.plot_tree(
            t,
            filled=True,
            feature_names=iris["feature_names"],
            class_names=["Y", "N"],
            node_ids=True,
        )
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Take tree 1 and *sepal length* (SL). It splits at two nodes — the root (#0) and node #8 — so its total reduction is the sum of both:

    $$RI_{SL_1}^{(1)} = \tfrac{150}{150}(0.482578) - \tfrac{63}{150}(0.061476) - \tfrac{87}{150}(0.436517) = 0.203578$$

    $$RI_{SL_2}^{(1)} = \tfrac{56}{150}(0.035077) - \tfrac{7}{150}(0.244898) - \tfrac{49}{150}(0) = 0.001667$$

    (Impurities recomputed at full precision, not read off the plot's rounded labels.) Doing the same for the other features in tree 1:

    | Feature $j$ | Total $RI_j^{(1)}$ | $FI_j^{(1)}$ |
    | --- | --- | --- |
    | SL | 0.205244 | 0.445716 |
    | SW | 0.000000 | 0.000000 |
    | PL | 0.035785 | 0.077712 |
    | PW | 0.219453 | 0.476572 |
    | **∑** | **0.460483** | **1.0** |

    *Sepal width* scores exactly 0 — tree 1 never split on it. Now average the three trees:

    | Feature $j$ | $FI_j^{(1)}$ | $FI_j^{(2)}$ | $FI_j^{(3)}$ | $FI_j$ |
    | --- | --- | --- | --- | --- |
    | SL | 0.445716 | 0.000000 | 0.000000 | 0.148572 |
    | SW | 0.000000 | 0.039738 | 0.000000 | 0.013246 |
    | PL | 0.077712 | 0.844925 | 0.162016 | 0.361551 |
    | PW | 0.476572 | 0.115337 | 0.837984 | 0.476631 |

    Read the per-tree columns before the average: SL is the *most* important feature in tree 1 and worth exactly nothing in trees 2 and 3. A single tree's importance ranking is noise. The average is the signal — and that averaging is the whole reason forest importances are usable at all.

    The final column should match `feature_importances_` exactly.
    """)
    return


@app.cell
def _(iris, rfc):
    print(iris["feature_names"])
    print(rfc.feature_importances_)
    # -> [0.14857187 0.01324612 0.36155096 0.47663104], matching the table above
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical example: what makes a hostel rating

    Survey data from Booking.com / TripAdvisor listings. Features are average category ratings (staff, room condition, value for money, …); the target is the hostel's overall score — so this is regression, and impurity is MSE rather than Gini. Same machinery.
    """)
    return


@app.cell
def _(np, pd, plt):
    from sklearn.ensemble import RandomForestRegressor

    DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    hostel_data = pd.read_csv(DATA_PATH + "hostel_factors.csv")

    features = {
        "f1": "Staff",
        "f2": "Hostel booking",
        "f3": "Check-in and check-out",
        "f4": "Room condition",
        "f5": "Shared kitchen condition",
        "f6": "Shared space condition",
        "f7": "Extra services",
        "f8": "General conditions & conveniences",
        "f9": "Value for money",
        "f10": "Customer Co-creation",
    }

    forest = RandomForestRegressor(n_estimators=1000, max_features=10, random_state=0)
    forest.fit(hostel_data.drop(["hostel", "rating"], axis=1), hostel_data["rating"])
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    num_to_plot = 10
    feature_indices = [ind + 1 for ind in indices[:num_to_plot]]

    print("Feature ranking:")
    for f in range(num_to_plot):
        print("%d. %s %f" % (f + 1, features["f" + str(feature_indices[f])], importances[indices[f]]))

    plt.figure(figsize=(15, 5))
    plt.title("Feature Importance")
    bars = plt.bar(
        range(num_to_plot),
        importances[indices[:num_to_plot]],
        color=[str(i / float(num_to_plot + 1)) for i in range(num_to_plot)],
        align="center",
    )
    plt.xticks(range(num_to_plot), feature_indices)
    plt.xlim([-1, num_to_plot])
    plt.legend(bars, [features["f" + str(i)] for i in feature_indices])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Staff** (0.18) and **value for money** (0.15) top the ranking — guests reward people and price-quality ratio above amenities. The actionable read: train staff, fix pricing.

    But read the *shape* of the ranking, not just the order. The spread from 0.18 down to 0.04 is gentle — nothing dominates and nothing is negligible, so dropping any of these features would cost accuracy. Contrast with a ranking where the top feature scores 0.7 and the rest are dust: that's the shape that says "you can cut features". And the gap between #1 and #2 (0.18 vs 0.15) is well inside the noise a different seed would produce, so calling staff strictly more important than price is over-reading the number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where these numbers lie to you

    Three traps, all real, all of which bite in practice:

    - **MDI inflates high-cardinality and continuous features.** More distinct values means more candidate split points, so such a feature wins splits by luck as much as by signal — a random ID column can score respectably. This structural bias is the strongest argument for permutation importance.
    - **MDI is measured on training data**, so it happily rewards features the forest used to overfit. Permutation importance on OOB rows (or any held-out set) doesn't.
    - **Correlated features split the credit.** Two near-duplicate columns each look half as important as either would alone, because trees pick between them at random. Neither looks essential; drop both and the model collapses. Never read low importance as "safe to remove" without checking correlation first.

    Rule of thumb: use `feature_importances_` for a free first look, then confirm anything you plan to *act* on — dropping features, writing conclusions — with `sklearn.inspection.permutation_importance` on held-out data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Importance = how much a feature helps, accumulated over trees.** Individual trees disagree wildly; the average across the forest is what's stable and readable.
    - **Two different computations share one name.** Permutation importance = accuracy drop when a column is shuffled on OOB rows. MDI = impurity gain summed over splits, on training data. Don't compare them or mix their intuitions.
    - **`feature_importances_` is MDI, and it's free** — the gains are recorded during fitting, so it costs nothing extra. That cheapness is why it's everywhere, not evidence that it's right.
    - **A feature a tree never used scores exactly 0 in that tree**, under both methods.
    - **MDI is biased toward continuous / high-cardinality features** — more candidate split points, more chances to win a split by luck.
    - **Correlated features split the credit** and each look unimportant. Low importance never proves a feature is droppable.
    - **Read the shape of the ranking, not just the order.** A flat profile means every feature earns its place; small gaps between neighbours are seed noise, not findings.
    - **Act on permutation importance, not MDI.** Use MDI to look, `permutation_importance` on held-out data to decide.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Random forests are unreadable as models but interrogable as feature rankers, via two distinct computations that share a name. Breiman's **permutation importance** shuffles one column within each tree's out-of-bag sample and measures the accuracy drop, averaged over trees and optionally normalized by the standard error — a counterfactual asking what the forest loses when a feature carries no information, evaluated on rows the tree never trained on. Sklearn's `feature_importances_` instead reports **mean decrease in impurity**: for every node, the sample-weighted impurity of the parent minus that of its children (Gini/entropy for classification, MSE for regression), summed over the nodes that split on each feature, normalized to sum to 1 per tree, then averaged across the ensemble — reproduced here by hand on a 3-tree Iris forest down to the last decimal, where the per-tree columns show a feature ranked first in one tree and worthless in the next two, which is exactly why the averaging matters. MDI comes free with fitting, explaining its ubiquity, but it is computed on training data and structurally favors continuous and high-cardinality features, while correlated columns divide the credit and each read as expendable when neither is. The hostel example shows both the payoff — staff and value-for-money drive ratings — and the discipline: use MDI for a first look, confirm with permutation importance on held-out data before dropping a feature or writing a conclusion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **feature importance** = a score ranking how much each input contributed to the model's predictions
    - **permutation importance** = accuracy drop when one feature's values are randomly shuffled
        - **out-of-bag (OOB) sample** = the rows a given tree's bootstrap left out, used as its free validation set
        - **normalized permutation importance** = the mean drop divided by the standard error of the per-tree drops
    - **MDI (mean decrease in impurity)** = importance as total impurity reduction across splits using the feature, averaged over trees
        - **reduction in impurity** = parent impurity minus sample-weighted child impurities at one node
    - **`feature_importances_`** = sklearn's fitted attribute holding MDI scores, summing to 1
    - **`estimators_`** = sklearn's fitted attribute holding the list of individual trees
    - **impurity** = how mixed a node's labels are; the quantity a split tries to reduce
    - **Gini impurity** = classification impurity measure, 0 (pure) to ~1 (mixed)
    - **entropy** = information-theoretic impurity measure, alternative to Gini
    - **MSE** = the impurity measure used by regression trees
    - **information gain** = the impurity drop achieved by a split
    - **`permutation_importance`** = `sklearn.inspection` function computing permutation importance on any fitted model
    - **high-cardinality bias** = MDI's tendency to overrate features with many distinct values
    """)
    return


if __name__ == "__main__":
    app.run()
