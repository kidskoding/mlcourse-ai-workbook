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
    # Topic 2 — Part 1: A taxonomy of plots

    **Why visualize?** In ML, visualization is not decoration for reports — it is a core working tool at every phase:

    - **Exploration first.** When you meet a new dataset, a few charts summarize it far faster than scrolling raw rows. The eye catches structure (skew, clusters, outliers, correlations) that numbers hide.
    - **Model interpretation & reporting.** We use plots to diagnose model behavior and communicate results, and sometimes to project high-dimensional spaces down to a readable 2D/3D picture.

    This part builds a **taxonomy of plots**, organized by how many variables you're looking at and their statistical types (univariate → multivariate → the whole dataset via dimensionality reduction).

    Part 2 (`notes-part2-seaborn-matplotlib-plotly.ipynb`) covers the *tooling* — pandas `.plot()`, Seaborn, Plotly — and when to reach for each.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The mental model: pick a plot by *how many* variables and *what type*

    Every choice of chart comes down to two questions.

    **How many variables at once?**
    - **Univariate** — one variable in isolation. You care about the *distribution* of its values.
    - **Multivariate** (the two-variable case is **bivariate**) — relationships *between* variables in a single figure.
    - **Whole-dataset** — all features at once, which usually requires dimensionality reduction.

    **What statistical type is each variable?**
    - **Quantitative** — ordered numeric values, either *discrete* (integer counts) or *continuous* (real measurements). E.g. *Total day minutes*.
    - **Categorical** — a fixed set of labels/groups. A **binary** feature is the special case of exactly 2 values (e.g. "Yes"/"No"); an **ordinal** feature is a categorical whose values have a natural order (e.g. *Customer service calls*: 0,1,2,…).

    The right plot is the intersection of these two axes. Part 1 walks that grid.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup and dataset

    This part uses the **telecom churn** dataset (one row per customer). The **target** is `Churn` — a binary flag: `True` = the customer left, `False` = retained. Later topics build models to predict it.

    - **Matplotlib** is the foundational plotting library everything else sits on.
    - **Seaborn** wraps Matplotlib with nicer defaults and one-liners for complex statistical plots. `sns.set()` restyles all plots.
    - The SVG config just makes inline charts sharp.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set()
    # Sharper inline graphics
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'svg'

    DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    df = pd.read_csv(DATA_URL + "telecom_churn.csv")
    df.head()
    return df, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Univariate visualization

    **Univariate analysis** looks at one feature at a time, ignoring the rest. The whole game is understanding the **distribution** of that feature's values. The tool you use depends on whether the feature is quantitative or categorical.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantitative features → histograms & density (KDE) plots

    A **histogram** slices the value range into equal-width **bins** and draws a bar per bin whose height = how many observations fell in it. Its *shape* hints at the underlying distribution (Gaussian/bell, exponential, skewed…). This matters because many ML methods *assume* a distribution shape (often Gaussian).

    Reading the two below: *Total day minutes* looks roughly **normal** (symmetric bell); *Total intl calls* is **right-skewed** (long tail to the right — most values small, a few large).
    """)
    return


@app.cell
def _(df):
    features = ["Total day minutes", "Total intl calls"]
    df[features].hist(figsize=(10, 4));
    return (features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A **density plot** (formally a **Kernel Density Estimate, KDE**) is a *smoothed* version of the histogram: instead of hard bins it lays a small smooth bump (kernel) over each data point and sums them into one continuous curve. Its advantage: the result doesn't jump around with an arbitrary bin-size choice.
    """)
    return


@app.cell
def _(df, features):
    df[features].plot(
        kind="density", subplots=True, layout=(1, 2), sharex=False, figsize=(10, 4)
    );
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Seaborn's `histplot()` combines both: the histogram bars plus the KDE curve on top. With `stat="density"` the bar heights are normalized to a density (area = 1) rather than raw counts, so they're comparable to the KDE.
    """)
    return


@app.cell
def _(df, sns):
    sns.histplot(df["Total intl calls"], kde=True, stat="density");
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantitative features → box plot

    A **box plot** summarizes a distribution with five numbers instead of showing every point:

    - The **box** spans the *interquartile range* (IQR): from **Q1** (25th percentile) to **Q3** (75th percentile) — the middle 50% of the data. The line inside is the **median** (50th percentile).
    - The **whiskers** extend to the furthest points still within `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`, where `IQR = Q3 − Q1`.
    - Points beyond the whiskers are drawn individually as **outliers**.

    Great for spotting spread, skew, and outliers at a glance. Here it shows large numbers of international calls are rare (a cluster of high outliers).
    """)
    return


@app.cell
def _(df, sns):
    sns.boxplot(x="Total intl calls", data=df);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantitative features → violin plot

    A **violin plot** is a box plot crossed with a KDE: a mirrored density curve on each side shows the full *shape* of the distribution (where data is dense vs. sparse), not just its quartiles.

    **Box vs. violin:** the box plot emphasizes summary *statistics* (quartiles, outliers of individual points); the violin emphasizes the *smoothed distribution as a whole*. Reach for a violin when the shape (e.g. bimodality) matters. Often, as here, the box already told the whole story and the violin adds nothing.
    """)
    return


@app.cell
def _(df, plt, sns):
    _, _axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
    sns.boxplot(data=df['Total intl calls'], ax=_axes[0])
    sns.violinplot(data=df['Total intl calls'], ax=_axes[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantitative features → exact numbers with `describe()`

    Charts give intuition; `describe()` gives the exact summary statistics (count, mean, std, min, the 25/50/75% percentiles, max) behind the box plot.
    """)
    return


@app.cell
def _(df, features):
    df[features].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical & binary features → frequency table

    For a categorical feature we don't ask about a smooth distribution — we ask *how often does each value occur*. `value_counts()` gives a **frequency table**, sorted most→least common by default.

    Here it exposes **class imbalance**: far more loyal (`False`) customers than churners (`True`). Imbalance matters later — accuracy can be misleading, and we may need to penalize errors on the rare "Churn" class.
    """)
    return


@app.cell
def _(df):
    df["Churn"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical & binary features → bar / count plot

    A **count plot** (Seaborn's `countplot()`) is the graphical version of the frequency table — one bar per category, height = its count.

    **Histogram vs. bar/count plot** (they look alike but differ):
    1. Histogram = *numeric* variable's distribution; bar plot = *categorical* counts.
    2. Histogram X-axis is numeric; a bar plot's X can be anything (strings, booleans, numbers).
    3. A histogram's X-axis is a real coordinate axis (bins can't be reordered); bar order is free — often sorted by height, but for **ordinal** features (like *Customer service calls*) kept in value order.

    *(Aside: Seaborn's `barplot()` is a different, confusingly-named function that shows an aggregate statistic — e.g. mean — of a numeric variable per category, not raw counts.)*
    """)
    return


@app.cell
def _(df, plt, sns):
    _, _axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.countplot(x='Churn', data=df, ax=_axes[0])
    sns.countplot(x='Customer service calls', data=df, ax=_axes[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multivariate visualization

    **Multivariate** plots reveal *relationships between* variables in one figure. Again, the plot type follows the variable types: quantitative×quantitative, quantitative×categorical, categorical×categorical.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantitative vs. Quantitative → correlation matrix (heatmap)

    `corr()` computes the pairwise **correlation** between every pair of numeric columns (−1 to +1). A **heatmap** renders that matrix as a color grid so strong correlations pop out visually.

    Why care: some algorithms (linear/logistic regression) behave badly with highly correlated inputs. Here the heatmap reveals that the four `*charge` columns are **dependent** — computed directly from the corresponding `*minutes` columns — so they carry no new information and can be dropped.
    """)
    return


@app.cell
def _(df, sns):
    # Keep only numeric features (drop categoricals/target)
    numerical = list(
        set(df.columns)
        - {"State", "International plan", "Voice mail plan",
           "Area code", "Churn", "Customer service calls"}
    )
    corr_matrix = df[numerical].corr()
    sns.heatmap(corr_matrix);
    return (numerical,)


@app.cell
def _(numerical):
    # Drop the redundant 'charge' columns (perfectly derived from 'minutes')
    numerical_1 = list(set(numerical) - {'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge'})
    return (numerical_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantitative vs. Quantitative → scatter plot & joint plot

    A **scatter plot** places each observation as a point at its (x, y) coordinates — the go-to for seeing the relationship between two numeric variables. An axis-aligned elliptical blob (like below) means the two are roughly uncorrelated.
    """)
    return


@app.cell
def _(df, plt):
    plt.scatter(df["Total day minutes"], df["Total night minutes"]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A **joint plot** (`sns.jointplot`) is a scatter plot *plus* the marginal histogram of each variable along the edges — one figure showing both the joint relationship and each individual distribution. Switch `kind="kde"` for a smoothed 2D density (contour) version — the bivariate cousin of the density plot.
    """)
    return


@app.cell
def _(df, sns):
    sns.jointplot(x="Total day minutes", y="Total night minutes", data=df, kind="scatter");
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantitative vs. Quantitative → pair plot (scatterplot matrix)

    A **pair plot** / **scatterplot matrix** is the brute-force overview of many numeric features at once: a grid where the diagonal holds each feature's own distribution and every off-diagonal cell is the scatter plot of that pair. Great for a quick scan, but it grows as *n²* and gets slow/unreadable with many features. (Use PNG, not SVG — SVG pairplots are very slow.)
    """)
    return


@app.cell
def _(df, numerical_1, sns):
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'png'
    # %config InlineBackend.figure_format = 'svg'
    sns.pairplot(df[numerical_1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantitative vs. Categorical

    To see how a numeric feature relates to a category (like the target `Churn`), two workhorses:

    **1. Color-coded scatter** via `lmplot(..., hue=...)` — the `hue` parameter injects a categorical dimension by coloring points. (`fit_reg=False` turns off the regression line.) Here churners lean toward the top-right = heavier phone use.
    """)
    return


@app.cell
def _(df, sns):
    sns.lmplot(
        x="Total day minutes", y="Total night minutes",
        data=df, hue="Churn", fit_reg=False
    );
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **2. Grouped box plots** — one box per category lets you compare distributions across groups. Looping box plots of every numeric feature split by `Churn` quickly reveals *which* features separate the two groups. The biggest gaps here: *Total day minutes*, *Customer service calls*, *Number vmail messages* — a preview of feature importance.
    """)
    return


@app.cell
def _(df, numerical_1, plt, sns):
    numerical_1.append('Customer service calls')
    fig, _axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 7))
    for idx, feat in enumerate(numerical_1):
        ax = _axes[idx // 4, idx % 4]
        sns.boxplot(x='Churn', y=feat, data=df, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel(feat)
    fig.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Zooming into the strongest signal — day minutes split by churn — as both box and violin. **Insight:** churners talk *more* on average. A business hypothesis: maybe rates are too high, so cutting call rates could reduce churn.
    """)
    return


@app.cell
def _(df, plt, sns):
    _, _axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    sns.boxplot(x='Churn', y='Total day minutes', data=df, ax=_axes[0])
    sns.violinplot(x='Churn', y='Total day minutes', data=df, ax=_axes[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When you need a numeric variable split across **two** categorical dimensions at once, `catplot()` builds a grid of facets (here, one box-plot panel per *Customer service calls* value, each split by *Churn*). **Insight:** past ~4 service calls, day-minutes stops being the churn driver — something else is going wrong.
    """)
    return


@app.cell
def _(df, sns):
    sns.catplot(
        x="Churn", y="Total day minutes", col="Customer service calls",
        data=df[df["Customer service calls"] < 8],
        kind="box", col_wrap=4, height=3, aspect=0.8,
    );
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical vs. Categorical → count plot with `hue`, and contingency tables

    Adding `hue` to a **count plot** stacks a second categorical dimension onto the bars. Splitting *Customer service calls* by *Churn* shows the churn rate jumps sharply at **4+ calls** — frustrated customers leave.
    """)
    return


@app.cell
def _(df, sns):
    sns.countplot(x="Customer service calls", hue="Churn", data=df);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same trick on the binary plan features. **Insight:** enabling the *International plan* strongly raises churn; *Voice mail plan* shows no such effect. A strong-signal feature is exactly what we want for prediction.
    """)
    return


@app.cell
def _(df, plt, sns):
    _, _axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    sns.countplot(x='International plan', hue='Churn', data=df, ax=_axes[0])
    sns.countplot(x='Voice mail plan', hue='Churn', data=df, ax=_axes[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A **contingency table** (a.k.a. **cross-tabulation**, `pd.crosstab`) is the pure-numbers tool for categorical×categorical: a table of counts for each combination of values. Reading along a row/column shows one variable's distribution *conditional* on the other.

    **Caution on small samples:** grouping churn by *State* gives just a handful of churners per state, so per-state churn rates (NJ/CA look high, HI/AK low) are unreliable noise — you'd need a proper association measure (Cramér's V, Matthews) to trust them.
    """)
    return


@app.cell
def _(df, pd):
    pd.crosstab(df["State"], df["Churn"]).T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Whole-dataset visualization & dimensionality reduction

    So far we hand-picked 2–3 variables at a time. To see **all** features together, pairwise scanning (`hist`/`pairplot`) breaks down: too slow, and still only pairwise.

    **Dimensionality reduction** solves this — derive a few new low-dimensional features that preserve most of the information, so we can plot the whole dataset in 2D. It's an **unsupervised** problem (we build the new features from the data itself, no labels).

    - **PCA** (Principal Component Analysis) — a *linear* method (studied later). Fast, but limited to linear structure.
    - **Manifold learning** — *non-linear* methods that can unfold curved structure. The famous one is **t-SNE**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## t-SNE

    **t-SNE** = *t-distributed Stochastic Neighbor Embedding*. Intuition (skip the heavy math): find a 2D layout of the points such that points **close** in the original high-dimensional space stay close, and points **far apart** stay far apart. It preserves *neighborhoods*.

    Prep: drop non-numeric/target columns, map the "Yes"/"No" binaries to 1/0, then **standardize** (subtract mean, divide by std) with `StandardScaler` so no feature dominates by scale.
    """)
    return


@app.cell
def _(df):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    X = df.drop(["Churn", "State"], axis=1)
    X["International plan"] = X["International plan"].map({"Yes": 1, "No": 0})
    X["Voice mail plan"] = X["Voice mail plan"].map({"Yes": 1, "No": 0})

    X_scaled = StandardScaler().fit_transform(X)
    return TSNE, X_scaled


@app.cell
def _(TSNE, X_scaled, plt):
    tsne = TSNE(random_state=17)
    tsne_repr = tsne.fit_transform(X_scaled)
    plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], alpha=0.5);
    return (tsne_repr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Coloring the same 2D embedding by `Churn` (and by the binary plan features) reveals structure the raw scatter hid: churners cluster in a few regions, and one dense churn cluster corresponds to customers *with* an international plan but *no* voicemail — combining what we found feature-by-feature earlier.
    """)
    return


@app.cell
def _(df, plt, tsne_repr):
    plt.scatter(
        tsne_repr[:, 0], tsne_repr[:, 1],
        c=df["Churn"].map({False: "blue", True: "orange"}), alpha=0.5,
    );
    return


@app.cell
def _(df, plt, tsne_repr):
    _, _axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
    for i, name in enumerate(['International plan', 'Voice mail plan']):
        _axes[i].scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=df[name].map({'Yes': 'orange', 'No': 'blue'}), alpha=0.5)
        _axes[i].set_title(name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **t-SNE caveats — don't over-read it:**
    - **Slow** — scikit-learn's version won't scale to large data (try Multicore-TSNE).
    - **Unstable** — the picture changes a lot with the random seed, so distances/cluster sizes aren't literal. Treat clusters as *hypotheses to verify*, not conclusions. (See distill.pub, "How to Use t-SNE Effectively".)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Choose the plot from the variable count × type grid.** Univariate numeric → histogram/KDE/box/violin. Univariate categorical → frequency table/count plot. Bivariate: num×num → scatter/joint/pairplot/heatmap-of-corr; num×cat → grouped box or hue-colored scatter; cat×cat → count plot with `hue` or crosstab.
    - **Histogram vs. bar plot** are not the same: numeric distribution vs. categorical counts.
    - **Box vs. violin:** box = quartile summary + outliers; violin = full smoothed shape. Prefer the box unless shape (e.g. bimodality) matters.
    - **`hue` / faceting** are how you fold an extra categorical dimension into a 2-variable plot — the key to hunting predictive signal (e.g. which features separate churners).
    - **Correlation heatmaps** catch redundant/derived features to drop before modeling.
    - **Dimensionality reduction (PCA linear, t-SNE non-linear)** lets you view the whole dataset in 2D — but t-SNE is slow, seed-sensitive, and only good for generating hypotheses.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Part 1 builds a **decision framework** for charts: name your variables (how many at once, what statistical type) and the framework points at the right plot — from a single histogram, through scatter/joint/pair plots and correlation heatmaps, up to a whole-dataset t-SNE map. Along the way each plot is used to extract a concrete churn-prediction insight, which is the real point: visualization is how you find the features that separate the classes before you model them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **univariate** = looking at one variable in isolation — you care about its distribution
    - **bivariate** = the relationship between exactly two variables
    - **multivariate** = relationships among several variables in one figure
    - **quantitative** = ordered numeric values, either discrete (integer counts) or continuous
    - **categorical** = a fixed set of labels with no inherent order
    - **binary** = a categorical with exactly two values
    - **ordinal** = a categorical whose values have a meaningful order
    - **histogram** = binned counts of a numeric variable
    - **KDE / density plot** = a smoothed, bin-free distribution curve
    - **box plot** = median + IQR box + whiskers + outliers
        - **IQR** = Q3 − Q1, the span covering the middle 50% of values
    - **violin plot** = a box plot fused with a mirrored KDE, showing full distribution shape
    - **count / bar plot** = bars giving the frequency of each category
    - **correlation matrix** = pairwise linear correlations between numeric features
    - **heatmap** = a matrix rendered as colored cells
    - **scatter plot** = one point at (x, y) per row, for two numeric features
    - **joint plot** = a scatter plot with marginal distributions on both axes
    - **pair plot / scatterplot matrix** = a grid of all pairwise scatters, distributions on the diagonal
    - **contingency table / crosstab** = counts per combination of two categorical features
    - **hue** = folding an extra categorical dimension into a plot by coloring points/bars
    - **dimensionality reduction** = deriving a few low-dimensional features that preserve the information
    - **PCA** = linear dimensionality reduction
    - **t-SNE** = non-linear manifold method preserving neighborhoods, for 2D visualization only
    """)
    return


if __name__ == "__main__":
    app.run()
