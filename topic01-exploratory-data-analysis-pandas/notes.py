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
    # Topic 1: Exploratory Data Analysis with Pandas

    **Exploratory data analysis (EDA)**: the step before any modeling: you load the data, look at its shape and types, summarize distributions, and slice it in different ways to build intuition and check simple assumptions. Historically this was the whole data-analysis process before ML existed, and it still pays off — you often discover a simple "if-else" rule that a fancy model must beat to justify itself
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What Pandas is, and its two core objects

    **Pandas** is a Python library for analyzing tabular data (`.csv`, `.tsv`, `.xlsx`). It lets you load, filter, aggregate, and reshape tables with SQL-like operations, and pairs naturally with Matplotlib/Seaborn for plots

    Two data structures do all the work:

    - **Series**: a one-dimensional labeled array holding values of a *single* type (like one column, plus an index). The **index** is the set of row labels attached to the values.
    - **DataFrame**: a two-dimensional table; think of it as a dict of Series sharing one index. Each **column** is a Series of one type.

    Vocabulary that recurs all course:
    - An **instance** (a.k.a. observation, sample) is one **row** — here, one telecom customer
    - A **feature** = one **column** is a measured attribute of the instance (e.g. `Total day minutes`)
    - **Churn** = a customer leaving the service. It's the **target** we ultimately want to predict; `1` = left, `0` = loyal.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    pd.set_option("display.precision", 2)
    return np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading data and first look

    `read_csv` reads a CSV file (local path or URL) into a DataFrame. `head(n)` shows the first `n` rows (default 5) — the reflex first move to eyeball what you loaded.
    """)
    return


@app.cell
def _(pd):
    DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    df = pd.read_csv(DATA_URL + "telecom_churn.csv")
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dataset has **3333 rows × 20 columns** — 3333 customers, 20 features (state, plan flags, call minutes/charges by time of day, customer-service calls, and the `Churn` target).

    Three quick inspectors:
    - `df.shape` → `(rows, columns)` tuple.
    - `df.columns` → the column labels (an Index object).
    - `df.info()` → per-column non-null counts and **dtype**, plus memory use.

    A **dtype** is the storage type of a column's values: `int64`/`float64` (numbers), `bool` (True/False), `object` (usually strings). `info()` is also how you spot **missing values** — if a column's non-null count is below the row count, some cells are empty. Here every column has 3333 non-nulls, so nothing is missing.
    """)
    return


@app.cell
def _(df):
    print(df.shape)
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Changing a column's type: `astype`

    `astype` casts a column to another dtype. Converting the boolean `Churn` to `int64` (True→1, False→0) makes it easy to average and cross-tabulate later — the mean of a 0/1 column is directly the fraction of 1s.
    """)
    return


@app.cell
def _(df):
    df["Churn"] = df["Churn"].astype("int64")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary statistics

    `describe()` reports, for every **numeric** column, the count, mean, standard deviation, min, max, median, and the 25%/75% quartiles. A **quartile** is a cut point: the 25% quartile is the value below which a quarter of the data falls; the median (50%) is the middle value. This is your first read on scale and spread of each feature.
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To summarize **non-numeric** columns instead, pass `include=`. For `object`/`bool` columns you get `count`, `unique` (number of distinct values), `top` (most frequent), and `freq` (how often the top appears). E.g. there are 51 distinct states; `International plan` is "No" for 3010 of 3333 customers.
    """)
    return


@app.cell
def _(df):
    df.describe(include=["object", "bool"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Counting categories: `value_counts`

    `value_counts()` tallies how many times each distinct value appears in a Series — the go-to for categorical/boolean columns. Add `normalize=True` to get **fractions** instead of raw counts.

    For `Churn`: 2850 loyal (0) vs 483 churned (1) — about **14.5% churn**. That's a bad rate for a real company, but the imbalance also matters for modeling: a model that always guesses "loyal" is already right ~85.5% of the time (more on that at the end).
    """)
    return


@app.cell
def _(df):
    df["Churn"].value_counts()
    df["Churn"].value_counts(normalize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sorting

    `sort_values(by=...)` returns the DataFrame reordered by one or more columns. `ascending=False` sorts high-to-low; pass a *list* of booleans to set the direction per column when sorting by several. Sorting doesn't change the data, only its order — handy for spotting extremes.
    """)
    return


@app.cell
def _(df):
    df.sort_values(by="Total day charge", ascending=False).head()

    # sort by several columns, each with its own direction
    df.sort_values(by=["Churn", "Total day charge"], ascending=[True, False]).head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Indexing and retrieving data

    Several ways to pull out pieces of a DataFrame:

    **Single column** — `df["Churn"]` returns that Series. Because `Churn` is 0/1, `df["Churn"].mean()` gives the churn proportion directly (0.145).
    """)
    return


@app.cell
def _(df):
    df["Churn"].mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Boolean indexing (filtering rows)

    **Boolean indexing** keeps only the rows where a condition is True. The condition `df["Churn"] == 1` produces a boolean Series (True/False per row); `df[that]` returns just the matching rows. It's the workhorse for "show me the subset where…" questions.

    Combine conditions with `&` (and) / `|` (or), and **wrap each condition in parentheses** — operator precedence bites otherwise.
    """)
    return


@app.cell
def _(df, np):
    # Average day-minutes among churned customers
    df[df["Churn"] == 1]["Total day minutes"].mean()  # ~206.9

    # Max international minutes among loyal customers with no international plan
    df[(df["Churn"] == 0) & (df["International plan"] == "No")][
        "Total intl minutes"
    ].max()  # 18.9

    # Average of all numeric features for churned users
    df.select_dtypes(include=np.number)[df["Churn"] == 1].mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Label vs. position: `.loc` and `.iloc`

    Two explicit indexers, easy to confuse:

    - **`.loc[rows, cols]`** selects by **label** (names). Label slices are **inclusive** of both ends: `df.loc[0:5, "State":"Area code"]` returns rows 0 through 5 *and* every column from `State` to `Area code`.
    - **`.iloc[rows, cols]`** selects by **integer position**. Position slices follow normal Python rules — the **end is excluded**: `df.iloc[0:5, 0:3]` is the first 5 rows and first 3 columns.

    (`df[:1]` / `df[-1:]` still work for grabbing the first / last row.)
    """)
    return


@app.cell
def _(df):
    df.loc[0:5, "State":"Area code"]  # by label, endpoints included
    df.iloc[0:5, 0:3]  # by position, end excluded
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Applying functions to cells, columns, and rows

    - **`apply(func)`** runs `func` on each **column** by default (e.g. `df.apply(np.max)` → max of every column). Pass **`axis=1`** to run it across each **row** instead. Lambdas are handy here.
    - **`map(dict)`** on a Series replaces values via a `{old: new}` mapping — great for recoding categories.

    Note the subtle difference between `map` and `replace`: for a value **not** in the mapping dictionary, `map` sets it to **NaN** (missing), whereas `replace` leaves it untouched. Prefer `replace` when you only want to remap *some* values.
    """)
    return


@app.cell
def _(df, np):
    df.apply(np.max)  # max of each column
    df[df["State"].apply(lambda state: state[0] == "W")].head()
    # axis=1 applies per row; here filter states starting with 'W'
    d = {"No": False, "Yes": True}
    df["International plan"] = df["International plan"].map(d)
    # recode Yes/No -> True/False
    df_1 = df.replace({"Voice mail plan": d})  # replace leaves unmapped values as-is
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grouping: split–apply–combine

    **`groupby`** implements the split–apply–combine pattern, the single most useful aggregation tool:

    ```
    df.groupby(by=grouping_columns)[columns_to_show].function()
    ```

    1. **Split** — partition rows into groups by the distinct values of `grouping_columns` (those values become the new index).
    2. **Select** — pick the columns to summarize (`columns_to_show`).
    3. **Apply** — compute a statistic per group.

    Use **`.agg([...])`** to apply *several* functions at once. Below, grouping by `Churn` shows that churned customers average noticeably more daytime minutes than loyal ones — an early signal that call-usage relates to churn.
    """)
    return


@app.cell
def _(df_1):
    columns_to_show = ["Total day minutes", "Total eve minutes", "Total night minutes"]
    df_1.groupby(["Churn"])[columns_to_show].describe(percentiles=[])
    # equivalent, choosing the statistics explicitly
    df_1.groupby(["Churn"])[columns_to_show].agg(["mean", "std", "min", "max"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary tables: crosstab and pivot_table

    A **contingency table** (a.k.a. cross-tabulation) counts how many rows fall into each combination of two categorical variables — it's how you see whether two categories move together.

    - **`pd.crosstab(a, b)`** builds that count table. `normalize=True` gives proportions instead of counts; `margins=True` adds row/column totals (an `All` line).
    - **`pivot_table`** is the spreadsheet-style generalization. Key arguments:
      - `values` — column(s) to compute statistics on,
      - `index` — column(s) to group rows by,
      - `aggfunc` — the statistic (`"mean"`, `"sum"`, `"max"`, …).
    """)
    return


@app.cell
def _(df_1, pd):
    pd.crosstab(df_1["Churn"], df_1["International plan"])
    pd.crosstab(df_1["Churn"], df_1["Voice mail plan"], normalize=True)
    df_1.pivot_table(
        ["Total day calls", "Total eve calls", "Total night calls"],
        ["Area code"],
        aggfunc="mean",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DataFrame transformations: add and drop

    **Add a column** — assign a computed Series to a new name: `df["Total charge"] = df["Total day charge"] + ...`. For explicit placement use `df.insert(loc, column, value)` (`loc` = position to insert at).

    **Drop rows/columns** — `df.drop(labels, axis=...)`. `axis=1` drops **columns**, `axis=0` (default) drops **rows**. `inplace=True` mutates the DataFrame in place; `inplace=False` (default) returns a modified copy and leaves the original alone.
    """)
    return


@app.cell
def _(df_1):
    # add a column directly from a computed expression
    df_1["Total charge"] = (
        df_1["Total day charge"]
        + df_1["Total eve charge"]
        + df_1["Total night charge"]
        + df_1["Total intl charge"]
    )
    df_1.drop(["Total charge"], axis=1, inplace=True)
    # drop columns (axis=1, in place); dropping rows uses axis=0
    df_1.drop([1, 2]).head()  # returns a copy without rows 1 and 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First attempt at "predicting" churn — without any ML

    EDA alone can produce a decent predictor. The idea: find features that clearly separate churners from loyal customers, then write a simple rule.

    **Signal 1 — International plan.** The crosstab (with `margins=True`) shows churn is far more common among customers who have the international plan. Plausibly, high/poorly-managed international charges breed discontent.
    """)
    return


@app.cell
def _(df_1, pd):
    pd.crosstab(df_1["Churn"], df_1["International plan"], margins=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Signal 2 — Customer service calls.** Cross-tabulating churn against the number of support calls shows the churn rate spikes sharply once a customer makes **4 or more** calls — a classic "this person is frustrated" indicator. A `countplot` (Seaborn) makes it obvious visually; visual EDA is Topic 2.
    """)
    return


@app.cell
def _(df_1, pd):
    pd.crosstab(df_1["Churn"], df_1["Customer service calls"], margins=True)
    df_1["Many_service_calls"] = (df_1["Customer service calls"] > 3).astype("int")
    # Turn the threshold into a binary feature: more than 3 service calls
    pd.crosstab(df_1["Many_service_calls"], df_1["Churn"], margins=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Combine the two signals** into one rule and check it against the data:

    ```
    Churn = 1  if  (International plan == True)  AND  (Customer service calls > 3)
    Churn = 0  otherwise
    ```
    """)
    return


@app.cell
def _(df_1, pd):
    pd.crosstab(
        df_1["Many_service_calls"] & df_1["International plan"],
        df_1["Churn"],
        margins=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Baselines: the numbers your ML must beat

    A **baseline** is the accuracy of a trivially simple predictor — the bar any "real" model has to clear to be worth its complexity. **Accuracy** = fraction of predictions that are correct.

    Two baselines from pure EDA:

    - **85.5% — the naive constant model.** 85.5% of customers are loyal, so *always* predicting "loyal" is right 85.5% of the time. Any model scoring below this is worthless.
    - **85.8% — the two-condition rule** (`International plan AND >3 service calls → churn`). From the last crosstab it's wrong only 464 + 9 times, so accuracy ≈ 1 − (464+9)/3333 ≈ **85.8%** — a hair above naive, from hand-written logic.

    Why this matters:
    - These baselines are the **starting line** for every later model. Decision trees (Topic 3) will *learn* rules like this one **automatically** from the data.
    - If a complex model beats 85.8% by only ~0.5%, that's a red flag — maybe the effort isn't justified and a two-line if-else suffices.
    - **Always** wrangle the data, plot it, and check simple assumptions *before* reaching for heavy models. In industry you ship the simple solution first, then iterate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **EDA first.** Look at shape, dtypes, missing values, and distributions before modeling — it builds intuition and catches problems early.
    - **Series = one column; DataFrame = the table.** Rows are instances, columns are features.
    - **Inspect:** `head`, `shape`, `columns`, `info`, `describe`, `value_counts`.
    - **Filter** with boolean indexing (parenthesize each condition, combine with `&`/`|`).
    - **Select** with `.loc` (labels, end inclusive) vs `.iloc` (positions, end exclusive).
    - **Transform** with `apply`/`map`/`replace`; **aggregate** with `groupby(...).agg(...)`.
    - **Summarize relationships** with `crosstab` and `pivot_table`.
    - **Establish baselines** (naive constant + simple rule) so you know what any ML model must beat.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    This module walks through Pandas end-to-end on a telecom-churn dataset: loading and inspecting a DataFrame, computing summary statistics, filtering with boolean indexing, selecting by label vs. position, transforming columns, and aggregating with `groupby`, `crosstab`, and `pivot_table`. It closes by using EDA alone to build two baselines — a 85.5% "always loyal" constant and an 85.8% two-condition rule — that set the bar for the machine-learning models in later topics. The lesson: understand and slice your data with simple tools first; complex models must justify themselves against these cheap baselines.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **EDA** = exploring/summarizing data before modeling
    - **churn** = a customer leaving (target: 1=left, 0=loyal)
    - **Series** = 1-D labeled single-type array
    - **DataFrame** = 2-D table of Series
        - **instance** = a row/observation
        - **feature** = a column/attribute
        - **index** = row labels
        - **dtype** = a column's storage type (int64/float64/bool/object)
    - **boolean indexing** = keep rows where a condition is True
    - **`.loc`** = select by label (end inclusive)
    - **`.iloc`** = select by integer position (end exclusive)
    - **`groupby`** = split–apply–combine aggregation
        - **`agg`** = apply several functions per group
    - **crosstab** = contingency table counting category combinations
    - **pivot_table** = spreadsheet-style grouped aggregation
    - **baseline** = accuracy of a trivial predictor
    - **accuracy** = fraction of correct predictions
    """)
    return


if __name__ == "__main__":
    app.run()
