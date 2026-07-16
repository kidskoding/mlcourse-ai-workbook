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
    # Topic 2 — Part 2: The tooling — pandas `.plot()`, Seaborn, Plotly

    This part re-covers the plots from Part 1 (`notes-part1-visual-data-analysis.ipynb`) but focuses on *which library to use*.

    It switches to a **video-game sales** dataset (one row per game: platform, genre, year, sales by region, critic/user scores). Some rows lack ratings, so `.dropna()` keeps only complete records.

    The three libraries are a **stack, not rivals**: pandas `.plot()` calls Matplotlib, and Seaborn is built on Matplotlib too — so anything Seaborn draws can still be tweaked with Matplotlib afterwards.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    sns.set()
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'svg'
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["image.cmap"] = "viridis"

    DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    df = pd.read_csv(DATA_URL + "video_games_sales.csv").dropna()

    # Some numeric columns loaded as object; fix dtypes
    df["User_Score"] = df["User_Score"].astype("float64")
    df["Year_of_Release"] = df["Year_of_Release"].astype("int64")
    df["User_Count"] = df["User_Count"].astype("int64")
    df["Critic_Count"] = df["Critic_Count"].astype("int64")
    print(df.shape)
    return df, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `DataFrame.plot()` — the quickest path

    Every pandas DataFrame has a `.plot()` method built on Matplotlib. For quick exploration it's often the fastest option — no extra imports. Switch chart types with `kind=` (`"line"` default, `"bar"`, etc.); tweak Matplotlib params like `rot` (x-tick rotation).

    Here: total sales per region over time, as a line chart then a bar chart.
    """)
    return


@app.cell
def _(df):
    sales_by_year = (
        df[[c for c in df.columns if "Sales" in c] + ["Year_of_Release"]]
        .groupby("Year_of_Release").sum()
    )
    sales_by_year.plot();
    return (sales_by_year,)


@app.cell
def _(sales_by_year):
    sales_by_year.plot(kind="bar", rot=45);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Seaborn — higher-level statistical plots

    **Seaborn** is a higher-level API on top of Matplotlib: better defaults (just `import seaborn as sns; sns.set()`) and one-liners for plots that would take a lot of raw Matplotlib code. The library-focused tour of its key functions:

    **`pairplot()`** — the scatterplot matrix again: diagonal = each feature's distribution, off-diagonal = pairwise scatters. (PNG, not SVG — it's slow.)
    """)
    return


@app.cell
def _(df, sns):
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'png'
    sns.pairplot(
        df[["Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]]
    );
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'svg'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`histplot()`** — distribution of one numeric variable, here critic scores, with the KDE overlaid (`kde=True`, `stat="density"`).
    """)
    return


@app.cell
def _(df, sns):
    sns.histplot(df["Critic_Score"], kde=True, stat="density");
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`jointplot()`** — a scatter of two numeric variables with each one's histogram on the margins; good for eyeballing a relationship (critic vs. user scores) and both marginals together.
    """)
    return


@app.cell
def _(df, sns):
    sns.jointplot(x="Critic_Score", y="User_Score", data=df, kind="scatter");
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`boxplot()`** — compare a numeric distribution across categories. Here critic scores across the top-5 platforms (`orient="h"` for horizontal boxes). Same box/whisker/outlier reading as Part 1.
    """)
    return


@app.cell
def _(df, sns):
    top_platforms = (
        df["Platform"].value_counts().sort_values(ascending=False).head(5).index.values
    )
    sns.boxplot(
        y="Platform", x="Critic_Score",
        data=df[df["Platform"].isin(top_platforms)], orient="h",
    );
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`heatmap()`** — shows a numeric value over a grid of **two** categorical variables (color = magnitude). Combine with `pivot_table` to reshape the data first. Here: total sales for each (platform, genre) pair; `annot=True` writes the numbers in each cell.
    """)
    return


@app.cell
def _(df, sns):
    platform_genre_sales = (
        df.pivot_table(index="Platform", columns="Genre",
                       values="Global_Sales", aggfunc="sum")
        .fillna(0)
    )
    sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=0.5);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotly — interactive plots

    **Plotly** builds *interactive* charts inside the notebook (no JavaScript needed). The payoff is a UI for exploration: hover to read exact values, click legend entries to hide series, zoom into regions.

    Core anatomy: a **`Figure`** = **data** (a list of `traces` — each trace is one line/bar/box series) + a **`layout`** (style, title). Call `iplot(fig)` to render.
    """)
    return


@app.cell
def _():
    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode, iplot

    init_notebook_mode(connected=True)
    return go, iplot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Line plot** — two traces (global sales and number of games) vs. year, in one interactive figure.
    """)
    return


@app.cell
def _(df, go, iplot):
    years_df = df.groupby('Year_of_Release')[['Global_Sales']].sum().join(df.groupby('Year_of_Release')[['Name']].count())
    years_df.columns = ['Global_Sales', 'Number_of_Games']
    _trace0 = go.Scatter(x=years_df.index, y=years_df['Global_Sales'], name='Global Sales')
    _trace1 = go.Scatter(x=years_df.index, y=years_df['Number_of_Games'], name='Number of games released')
    _fig = go.Figure(data=[_trace0, _trace1], layout={'title': 'Statistics for video games'})
    iplot(_fig, show_link=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Bar chart** — `go.Bar` traces to compare platforms by sales vs. number of releases.
    """)
    return


@app.cell
def _(df, go, iplot):
    platforms_df = df.groupby('Platform')[['Global_Sales']].sum().join(df.groupby('Platform')[['Name']].count())
    platforms_df.columns = ['Global_Sales', 'Number_of_Games']
    platforms_df.sort_values('Global_Sales', ascending=False, inplace=True)
    _trace0 = go.Bar(x=platforms_df.index, y=platforms_df['Global_Sales'], name='Global Sales')
    _trace1 = go.Bar(x=platforms_df.index, y=platforms_df['Number_of_Games'], name='Number of games released')
    _fig = go.Figure(data=[_trace0, _trace1], layout={'title': 'Market share by gaming platform'})
    iplot(_fig, show_link=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Box plot** — one `go.Box` trace per genre gives interactive box plots of critic scores by genre.
    """)
    return


@app.cell
def _(df, go, iplot):
    data = [go.Box(y=df[df.Genre == genre].Critic_Score, name=genre)
            for genre in df.Genre.unique()]
    iplot(data, show_link=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Pick by need, not habit.** pandas `.plot()` for fast throwaway exploration → Seaborn for polished statistical plots in one line → Plotly when interactivity (hover/zoom/toggle) actually pays off.
    - **They stack.** `.plot()` is pandas calling Matplotlib; Seaborn wraps Matplotlib with better defaults. You can always drop down a level to fix a detail.
    - **Seaborn one-liners replace whole Matplotlib blocks:** `jointplot` = scatter + two marginals; `heatmap` = a pivot table rendered as color; `boxplot` = a distribution per category.
    - **Plotly thinks in traces.** Build a list of `go.Scatter` / `go.Bar` / `go.Box` traces and hand it to `iplot`. The cost is a heavier notebook and a browser dependency.
    - **Reshape before you plot.** Nearly every cell here is a `groupby` / `pivot_table` feeding a one-line plot call — the data prep is the real work, the chart is the easy part.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Part 2 maps the Part 1 taxonomy onto **three libraries stacked by abstraction**. pandas `DataFrame.plot()` is the shortest path from a reshaped frame to a chart — no extra imports, switch types with `kind=`. Seaborn sits a level up: nicer defaults and one-liners (`histplot`, `jointplot`, `boxplot`, `heatmap`) for plots that would take a block of Matplotlib. Plotly serves a different need — interactive, browser-rendered figures built from traces, worth it when hovering and zooming genuinely help. Knowing which to grab, and remembering that the real work is the `groupby`/`pivot_table` that feeds them, is the whole lesson.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **Matplotlib** = the base plotting library the others are built on
    - **`DataFrame.plot()`** = pandas' built-in Matplotlib wrapper, the quickest path to a chart
        - **`kind=`** = the `.plot()` argument selecting chart type (`"line"`, `"bar"`, `"box"`, …)
    - **Seaborn** = higher-level statistical wrapper over Matplotlib, with better defaults
        - **`histplot()`** = histogram of one numeric variable, optional KDE overlay
        - **`jointplot()`** = scatter of two numeric variables plus their marginal distributions
        - **`boxplot()`** = one box per category, comparing a numeric distribution across them
        - **`heatmap()`** = a numeric value over a grid of two categoricals, drawn as colored cells
    - **Plotly** = interactive plotting library rendered in the browser (hover, zoom, toggle)
        - **trace** = one data series in a Plotly figure (`go.Scatter`, `go.Bar`, `go.Box`)
        - **`iplot()`** = renders a list of traces inline in the notebook
    - **`sns.set()`** = applies Seaborn's styling to every plot in the session
    """)
    return


if __name__ == "__main__":
    app.run()
