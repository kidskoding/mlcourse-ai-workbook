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
    # Assignment #9: Time series analysis

    **Submit answers [here](https://docs.google.com/forms/d/1UYQ_WYSpsV3VSlZAzhSN_YXmyjV7YlTP8EYMg8M8SoM)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Goal**: To use Prophet and ARIMA to analyze the number of views for a Wikipedia [page](https://en.wikipedia.org/wiki/Machine_learning) on Machine Learning
    """)
    return


@app.cell
def _():
    import os
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import requests
    from plotly import __version__
    from plotly import graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
    from IPython.display import display, IFrame

    print(__version__)  # need 1.9.0 or greater
    init_notebook_mode(connected=True)
    return IFrame, display, go, pd, plot


@app.cell
def _(IFrame, display, go, plot):
    def plotly_df(df, title="", width=800, height=500):
        """Visualize all the dataframe columns as line plots."""
        common_kw = dict(x=df.index, mode="lines")
        data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
        layout = dict(title=title)
        fig = dict(data=data, layout=layout)

        # in a Jupyter Notebook, the following should work
        # iplot(fig, show_link=False)

        # in a Jupyter Book, we save a plot offline and then render it with IFrame
        plot_path = f"../../_static/plotly_htmls/{title}.html".replace(" ", "_")
        plot(fig, filename=plot_path, show_link=False, auto_open=False)
        display(IFrame(plot_path, width=width, height=height))

    return (plotly_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data preparation
    """)
    return


@app.cell
def _():
    # for Jupyter-book, we copy data from GitHub, locally, to save Internet traffic,
    # you can specify the data/ folder from the root of your cloned
    # https://github.com/Yorko/mlcourse.ai repo, to save Internet traffic
    DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    return (DATA_PATH,)


@app.cell
def _(DATA_PATH, pd):
    df = pd.read_csv(DATA_PATH + "wiki_machine_learning.csv", sep=" ")
    df = df[df["count"] != 0]
    df.head()
    return (df,)


@app.cell
def _(df):
    df.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predicting with FB Prophet
    We will train on the first 5 months and predict the number of trips for June.
    """)
    return


@app.cell
def _(df, pd):
    df.date = pd.to_datetime(df.date)
    return


@app.cell
def _(df, plotly_df):
    plotly_df(df=df.set_index("date")[["count"]], title="assign9_plot")
    return


@app.cell
def _():
    from prophet import Prophet

    return


@app.cell
def _(df):
    predictions = 30
    df_1 = df[["date", "count"]]
    df_1.columns = ["ds", "y"]
    df_1.tail()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **<font color='red'>Question 1:</font>** What is the prediction of the number of views of the wiki page on January 20? Round to the nearest integer.

    - 4947
    - 3426
    - 5229
    - 2744
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Estimate the quality of the prediction with the last 30 points.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **<font color='red'>Question 2:</font> What is MAPE equal to?**

    - 34.5
    - 42.42
    - 5.39
    - 65.91
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **<font color='red'>Question 3:</font> What is MAE equal to?**

    - 355
    - 4007
    - 600
    - 903
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predicting with ARIMA
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from scipy import stats

    plt.rcParams["figure.figsize"] = (15, 10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **<font color='red'>Question 4:</font> Let's verify the stationarity of the series using the Dickey-Fuller test. Is the series stationary? What is the p-value?**

    - Series is stationary, p_value = 0.107
    - Series is not stationary, p_value = 0.107
    - Series is stationary, p_value = 0.001
    - Series is not stationary, p_value = 0.001
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Next, we turn to the construction of the SARIMAX model (`sm.tsa.statespace.SARIMAX`).<br> <font color='red'>Question 5:</font> What parameters are the best for the model according to the `AIC` criterion?**

    - D = 1, d = 0, Q = 0, q = 2, P = 3, p = 1
    - D = 2, d = 1, Q = 1, q = 2, P = 3, p = 1
    - D = 1, d = 1, Q = 1, q = 2, P = 3, p = 1
    - D = 0, d = 0, Q = 0, q = 2, P = 3, p = 1
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


if __name__ == "__main__":
    app.run()
