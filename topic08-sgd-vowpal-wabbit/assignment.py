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
    # Assignment #8: Implementation of online regressor

    **No submission — a built-in test checks your implementation.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Goal**: implement a regressor trained with stochastic gradient descent (SGD). Fill in the missing code. If you do everything right, you'll pass a simple embedded test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear regression and Stochastic Gradient Descent
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.base import BaseEstimator
    from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    from matplotlib import pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler

    return BaseEstimator, StandardScaler, pd, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Implement class `SGDRegressor`. Specification:
    - class is inherited from `sklearn.base.BaseEstimator`
    - constructor takes parameters `eta` – gradient step ($10^{-3}$ by default) and `n_epochs` – dataset pass count (3 by default)
    - Constructor also creates `mse_` and `weights_` lists to track mean squared error and weight vector during gradient descent iterations
    - Class has `fit` and `predict` methods
    - The `fit` method takes matrix `X` and vector `y` (`numpy.array` objects) as parameters, appends a column of ones to  `X` on the left side, initializes weight vector `w` with **zeros** and then makes `n_epochs` iterations of weight updates (you may refer to this [article](https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-8-vowpal-wabbit-fast-learning-with-gigabytes-of-data-60f750086237) for details), and for every iteration logs mean squared error and weight vector `w` in the corresponding lists created in the constructor.
    - Additionally, the `fit` method will create `w_` variable to store weights which produce minimal mean squared error
    - The `fit` method returns the current instance of the `SGDRegressor` class, i.e. `self`
    - The `predict` method takes the `X` matrix, adds a column of ones to the left side and returns a prediction vector, using weight vector `w_`, created by the `fit` method.
    """)
    return


@app.cell
def _(BaseEstimator):
    class SGDRegressor(BaseEstimator):
        # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
        def __init__(self):
            pass

        def fit(self, X, y):
            pass

        def predict(self, X):
            pass

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's test out the algorithm on height/weight data. We will predict heights (in inches) based on weights (in lbs).
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
    data_demo = pd.read_csv(DATA_PATH + "weights_heights.csv")
    return (data_demo,)


@app.cell
def _(data_demo, plt):
    plt.scatter(data_demo["Weight"], data_demo["Height"])
    plt.xlabel("Weight (lbs)")
    plt.ylabel("Height (Inch)")
    plt.grid()
    return


@app.cell
def _(data_demo):
    X, y = data_demo["Weight"].values, data_demo["Height"].values
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Perform train/test split and scale data.
    """)
    return


@app.cell
def _(X, train_test_split, y):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, random_state=17
    )
    return X_train, X_valid


@app.cell
def _(StandardScaler, X_train, X_valid):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape([-1, 1]))
    X_valid_scaled = scaler.transform(X_valid.reshape([-1, 1]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Train created `SGDRegressor` with `(X_train_scaled, y_train)` data. Leave default parameter values for now.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Draw a chart with training process  – dependency of mean squared error from the i-th SGD iteration number.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Print the minimal value of mean squared error and the best weights vector.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Draw chart of model weights ($w_0$ and $w_1$) behavior during training.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make a prediction for hold-out  set `(X_valid_scaled, y_valid)` and check MSE value.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    sgd_holdout_mse = 10
    return (sgd_holdout_mse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Do the same thing for `LinearRegression` class from `sklearn.linear_model`. Evaluate MSE for hold-out set.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    linreg_holdout_mse = 9
    return (linreg_holdout_mse,)


@app.cell
def _(linreg_holdout_mse, sgd_holdout_mse):
    try:
        assert (sgd_holdout_mse - linreg_holdout_mse) < 1e-4
        print("Correct!")
    except AssertionError:
        print(
            "Something's not good.\n Linreg's holdout MSE: {}"
            "\n SGD's holdout MSE: {}".format(linreg_holdout_mse, sgd_holdout_mse)
        )
    return


if __name__ == "__main__":
    app.run()
