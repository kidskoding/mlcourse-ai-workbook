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
    # Assignment #10: Gradient boosting

    **Submit via the [Kaggle Inclass competition](https://www.kaggle.com/c/flight-delays-spring-2018)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Your task is to beat at least 2 benchmarks in this [Kaggle Inclass competition](https://www.kaggle.com/c/flight-delays-spring-2018). Here, you won’t be provided with detailed instructions. We only give you a brief description of how the second benchmark was achieved using Xgboost. Hopefully, at this stage of the course, it's enough for you to take a quick look at the data in order to understand that this is the type of task where gradient boosting will perform well. Most likely it will be Xgboost; however, we have plenty of categorical features here.
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    return XGBClassifier, pd, roc_auc_score, train_test_split


@app.cell
def _():
    # for Jupyter-book, we copy data from GitHub, locally, to save Internet traffic,
    # you can specify the data/ folder from the root of your cloned
    # https://github.com/Yorko/mlcourse.ai repo, to save Internet traffic
    DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    return (DATA_PATH,)


@app.cell
def _(DATA_PATH, pd):
    train = pd.read_csv(DATA_PATH + "flight_delays_train.csv")
    test = pd.read_csv(DATA_PATH + "flight_delays_test.csv")
    return test, train


@app.cell
def _(train):
    train.head()
    return


@app.cell
def _(test):
    test.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given flight departure time, carrier's code, departure airport, destination location, and flight distance, you have to predict departure delay for more than 15 minutes. As the simplest benchmark, let's take an Xgboost classifier and two features that are easiest to take: DepTime and Distance. Such a model results in 0.68202 on the LB.
    """)
    return


@app.cell
def _(test, train, train_test_split):
    X_train = train[["Distance", "DepTime"]].values
    y_train = train["dep_delayed_15min"].map({"Y": 1, "N": 0}).values
    X_test = test[["Distance", "DepTime"]].values

    X_train_part, X_valid, y_train_part, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=17
    )
    return (
        X_test,
        X_train,
        X_train_part,
        X_valid,
        y_train,
        y_train_part,
        y_valid,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll train Xgboost with default parameters on a part of the data and estimate holdout ROC AUC.
    """)
    return


@app.cell
def _(
    XGBClassifier,
    X_train_part,
    X_valid,
    roc_auc_score,
    y_train_part,
    y_valid,
):
    xgb_model = XGBClassifier(seed=17)

    xgb_model.fit(X_train_part, y_train_part)
    xgb_valid_pred = xgb_model.predict_proba(X_valid)[:, 1]

    roc_auc_score(y_valid, xgb_valid_pred)
    return (xgb_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we do the same with the whole training set, make predictions to the test set and form a submission file. This is how you beat the first benchmark.
    """)
    return


@app.cell
def _(X_test, X_train, pd, xgb_model, y_train):
    xgb_model.fit(X_train, y_train)
    xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]

    pd.Series(xgb_test_pred, name="dep_delayed_15min").to_csv(
        "xgb_2feat.csv", index_label="id", header=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The second benchmark in the leaderboard was achieved as follows:

    - Features `Distance` and `DepTime` were taken unchanged
    - A feature `Flight` was created from features `Origin` and `Dest`
    - Features `Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier` and `Flight` were transformed with OHE (`LabelBinarizer`)
    - Logistic regression and gradient boosting (xgboost) were trained. Xgboost hyperparameters were tuned via cross-validation. First, the hyperparameters responsible for model complexity were optimized, then the number of trees was fixed at 500 and learning rate was tuned.
    - Predicted probabilities were made via cross-validation using `cross_val_predict`. A linear mixture of logistic regression and gradient boosting predictions was set in the form $w_1 * p_{logit} + (1 - w_1) * p_{xgb}$, where $p_{logit}$ is a probability of class 1, predicted by logistic regression, and $p_{xgb}$ – the same for xgboost. $w_1$ weight was selected manually.
    - A similar combination of predictions was made for test set.

    Following the same steps is not mandatory. That’s just a description of how the result was achieved by the author of this assignment. Perhaps you might not want to follow the same steps, and instead, let’s say, add a couple of good features and train a random forest of a thousand trees.

    Good luck!
    """)
    return


if __name__ == "__main__":
    app.run()
