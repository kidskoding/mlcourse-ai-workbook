# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
# ]
# ///

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
    # Assignment #5: Logistic Regression and Random Forest in the credit scoring problem

    **Submit answers [here](https://docs.google.com/forms/d/1gKt0DA4So8ohKAHZNCk58ezvg7K_tik26d9QND7WC6M)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 1.** There are 5 jurors in a courtroom. Each of them can correctly identify the guilt of the defendant with 70% probability, independent of one another. What is the probability that the jurors will jointly reach the correct verdict if the final decision is by majority vote?

    1. 70.00%
    2. 83.20%
    3. 83.70%
    4. 87.50%

    ## Credit scoring problem setup

    ### Problem

    Predict whether the customer will repay their credit within 90 days. This is a binary classification problem; we will assign customers into good or bad categories based on our prediction.

    ### Data description

    | Feature | Variable Type | Value Type | Description |
    |:--------|:--------------|:-----------|:------------|
    | age | Input Feature | integer | Customer age |
    | DebtRatio | Input Feature | real | Total monthly loan payments (loan, alimony, etc.) / Total monthly income percentage |
    | NumberOfTime30-59DaysPastDueNotWorse | Input Feature | integer | The number of cases when client has overdue 30-59 days (not worse) on other loans during the last 2 years |
    | NumberOfTimes90DaysLate | Input Feature | integer | Number of cases when customer had 90+dpd overdue on other credits |
    | NumberOfTime60-89DaysPastDueNotWorse | Input Feature | integer | Number of cases when customer has 60-89dpd (not worse) during the last 2 years |
    | NumberOfDependents | Input Feature | integer | The number of customer dependents |
    | SeriousDlqin2yrs | Target Variable | binary: <br>0 or 1 | Customer hasn't paid the loan debt within 90 days |
    """)
    return


@app.cell
def _():
    # Disable warnings in Anaconda
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd

    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()
    return (pd,)


@app.cell
def _():
    from matplotlib import rcParams

    rcParams["figure.figsize"] = 11, 8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's write the function that will replace *NaN* values with the median for each column.
    """)
    return


@app.function
def fill_nan(table):
    for col in table.columns:
        table[col] = table[col].fillna(table[col].median())
    return table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, read the data:
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
    data = pd.read_csv(DATA_PATH + "credit_scoring_sample.csv", sep=";")
    data.head()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the variable types:
    """)
    return


@app.cell
def _(data):
    data.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check the class balance:
    """)
    return


@app.cell
def _(data):
    ax = data["SeriousDlqin2yrs"].hist(orientation="horizontal", color="red")
    ax.set_xlabel("number_of_observations")
    ax.set_ylabel("unique_value")
    ax.set_title("Target distribution")

    print("Distribution of the target:")
    data["SeriousDlqin2yrs"].value_counts() / data.shape[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Separate the input variable names by excluding the target:
    """)
    return


@app.cell
def _(data):
    independent_columns_names = [x for x in data if x != "SeriousDlqin2yrs"]
    independent_columns_names
    return (independent_columns_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Apply the function to replace *NaN* values:
    """)
    return


@app.cell
def _(data):
    table = fill_nan(data)
    return (table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Separate the target variable and input features:
    """)
    return


@app.cell
def _(independent_columns_names, table):
    X = table[independent_columns_names]
    y = table["SeriousDlqin2yrs"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bootstrapping

    **Question 2.** Make an interval estimate of the average age for the customers who delayed repayment at the 90% confidence level. Use the example from the article as reference, if needed. Also, use `np.random.seed(0)` as before. What is the resulting interval estimate?

    1. 52.59 – 52.86
    2. 45.71 – 46.13
    3. 45.68 – 46.17
    4. 52.56 – 52.88
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logistic regression

    Let's set up to use logistic regression:
    """)
    return


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    return LogisticRegression, StratifiedKFold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we will create a `LogisticRegression` model and use `class_weight='balanced'` to make up for our unbalanced classes.
    """)
    return


@app.cell
def _(LogisticRegression):
    lr = LogisticRegression(random_state=5, class_weight="balanced")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's try to find the best regularization coefficient, which is the coefficient `C` for logistic regression.
    """)
    return


@app.cell
def _():
    _parameters = {"C": (0.0001, 0.001, 0.01, 0.1, 1, 10)}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to find the optimal value of `C`, let's apply stratified 5-fold validation and look at the *ROC AUC* against different values of the parameter `C`. Use the `StratifiedKFold` function for this:
    """)
    return


@app.cell
def _(StratifiedKFold):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 3.** Perform a *Grid Search* with the scoring metric "roc_auc" for the parameter `C`. Which value of the parameter `C` is optimal?

    1. 0.0001
    2. 0.001
    3. 0.01
    4. 0.1
    5. 1
    6. 10
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 4.** Can we consider the best model stable? The model is *stable* if the standard deviation on validation is less than 0.5%. Save the *ROC AUC* value of the best model; it will be useful for the following tasks.

    1. Yes
    2. No
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature importance

    **Question 5.** *Feature importance* is defined by the absolute value of its corresponding coefficient. First, you need to normalize all of the feature values so that it will be valid to compare them. What is the most important feature for the best logistic regression model?

    1. age
    2. NumberOfTime30-59DaysPastDueNotWorse
    3. DebtRatio
    4. NumberOfTimes90DaysLate
    5. NumberOfTime60-89DaysPastDueNotWorse
    6. MonthlyIncome
    7. NumberOfDependents
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 6.** Calculate how much `DebtRatio` affects our prediction using the [softmax function](https://en.wikipedia.org/wiki/Softmax_function). What is its value?

    1. 0.38
    2. -0.02
    3. 0.11
    4. 0.24
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 7.** Let's see how we can interpret the impact of our features. For this, recalculate the logistic regression with absolute values, that is without scaling. Next, modify the customer's age by adding 20 years, keeping the other features unchanged. How many times will the chance that the customer will not repay their debt increase? You can find an example of the theoretical calculation [here](https://www.unm.edu/~schrader/biostat/bio2/Spr06/lec11.pdf).

    1. -0.01
    2. 0.70
    3. 8.32
    4. 0.66
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest

    Import the Random Forest classifier:
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import RandomForestClassifier

    return (RandomForestClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Initialize Random Forest with 100 trees and balance target classes:
    """)
    return


@app.cell
def _(RandomForestClassifier):
    rf = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will search for the best parameters among the following values:
    """)
    return


@app.cell
def _():
    _parameters = {
        "max_features": [1, 2, 4],
        "min_samples_leaf": [3, 5, 7, 9],
        "max_depth": [5, 10, 15],
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Also, we will use the stratified k-fold validation again. You should still have the `skf` variable.

    **Question 8.** How much higher is the *ROC AUC* of the best random forest model than that of the best logistic regression on validation? Select the closest answer.

    1. 0.04
    2. 0.03
    3. 0.02
    4. 0.01
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 9.** What feature has the weakest impact in the Random Forest model?

    1. age
    2. NumberOfTime30-59DaysPastDueNotWorse
    3. DebtRatio
    4. NumberOfTimes90DaysLate
    5. NumberOfTime60-89DaysPastDueNotWorse
    6. MonthlyIncome
    7. NumberOfDependents
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 10.** What is the most significant advantage of using *Logistic Regression* versus *Random Forest* for this problem?

    1. Spent less time for model fitting;
    2. Fewer variables to iterate;
    3. Feature interpretability;
    4. Linear properties of the algorithm.

    ## Bagging

    Import modules and set up the parameters for bagging:
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import RandomizedSearchCV, cross_val_score

    _parameters = {
        "max_features": [2, 3, 4],
        "max_samples": [0.5, 0.7, 0.9],
        "base_estimator__C": [0.0001, 0.001, 0.01, 1, 10, 100],
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 11.** Fit a bagging classifier with `random_state=42`. For the base classifiers, use 100 logistic regressors and use `RandomizedSearchCV` instead of `GridSearchCV`. It will take a lot of time to iterate over all 54 variants, so set the maximum number of iterations for `RandomizedSearchCV` to 20. Don't forget to set the parameters `cv` and `random_state=1`. What is the best *ROC AUC* you achieve?

    1. 80.75%
    2. 80.12%
    3. 79.62%
    4. 76.50%
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 12.** Give an interpretation of the best parameters for bagging. Why are these values of `max_features` and `max_samples` the best?

    1. For bagging it's important to use as few features as possible;
    2. Bagging works better on small samples;
    3. Less correlation between single models;
    4. The higher the number of features, the lower the loss of information.
    """)
    return


if __name__ == "__main__":
    app.run()
