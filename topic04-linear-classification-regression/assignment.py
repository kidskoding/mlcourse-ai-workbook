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
    # Assignment #4: Exploring OLS, Lasso and Random Forest in a regression task

    **Submit answers [here](https://docs.google.com/forms/d/1aHyK58W6oQmNaqEfvpLTpo6Cb0-ntnvJ18rZcvclkvw)**
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LassoCV, LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **We are working with UCI Wine quality dataset (no need to download it – it's already there, in course repo and in Kaggle Dataset).**
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
    data = pd.read_csv(DATA_PATH + "winequality-white.csv", sep=";")
    return (data,)


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data):
    data.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Separate the target feature, split data in 7:3 proportion (30% form a holdout set, use random_state=17), and preprocess data with `StandardScaler`.**
    """)
    return


@app.cell
def _():
    # y = None

    # X_train, X_holdout, y_train, y_holdout = train_test_split
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform
    # X_holdout_scaled = scaler.transform
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear regression

    **Train a simple linear regression model (Ordinary Least Squares).**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # linreg =
    # linreg.fit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **<font color='red'>Question 1:</font> What are mean squared errors of model predictions on train and holdout sets?**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # print("Mean squared error (train): %.3f"
    # print("Mean squared error (test): %.3f" %
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Sort features by their influence on the target feature (wine quality). Beware that both large positive and large negative coefficients mean large influence on target. It's handy to use `pandas.DataFrame` here.**

    **<font color='red'>Question 2:</font> Which feature does this linear regression model treat as the most influential on wine quality?**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # linreg_coef = pd.DataFrame
    # linreg_coef.sort_values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lasso regression

    **Train a LASSO model with $\alpha = 0.01$ (weak regularization) and scaled data. Again, set random_state=17.**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # lasso1 = Lasso
    # lasso1.fit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Which feature is the least informative in predicting wine quality, according to this LASSO model?**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # lasso1_coef = pd.DataFrame
    # lasso1_coef.sort_values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Train LassoCV with random_state=17 to choose the best value of $\alpha$ in 5-fold cross-validation.**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # alphas = np.logspace(-6, 2, 200)
    # lasso_cv = LassoCV
    # lasso_cv.fit
    return


@app.cell
def _():
    # lasso_cv.alpha_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **<font color='red'>Question 3:</font> Which feature is the least informative in predicting wine quality, according to the tuned LASSO model?**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # lasso_cv_coef = pd.DataFrame
    # lasso_cv_coef.sort_values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **<font color='red'>Question 4:</font> What are mean squared errors of tuned LASSO predictions on train and holdout sets?**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # print("Mean squared error (train): %.3f"
    # print("Mean squared error (test): %.3f" %
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest

    **Train a Random Forest with out-of-the-box parameters, setting only random_state to be 17.**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # forest = RandomForestRegressor
    # forest.fit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **<font color='red'>Question 5:</font> What are mean squared errors of RF model on the training set, in cross-validation (cross_val_score with scoring='neg_mean_squared_error' and other arguments left with default values) and on holdout set?**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # print("Mean squared error (train): %.3f" %
    # print("Mean squared error (cv): %.3f" %
    # print("Mean squared error (test): %.3f" %
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Tune the `max_features` and `max_depth` hyperparameters with GridSearchCV and again check mean cross-validation MSE and MSE on holdout set.**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # forest_params = {'max_depth': list(range(10, 25)),
    #                  'min_samples_leaf': list(range(1, 8)),
    #                  'max_features': list(range(6,12))}

    # locally_best_forest = GridSearchCV
    # locally_best_forest.fit
    return


@app.cell
def _():
    # locally_best_forest.best_params_, locally_best_forest.best_score_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **<font color='red'>Question 6:</font> What are mean squared errors of tuned RF model in cross-validation (cross_val_score with scoring='neg_mean_squared_error' and other arguments left with default values) and on holdout set?**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # print("Mean squared error (cv): %.3f" %
    # print("Mean squared error (test): %.3f" %
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Output RF's feature importance. Again, it's nice to present it as a DataFrame.**<br>
    **<font color='red'>Question 7:</font> What is the most important feature, according to the Random Forest model?**
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # rf_importance = pd.DataFrame
    # rf_importance.sort_values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Make conclusions about the performance of the 3 explored models in this particular prediction task.**
    """)
    return


if __name__ == "__main__":
    app.run()
