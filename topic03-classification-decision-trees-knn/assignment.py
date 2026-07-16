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
    # Assignment #3: Decision trees with a toy task and the UCI Adult dataset

    **Submit answers [here](https://docs.google.com/forms/d/1wfWYYoqXTkZNOPy1wpewACXaj2MZjBdLOL58htGWYBA)**
    """)
    return


@app.cell
def _():
    import collections
    from io import StringIO

    import numpy as np
    import pandas as pd
    import pydotplus  # pip install pydotplus
    import seaborn as sns
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier, export_graphviz

    from matplotlib import pyplot as plt

    plt.rcParams["figure.figsize"] = (10, 8)
    return GridSearchCV, LabelEncoder, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1. Toy dataset "Will They? Won't They?"

        We'll go through a toy example of binary classification - Person A is deciding whether they will go on a second date with Person B. It will depend on their looks, eloquence, alcohol consumption (only for example), and how much money was spent on the first date.

    ### Creating the dataset
    """)
    return


@app.cell
def _(pd):
    # Create dataframe with dummy variables
    def create_df(dic, feature_list):
        out = pd.DataFrame(dic)
        out = pd.concat([out, pd.get_dummies(out[feature_list])], axis=1)
        out.drop(feature_list, axis=1, inplace=True)
        return out

    # Some feature values are present in train and absent in test and vice-versa.
    def intersect_features(train, test):
        common_feat = list(set(train.keys()) & set(test.keys()))
        return train[common_feat], test[common_feat]

    return create_df, intersect_features


@app.cell
def _():
    features = ["Looks", "Alcoholic_beverage", "Eloquence", "Money_spent"]
    return (features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training data
    """)
    return


@app.cell
def _(LabelEncoder, create_df, features):
    df_train = {}
    df_train["Looks"] = [
        "handsome",
        "handsome",
        "handsome",
        "repulsive",
        "repulsive",
        "repulsive",
        "handsome",
    ]
    df_train["Alcoholic_beverage"] = ["yes", "yes", "no", "no", "yes", "yes", "yes"]
    df_train["Eloquence"] = [
        "high",
        "low",
        "average",
        "average",
        "low",
        "high",
        "average",
    ]
    df_train["Money_spent"] = [
        "lots",
        "little",
        "lots",
        "little",
        "lots",
        "lots",
        "lots",
    ]
    df_train["Will_go"] = LabelEncoder().fit_transform(
        ["+", "-", "+", "-", "-", "+", "+"]
    )

    df_train = create_df(df_train, features)
    df_train
    return (df_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test data
    """)
    return


@app.cell
def _(create_df, features):
    df_test = {}
    df_test["Looks"] = ["handsome", "handsome", "repulsive"]
    df_test["Alcoholic_beverage"] = ["no", "yes", "yes"]
    df_test["Eloquence"] = ["average", "high", "average"]
    df_test["Money_spent"] = ["lots", "little", "lots"]
    df_test = create_df(df_test, features)
    df_test
    return (df_test,)


@app.cell
def _(df_test, df_train, intersect_features):
    # Some feature values are present in train and absent in test and vice-versa.
    y = df_train["Will_go"]
    df_train_1, df_test_1 = intersect_features(train=df_train, test=df_test)
    df_train_1
    return (df_test_1,)


@app.cell
def _(df_test_1):
    df_test_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Draw a decision tree (by hand or in any graphics editor) for this dataset. Optionally you can also implement tree construction and draw it here.

    1\. What is the entropy $S_0$ of the initial system? By system states, we mean values of the binary feature "Will_go" - 0 or 1 - two states in total.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2\. Let's split the data by the feature "Looks_handsome". What is the entropy $S_1$ of the left group - the one with "Looks_handsome". What is the entropy $S_2$ in the opposite group? What is the information gain (IG) if we consider such a split?
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train a decision tree using sklearn on the training data. You may choose any depth for the tree.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Additional: display the resulting tree using graphviz. You can use pydot or a web-service, e.g. [this one](https://onlineconvertfree.com/convert-format/dot-to-png/).
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2. Functions for calculating entropy and information gain.

    Consider the following warm-up example: we have 9 blue balls and 11 yellow balls. Let ball have label **1** if it is blue, **0** otherwise.
    """)
    return


@app.cell
def _():
    balls = [1 for i in range(9)] + [0 for i in range(11)]
    return (balls,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/mlcourse_ai_jupyter_book/_static/img/decision_tree3.png">

    Next split the balls into two groups:

    <img src="https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/mlcourse_ai_jupyter_book/_static/img/decision_tree4.png">
    """)
    return


@app.cell
def _():
    # two groups
    balls_left = [1 for i in range(8)] + [0 for i in range(5)]  # 8 blue and 5 yellow
    balls_right = [1 for i in range(1)] + [0 for i in range(6)]  # 1 blue and 6 yellow
    return balls_left, balls_right


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implement a function to calculate the Shannon Entropy
    """)
    return


@app.function
def entropy(a_list):
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tests
    """)
    return


@app.cell
def _(balls, balls_left, balls_right):
    print(entropy(balls))  # 9 blue и 11 yellow
    print(entropy(balls_left))  # 8 blue и 5 yellow
    print(entropy(balls_right))  # 1 blue и 6 yellow
    print(entropy([1, 2, 3, 4, 5, 6]))  # entropy of a fair 6-sided die
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3\. What is the entropy of the state given by the list **balls_left**?

    4\. What is the entropy of a fair die? (where we look at a die as a system with 6 equally probable states)?
    """)
    return


@app.function
# information gain calculation
def information_gain(root, left, right):
    """root - initial data, left and right - two partitions of initial data"""

    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    5\. What is the information gain from splitting the initial dataset into **balls_left** and **balls_right** ?
    """)
    return


@app.function
def information_gains(X, y):
    """Outputs information gain when splitting with each feature"""

    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional:
    - Implement a decision tree building algorithm by calling `information_gains` recursively
    - Plot the resulting tree

    ## Part 3. The "Adult" dataset

    **Dataset description:**

    [Dataset](https://archive.ics.uci.edu/dataset/2/adult) UCI Adult (no need to download it, we have a copy in the course repository): classify people using demographic data - whether they earn more than \$50,000 per year or not.

    Feature descriptions:

    - **Age** – continuous feature
    - **Workclass** –  continuous feature
    - **fnlwgt** – final weight of object, continuous feature
    - **Education** –  categorical feature
    - **Education_Num** – number of years of education, continuous feature
    - **Martial_Status** –  categorical feature
    - **Occupation** –  categorical feature
    - **Relationship** – categorical feature
    - **Race** – categorical feature
    - **Sex** – categorical feature
    - **Capital_Gain** – continuous feature
    - **Capital_Loss** – continuous feature
    - **Hours_per_week** – continuous feature
    - **Country** – categorical feature

    **Target** – earnings level, categorical (binary) feature.

    **Reading train and test data**
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
    data_train = pd.read_csv(DATA_PATH + "adult_train.csv", sep=";")
    return (data_train,)


@app.cell
def _(data_train):
    data_train.tail()
    return


@app.cell
def _(DATA_PATH, pd):
    data_test = pd.read_csv(DATA_PATH + "adult_test.csv", sep=";")
    return (data_test,)


@app.cell
def _(data_test):
    data_test.tail()
    return


@app.cell
def _(data_test, data_train):
    # necessary to remove rows with incorrect labels in test dataset
    data_test_1 = data_test[
        (data_test["Target"] == " >50K.") | (data_test["Target"] == " <=50K.")
    ]
    data_train.loc[data_train["Target"] == " <=50K", "Target"] = 0
    data_train.loc[data_train["Target"] == " >50K", "Target"] = 1
    data_test_1.loc[data_test_1["Target"] == " <=50K.", "Target"] = 0
    # encode target variable as integer
    data_test_1.loc[data_test_1["Target"] == " >50K.", "Target"] = 1
    return (data_test_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Primary data analysis**
    """)
    return


@app.cell
def _(data_test_1):
    data_test_1.describe(include="all").T
    return


@app.cell
def _(data_train):
    data_train["Target"].value_counts()
    return


@app.cell
def _(data_train, plt):
    fig = plt.figure(figsize=(25, 15))
    cols = 5
    rows = int(data_train.shape[1] / cols)
    for i, column in enumerate(data_train.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if data_train.dtypes[column] == object:
            data_train[column].value_counts().plot(kind="bar", axes=ax)
        else:
            data_train[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Checking data types**
    """)
    return


@app.cell
def _(data_train):
    data_train.dtypes
    return


@app.cell
def _(data_test_1):
    data_test_1.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we see, in the test data, age is treated as type **object**. We need to fix this.
    """)
    return


@app.cell
def _(data_test_1):
    data_test_1["Age"] = data_test_1["Age"].astype(int)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Also we'll cast all **float** features to **int** type to keep types consistent between our train and test data.
    """)
    return


@app.cell
def _(data_test_1):
    data_test_1["fnlwgt"] = data_test_1["fnlwgt"].astype(int)
    data_test_1["Education_Num"] = data_test_1["Education_Num"].astype(int)
    data_test_1["Capital_Gain"] = data_test_1["Capital_Gain"].astype(int)
    data_test_1["Capital_Loss"] = data_test_1["Capital_Loss"].astype(int)
    data_test_1["Hours_per_week"] = data_test_1["Hours_per_week"].astype(int)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Save targets separately.
    """)
    return


@app.cell
def _(data_test_1, data_train):
    y_train = data_train.pop("Target")
    y_test = data_test_1.pop("Target")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Fill in missing data for continuous features with their median values, for categorical features with their mode.**
    """)
    return


@app.cell
def _(data_train):
    # choose categorical and continuous features from data

    categorical_columns = [
        c for c in data_train.columns if data_train[c].dtype.name == "object"
    ]
    numerical_columns = [
        c for c in data_train.columns if data_train[c].dtype.name != "object"
    ]

    print("categorical_columns:", categorical_columns)
    print("numerical_columns:", numerical_columns)
    return categorical_columns, numerical_columns


@app.cell
def _(data_train):
    # we see some missing values
    data_train.info()
    return


@app.cell
def _(categorical_columns, data_test_1, data_train, numerical_columns):
    # fill missing data
    for c in categorical_columns:
        data_train[c] = data_train[c].fillna(data_train[c].mode()[0])
        data_test_1[c] = data_test_1[c].fillna(data_train[c].mode()[0])
    for c in numerical_columns:
        data_train[c] = data_train[c].fillna(data_train[c].median())
        data_test_1[c] = data_test_1[c].fillna(data_train[c].median())
    return


@app.cell
def _(data_train):
    # no more missing values
    data_train.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll dummy code some categorical features: **Workclass**, **Education**, **Martial_Status**, **Occupation**, **Relationship**, **Race**, **Sex**, **Country**. It can be done via pandas method **get_dummies**
    """)
    return


@app.cell
def _(categorical_columns, data_test_1, data_train, numerical_columns, pd):
    data_train_1 = pd.concat(
        [
            data_train[numerical_columns],
            pd.get_dummies(data_train[categorical_columns]),
        ],
        axis=1,
    )
    data_test_2 = pd.concat(
        [
            data_test_1[numerical_columns],
            pd.get_dummies(data_test_1[categorical_columns]),
        ],
        axis=1,
    )
    return data_test_2, data_train_1


@app.cell
def _(data_test_2, data_train_1):
    set(data_train_1.columns) - set(data_test_2.columns)
    return


@app.cell
def _(data_test_2, data_train_1):
    (data_train_1.shape, data_test_2.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **There is no Holland in the test data. Create new zero-valued feature.**
    """)
    return


@app.cell
def _(data_test_2):
    data_test_2["Country_ Holand-Netherlands"] = 0
    return


@app.cell
def _(data_test_2, data_train_1):
    set(data_train_1.columns) - set(data_test_2.columns)
    return


@app.cell
def _(data_train_1):
    data_train_1.head(2)
    return


@app.cell
def _(data_test_2):
    data_test_2.head(2)
    return


@app.cell
def _(data_test_2, data_train_1):
    X_train = data_train_1
    X_test = data_test_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Decision tree without parameter tuning

    Train a decision tree **(DecisionTreeClassifier)** with a maximum depth of 3, and evaluate the accuracy metric on the test data. Use parameter **random_state = 17** for results reproducibility.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # tree =
    # tree.fit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make a prediction with the trained model on the test data.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # tree_predictions = tree.predict
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # accuracy_score
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    6\. What is the test set accuracy of a decision tree with maximum tree depth of 3 and **random_state = 17**?

    ### 3.2 Decision tree with parameter tuning

    Train a decision tree **(DecisionTreeClassifier, random_state = 17).** Find the optimal maximum depth using 5-fold cross-validation **(GridSearchCV)**.
    """)
    return


@app.cell
def _(GridSearchCV):
    tree_params = {"max_depth": range(2, 11)}

    locally_best_tree = GridSearchCV

    locally_best_tree.fit
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Train a decision tree with maximum depth of 9 (it is the best **max_depth** in my case), and compute the test set accuracy. Use parameter **random_state = 17** for reproducibility.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # tuned_tree =
    # tuned_tree.fit
    # tuned_tree_predictions = tuned_tree.predict
    # accuracy_score
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    7\. What is the test set accuracy of a decision tree with maximum depth of 9 and **random_state = 17**?

    ### 3.3 (Optional) Random forest without parameter tuning

    Let's take a sneak peek of upcoming lectures and try to use a random forest for our task. For now, you can imagine a random forest as a bunch of decision trees, trained on slightly different subsets of the training data.

    Train a random forest **(RandomForestClassifier)**. Set the number of trees to 100 and use **random_state = 17**.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    # rf =
    # rf.fit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make predictions for the test data and assess accuracy.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 (Optional) Random forest with parameter tuning

    Train a random forest **(RandomForestClassifier)**. Tune the maximum depth and maximum number of features for each tree using **GridSearchCV**.
    """)
    return


@app.cell
def _():
    # forest_params = {'max_depth': range(10, 21),
    #                 'max_features': range(5, 105, 20)}

    # locally_best_forest = GridSearchCV

    # locally_best_forest.fit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make predictions for the test data and assess accuracy.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


if __name__ == "__main__":
    app.run()
