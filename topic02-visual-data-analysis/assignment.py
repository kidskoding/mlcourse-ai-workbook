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
    # Assignment #2: Analyzing cardiovascular disease data

    **Submit answers [here](https://docs.google.com/forms/d/13cE_tSIb6hsScQvvWUJeu1MEHE5L6vnxQUbDYpXsf24)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Predict the presence or absence of cardiovascular disease (CVD) using the patient examination results.

    ### Data description

    There are 3 types of input features:

    - *Objective*: factual information;
    - *Examination*: results of medical examination;
    - *Subjective*: information given by the patient.

    | Feature | Variable Type | Variable      | Value Type |
    |---------|--------------|---------------|------------|
    | Age | Objective Feature | age | int (days) |
    | Height | Objective Feature | height | int (cm) |
    | Weight | Objective Feature | weight | float (kg) |
    | Gender | Objective Feature | gender | categorical code |
    | Systolic blood pressure | Examination Feature | ap_hi | int |
    | Diastolic blood pressure | Examination Feature | ap_lo | int |
    | Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
    | Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
    | Smoking | Subjective Feature | smoke | binary |
    | Alcohol intake | Subjective Feature | alco | binary |
    | Physical activity | Subjective Feature | active | binary |
    | Presence or absence of cardiovascular disease | Target Variable | cardio | binary |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##  Part 1. Preliminary data analysis

    Initialize the environment
    """)
    return


@app.cell
def _():
    # Import all required modules
    # Disable warnings
    import warnings
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")
    import seaborn as sns

    sns.set()
    import matplotlib

    # Import plotting modules and set up
    import matplotlib.pyplot as plt

    # '%matplotlib inline' command supported automatically in marimo
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'retina'
    import matplotlib.ticker

    return pd, plt, sns


@app.cell
def _(sns):
    # Tune the visual settings for figures in `seaborn`
    sns.set_context(
        "notebook", font_scale=1.5, rc={"figure.figsize": (11, 8), "axes.titlesize": 18}
    )

    from matplotlib import rcParams

    rcParams["figure.figsize"] = 11, 8
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
    df = pd.read_csv(DATA_PATH + "mlbootcamp5_train.csv", sep=";")
    print("Dataset size: ", df.shape)
    df.head()
    return (df,)


@app.cell
def _(df, pd, plt, sns):
    _df_uniques = pd.melt(
        frame=df,
        value_vars=[
            "gender",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "cardio",
        ],
    )
    _df_uniques = (
        pd.DataFrame(_df_uniques.groupby(["variable", "value"])["value"].count())
        .sort_index(level=[0, 1])
        .rename(columns={"value": "count"})
        .reset_index()
    )
    sns.catplot(x="variable", y="count", hue="value", data=_df_uniques, kind="bar")
    plt.xticks(rotation="vertical")
    return


@app.cell
def _(df, pd, plt, sns):
    _df_uniques = pd.melt(
        frame=df,
        value_vars=["gender", "cholesterol", "gluc", "smoke", "alco", "active"],
        id_vars=["cardio"],
    )
    _df_uniques = (
        pd.DataFrame(
            _df_uniques.groupby(["variable", "value", "cardio"])["value"].count()
        )
        .sort_index(level=[0, 1])
        .rename(columns={"value": "count"})
        .reset_index()
    )
    sns.catplot(
        x="variable", y="count", hue="value", col="cardio", data=_df_uniques, kind="bar"
    )
    plt.xticks(rotation="vertical")
    return


@app.cell
def _(df):
    for c in df.columns:
        n = df[c].nunique()
        print(c)
        if n <= 3:
            print(n, sorted(df[c].value_counts().to_dict().items()))
        else:
            print(n)
        print(10 * "-")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1. Basic observations

    **Question 1.1. (1 point). How many men and women are present in this dataset? Values of the `gender` feature were not given (whether "1" stands for women or for men) – figure this out by analyzing height, making the assumption that men are taller on average.**

    1. 45530 women and 24470 men
    2. 45530 men and 24470 women
    3. 45470 women and 24530 men
    4. 45470 men and 24530 women

    **Answer**: [put correct answer choice # here]

    ---

    **Question 1.2. (1 point). Who more often report consuming alcohol – men or women?**
    1. women
    2. men

    **Answer**: [put correct answer choice # here]

    ---

    **Question 1.3. (1 point). What's the rounded difference between the percentages of smokers among men and women?**
    1. 4
    2. 16
    3. 20
    4. 24

    **Answer**: [put correct answer choice # here]

    ---

    **Question 1.4. (1 point). What's the rounded difference between median values of age (in months) for non-smokers and smokers? You'll need to figure out the units of feature `age` in this dataset.**

    1. 5
    2. 10
    3. 15
    4. 20

    **Answer**: [put correct answer choice # here]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2. Risk maps

    On the website for the European Society of Cardiology, a [SCORE scale](https://www.escardio.org/Education/Practice-Tools/CVD-prevention-toolbox/SCORE-Risk-Charts) is provided. It is used for calculating the risk of death from a cardiovascular decease in the next 10 years. Here it is:

    ![image.png](attachment:9865cf2a-27aa-40ee-88d1-5040ac65c7e9.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take a look at the upper-right rectangle, which shows a subset of smoking men aged from 60 to 65. (It's not obvious, but the values in the figure represent the upper bound).

    We see the value 9 in the lower-left corner of the rectangle and 47 in the upper-right. This means that, for people in this gender-age group whose systolic pressure is less than 120, the risk of a CVD is estimated to be 5 times lower than for those with the pressure in the interval [160,180).

    Let's calculate that same ratio using our data.

    Clarifications:
    - Calculate ``age_years`` feature – round age to the nearest number of years. For this task, select only the people of age 60 to 64, inclusive.
    - Cholesterol level categories differ between the figure and our dataset. The conversion for the ``cholesterol`` feature is as follows: 4 mmol/l $\rightarrow$ 1, 5-7 mmol/l $\rightarrow$ 2, 8 mmol/l $\rightarrow$ 3.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 1.5. (2 points). Calculate fractions of ill people (with CVD) in the two groups of people described in the task. What's the ratio of these two fractions?**

    1. 1
    2. 2
    3. 3
    4. 4

    **Answer**: [put correct answer choice # here]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3. Analyzing BMI

    Create a new feature – BMI ([Body Mass Index](https://en.wikipedia.org/wiki/Body_mass_index)). To do this, divide weight in kilograms by the square of the height in meters. Normal BMI values are said to be from 18.5 to 25.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 1.6. (2 points). Choose the correct statements:**

    1. Median BMI in the sample is within boundaries of normal values.
    2. Women's BMI is on average higher than men's.
    3. Healthy people have higher median BMI than ill people.
    4. In the segment of healthy and non-drinking men BMI is closer to the norm than in the segment of healthy and non-drinking women

    **Answer**: [put correct answer choice # here]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.4. Cleaning data

    ### Task:
    We can see that the data is not perfect. It contains "dirt" and inaccuracies. We'll see this better as we visualize the data.

    Filter out the following patient segments (we consider these as erroneous data)

    - diastolic pressure is higher than systolic.
    - height is strictly less than 2.5 percentile (Use `pd.Series.quantile` to compute this value. If you are not familiar with the function, please read the docs.)
    - height is strictly more than 97.5 percentile
    - weight is strictly less than 2.5 percentile
    - weight is strictly more than 97.5 percentile

    This is not everything that we can do to clean this data, but this is sufficient for now.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 1.7. (2 points). What percent of the original data (rounded) did we filter out in the previous step?**

    1. 8
    2. 9
    3. 10
    4. 11

    **Answer**: [put correct answer choice # here]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2. Visual data analysis <a class="tocSkip">

    ## 2.1. Correlation matrix visualization

    To understand the features better, you can create a matrix of the correlation coefficients between the features. Use the filtered dataset from now on.

    Plot a correlation matrix using [`heatmap()`](http://seaborn.pydata.org/generated/seaborn.heatmap.html). You can create the matrix using the standard `pandas` tools with the default parameters.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 2.1. (1 point).** Which pair of features has the strongest Pearson's correlation with the *gender* feature?

    1. Cardio, Cholesterol
    2. Height, Smoke
    3. Smoke, Alco
    4. Height, Weight

    **Answer**: [put correct answer choice # here]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2. Height distribution of men and women

    From our exploration of the unique values earlier, we know that the gender is encoded by the values *1* and *2*. Although you do not know the mapping of these values to gender, you can figure that out graphically by looking at the mean values of height and weight for each value of the *gender* feature.

    ### Task:

    Create a violin plot for the height and gender using [`violinplot()`](https://seaborn.pydata.org/generated/seaborn.violinplot.html). Use the parameters:
    - `hue` to split by gender;
    - `scale` to evaluate the number of records for each gender.

    In order for the plot to render correctly, you need to convert your `DataFrame` to *long* format using the `melt()` function from `pandas`. Here is [an example](https://stackoverflow.com/a/41575149/3338479) of this for your reference.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3. Rank correlation

    In most cases, *the Pearson coefficient of linear correlation* is more than enough to discover patterns in data.
    But let's go a little further and calculate a [rank correlation](https://en.wikipedia.org/wiki/Rank_correlation). It will help us to identify such feature pairs in which the lower rank in the variational series of one feature always precedes the higher rank in the other one (and we have the opposite in the case of negative correlation).

    Calculate and plot a correlation matrix using the [Spearman's rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient).
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 2.2. (1 point).** Which pair of features has the strongest Spearman rank correlation?

    1. Height, Weight
    2. Age, Weight
    3. Cholesterol, Gluc
    4. Cardio, Cholesterol
    5. Ap_hi, Ap_lo
    6. Smoke, Alco

    **Answer**: [put correct answer choice # here]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 2.3. (1 point).** Why do these features have strong rank correlation?

    1. Inaccuracies in the data (data acquisition errors).
    2. Relation is wrong, these features should not be related.
    3. Nature of the data.

    **Answer**: [put correct answer choice # here]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.4. Age

    Previously, we calculated the age of the respondents in years at the moment of examination.

    Create a *count plot* using [`countplot()`](http://seaborn.pydata.org/generated/seaborn.countplot.html) with the age on the *X* axis and the number of people on the *Y* axis. Your resulting plot should have two columns for each age, corresponding to the number of people for each *cardio* class of that age.
    """)
    return


@app.cell
def _():
    # TODO: answer the question by writing your Python code here; be sure to print out your final answer in a properly formatted way
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question 2.4. (1 point).** What is the smallest age at which the number of people with CVD outnumbers the number of people without CVD?

    1. 44
    2. 55
    3. 64
    4. 70

    **Answer**: [put correct answer choice # here]
    """)
    return


if __name__ == "__main__":
    app.run()
