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
    # Assignment #6: Beating baselines in a competition

    **Submit via the [Kaggle leaderboard](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/leaderboard)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Following simple baselines in the ["Alice" competition](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) (see [Topic 4](topic04_intro)), check out a bit more advanced Notebooks:

     - ["Correct time-aware cross-validation scheme"](https://www.kaggle.com/kashnitsky/correct-time-aware-cross-validation-scheme);
     - ["Model validation in a competition"](https://www.kaggle.com/kashnitsky/model-validation-in-a-competition);

    Go on with feature engineering and try to achieve ~ 0.955 (or higher) ROC AUC on the [Public Leaderboard](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/leaderboard). Alternatively, if a better solution is already shared by the time you join the competition, try to improve the best publicly shared solution by at least 0.5%. However, **please do not share high-performing solutions**, it ruins the competitive spirit of the competition and also hurts some other courses which also have this competition in their syllabus.

    You might want to check out [Bonus Assignment 4](bonus04) to achieve this.
    """)
    return


if __name__ == "__main__":
    app.run()
