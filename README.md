# mlcourse.ai-workbook

workbook for [mlcourse.ai](https://mlcourse.ai/book/index.html) - a free, open, and self-paced Machine Learning course by OpenDataScience that strikes the perfect balance between theory and practice

## Running locally

Notebooks are [marimo](https://marimo.io) `.py` files.

Install marimo:

```sh
pip install marimo
# or
uv tool install marimo
```

Install the workbook's dependencies and activate the environment (needs [uv](https://docs.astral.sh/uv/)):

```sh
uv sync
source .venv/bin/activate
```

Then open any notebook (browser tab; `--watch` picks up edits made outside marimo):

```sh
marimo edit --watch topic01-exploratory-data-analysis-pandas/notes.py
```

Or run one end-to-end as a plain script:

```sh
python topic01-exploratory-data-analysis-pandas/notes.py
```

Each topic directory has `notes*.py` (study notes; multi-part topics have one file per course page) and an `assignment.py` with `# TODO` cells to fill in.

## Modules

| # | Module | Overview |
|---|--------|----------|
| 1 | [Exploratory Data Analysis with Pandas](topic01-exploratory-data-analysis-pandas/notes.py) | Load, slice, group, and reshape tabular data with Pandas to explore a dataset and build simple baselines before any modeling. |
| 2 | [Visual Data Analysis in Python](topic02-visual-data-analysis/) | Plot distributions and relationships with Matplotlib, Seaborn, and Plotly to spot patterns and dependencies in data. |
| 3 | [Classification, Decision Trees, and k-NN](topic03-classification-decision-trees-knn/notes.py) | Learn how decision trees and k-nearest-neighbors classify data, and how tree splitting and hyperparameters affect fit. |
| 4 | [Linear Classification and Regression](topic04-linear-classification-regression/) | Cover least squares, logistic regression, regularization, and validation/learning curves for linear models. |
| 5 | [Bagging and Random Forest](topic05-bagging-random-forest/) | Reduce variance by aggregating many trees via bootstrap bagging and random forests, and read feature importances. |
| 6 | [Feature Engineering and Feature Selection](topic06-feature-engineering-selection/notes.py) | Craft, transform, and select features from raw, text, and categorical data to improve model performance. |
| 7 | [Unsupervised Learning: PCA and Clustering](topic07-unsupervised-pca-clustering/notes.py) | Reduce dimensionality with PCA and group unlabeled data with k-means and other clustering methods. |
| 8 | [SGD — Vowpal Wabbit: Learning with Gigabytes of Data](topic08-sgd-vowpal-wabbit/notes.py) | Scale to huge datasets with stochastic gradient descent, the hashing trick, and Vowpal Wabbit. |
| 9 | [Time Series Analysis with Python](topic09-time-series-analysis/) | Forecast time series with smoothing, ARIMA, and Facebook Prophet, including trend and seasonality handling. |
| 10 | [Gradient Boosting](topic10-gradient-boosting/notes.py) | Understand the gradient boosting algorithm and its libraries (XGBoost, LightGBM, CatBoost) for winning tabular models. |

## Course summary

The course walks the full applied-ML workflow on tabular data. It opens with the groundwork: exploring and visualizing data (topics 1–2) to build intuition and baselines before modeling. It then covers the core supervised models: decision trees and k-NN (topic 3), linear and logistic regression with regularization (topic 4), and tree ensembles via bagging and random forests (topic 5). Topic 6 makes models better through feature engineering and selection, while topic 7 steps into unsupervised learning with PCA and clustering. The final stretch handles scale and specialized data: stochastic gradient descent with Vowpal Wabbit for gigabyte-sized datasets (topic 8), time series forecasting (topic 9), and gradient boosting (topic 10), the go-to method for winning tabular problems. Throughout, the theme is the same: start simple, understand the mechanism, and validate before reaching for complexity.
