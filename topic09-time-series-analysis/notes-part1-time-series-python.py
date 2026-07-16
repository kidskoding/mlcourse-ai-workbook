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
    # Topic 9 — Part 1: Time Series Analysis in Python

    A **time series** is data indexed by time order. That ordering is not decoration — it carries information a random sample doesn't have, and it also takes away tools you're used to (you cannot shuffle rows, so plain k-fold CV is off the table).

    This part walks the whole ladder, cheapest first: naive averages → exponential smoothing → Holt-Winters → the econometric SARIMA route → and finally "just make features and fit a regression". The punchline arrives early and stays: the fanciest model is rarely the one worth shipping. Fast/good/cheap wins in practice, and SARIMA is none of the three.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    `statsmodels` is the econometrics half of the toolkit (ACF/PACF, Dickey-Fuller, SARIMAX); `scipy.optimize.minimize` does the parameter fitting for the hand-rolled smoothers. Everything else is the usual pandas/sklearn stack.

    Two real mobile-game series carry the whole lesson:
    - **`ads`** — ads watched per **hour**. Strong daily (24-hour) seasonality, no visible trend.
    - **`currency`** — in-game currency spent per **day**. Upward trend and a ~30-day seasonality.

    Note `index_col` + `parse_dates` on load: getting a real `DatetimeIndex` up front is what makes `shift`, `rolling`, and the date features work later.
    """)
    return


@app.cell
def _():
    import warnings
    from itertools import product

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.tsa.api as smt
    from scipy.optimize import minimize
    from tqdm.notebook import tqdm

    sns.set()
    warnings.filterwarnings("ignore")
    # '%matplotlib inline' command supported automatically in marimo
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'retina'

    DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    ads = pd.read_csv(DATA_PATH + "ads.csv", index_col=["Time"], parse_dates=["Time"])
    currency = pd.read_csv(
        DATA_PATH + "currency.csv", index_col=["Time"], parse_dates=["Time"]
    )
    return ads, currency, minimize, np, pd, plt, product, sm, smt, sns, tqdm


@app.cell
def _(ads, currency, plt):
    plt.figure(figsize=(12, 6))
    plt.plot(ads.Ads)
    plt.title("Ads watched (hourly data)")
    plt.grid(True)

    plt.figure(figsize=(12, 6))
    plt.plot(currency.GEMS_GEMS_SPENT)
    plt.title("In-game currency spent (daily data)")
    plt.grid(True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecast quality metrics

    Pick the metric *before* the model, because the metric is what you'll optimize.

    - **$R^2$** — fraction of variance explained, range $(-\infty, 1]$. Scale-free, so it says nothing about the size of your errors in real units.
    - **MAE** — $\frac{1}{n}\sum |y_i - \hat y_i|$. Same unit as the series, so it's interpretable.
    - **MedAE** — the *median* absolute error. Same idea, but robust to outliers.
    - **MSE** — $\frac{1}{n}\sum (y_i - \hat y_i)^2$. Punishes big misses hardest; the default for a reason.
    - **MSLE** — MSE on $\log(1+y)$. Use it when the series grows exponentially; it makes small errors matter too.
    - **MAPE** — MAE as a percentage. Not in sklearn's older API, so we write it; it's the one your manager actually understands.

    **The trap:** MAPE and MSLE are undefined at $y = 0$ and explode near it. Quietly dropping the $y_i = 0$ rows is worse than the disease — those are exactly the cases where you predicted a lot and reality delivered nothing. Know your series' floor before reaching for a percentage metric.
    """)
    return


@app.cell
def _(np):
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        mean_squared_log_error,
        median_absolute_error,
        r2_score,
    )


    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        mean_squared_log_error,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Move, smooth, evaluate

    The baseline to beat is embarrassing and often unbeatable: $\hat y_t = y_{t-1}$, "tomorrow looks like today". One step up is the **moving average** — predict the mean of the last $k$ observations:

    $$\hat y_t = \frac{1}{k}\sum_{n=1}^{k} y_{t-n}$$

    This only forecasts **one step ahead**: to get the next value you need the previous ones to have actually happened. So its real job isn't forecasting — it's **smoothing**, i.e. killing noise so a trend becomes visible. `DataFrame.rolling(window).mean()` is the pandas version. Wider window = smoother line = more lag.
    """)
    return


@app.cell
def _(ads, np):
    def moving_average(series, n):
        """Average of the last n observations."""
        return np.average(series[-n:])


    moving_average(ads, 24)  # forecast for the next hour from the past day
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Smoothing to see the trend, plus cheap anomaly detection

    `plotMovingAverage` does three things worth separating in your head:
    1. plots the rolling mean (the trend),
    2. optionally draws a **confidence band** at `rolling_mean ± (MAE + 1.96·σ)` of the smoothing residuals,
    3. optionally flags anything outside that band as an **anomaly**.

    That's a whole anomaly detector in ~10 lines — no training, no labels.
    """)
    return


@app.cell
def _(mean_absolute_error, np, pd, plt):
    def plotMovingAverage(
        series, window, plot_intervals=False, scale=1.96, plot_anomalies=False
    ):
        """Plot the rolling mean, optional confidence band, optional anomalies."""
        rolling_mean = series.rolling(window=window).mean()

        plt.figure(figsize=(15, 5))
        plt.title("Moving average\n window size = {}".format(window))
        plt.plot(rolling_mean, "g", label="Rolling mean trend")

        if plot_intervals:
            mae = mean_absolute_error(series[window:], rolling_mean[window:])
            deviation = np.std(series[window:] - rolling_mean[window:])
            lower_bond = rolling_mean - (mae + scale * deviation)
            upper_bond = rolling_mean + (mae + scale * deviation)
            plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "r--")

            if plot_anomalies:
                anomalies = pd.DataFrame(index=series.index, columns=series.columns)
                anomalies[series < lower_bond] = series[series < lower_bond]
                anomalies[series > upper_bond] = series[series > upper_bond]
                plt.plot(anomalies, "ro", markersize=10)

        plt.plot(series[window:], label="Actual values")
        plt.legend(loc="upper left")
        plt.grid(True)

    return (plotMovingAverage,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Window size is the whole game. On hourly ads data: 4 hours barely denoises, 12 hours starts to shape it, and **24 hours reveals the daily trend** — weekends up (more time to play), weekdays down. Match the window to the seasonal period you want to average *out*.
    """)
    return


@app.cell
def _(ads, plotMovingAverage):
    plotMovingAverage(ads, 4)
    plotMovingAverage(ads, 12)
    plotMovingAverage(ads, 24)  # daily smoothing on hourly data -> the daily trend
    plotMovingAverage(ads, 4, plot_intervals=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now break the data on purpose and see whether the detector catches it: knock one observation down by 80%.
    """)
    return


@app.cell
def _(ads, plotMovingAverage):
    ads_anomaly = ads.copy()
    ads_anomaly.iloc[-20] = ads_anomaly.iloc[-20] * 0.2  # simulate an 80% drop in ads

    plotMovingAverage(ads_anomaly, 4, plot_intervals=True, plot_anomalies=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It catches it. Then run the same detector on the currency series with a 7-day window and it falls apart — it flags nearly every 30-day peak as an anomaly.

    **The lesson:** a moving-average band only knows about noise around the window. The currency series has monthly seasonality the 7-day window never modelled, so real seasonal peaks look "abnormal". Unmodelled seasonality shows up as false positives.
    """)
    return


@app.cell
def _(currency, plotMovingAverage):
    plotMovingAverage(currency, 7, plot_intervals=True, plot_anomalies=True)  # weekly smoothing
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Weighted average

    A one-line upgrade: keep the last $k$ values but weight them, largest weight on the most recent, weights summing to 1.

    $$\hat y_t = \sum_{n=1}^{k} \omega_n y_{t+1-n}$$

    Still only one step ahead, still ad-hoc (you pick the weights by hand). It's the conceptual bridge to exponential smoothing, which picks the weights for you.
    """)
    return


@app.cell
def _(ads):
    def weighted_average(series, weights):
        """Weighted average; weights sorted descending (most recent gets weights[0])."""
        result = 0.0
        for n in range(len(weights)):
            result = result + series.iloc[-n - 1] * weights[n]
        return float(result)
    weighted_average(ads.Ads, [0.6, 0.3, 0.1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exponential smoothing

    Instead of weighting the last $k$ values and ignoring the rest, weight **all** of them, decaying exponentially into the past:

    $$\hat y_t = \alpha \cdot y_t + (1 - \alpha) \cdot \hat y_{t-1}$$

    The model value is a blend of the current observation and the previous model value — and since that previous value already contains its own $(1-\alpha)$ blend, the weights on old data decay geometrically. That recursion is where the "exponential" hides.

    $\alpha$ is the **smoothing factor** = how fast you forget. Small $\alpha$ → long memory, smooth line, slow to react. $\alpha = 1$ → the model is just "the current value".
    """)
    return


@app.cell
def _(ads, currency, plt):
    def exponential_smoothing(series, alpha):
        """alpha - float [0.0, 1.0], smoothing parameter."""
        result = [series.iloc[0]]  # first value is the series itself
        for n in range(1, len(series)):
            result.append(alpha * series.iloc[n] + (1 - alpha) * result[n - 1])
        return result


    def plotExponentialSmoothing(series, alphas):
        with plt.style.context("bmh"):
            plt.figure(figsize=(15, 7))
            for alpha in alphas:
                plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
            plt.plot(series.values, "c", label="Actual")
            plt.legend(loc="best")
            plt.axis("tight")
            plt.title("Exponential Smoothing")
            plt.grid(True)


    plotExponentialSmoothing(ads.Ads, [0.3, 0.05])
    plotExponentialSmoothing(currency.GEMS_GEMS_SPENT, [0.3, 0.05])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Double exponential smoothing (Holt)

    Single smoothing still only gives one future point. Decompose the series into two components and smooth each: **level** $\ell$ (where the series is) and **trend** $b$ (which way it's heading). The trend gets its own exponential smoothing, on the assumption that recent changes in direction predict future changes.

    $$\ell_x = \alpha y_x + (1-\alpha)(\ell_{x-1} + b_{x-1})$$
    $$b_x = \beta(\ell_x - \ell_{x-1}) + (1-\beta) b_{x-1}$$
    $$\hat y_{x+1} = \ell_x + b_x$$

    Now there are two knobs: $\alpha$ smooths the series around the trend, $\beta$ smooths the trend itself. Bigger = more weight on recent data = less smoothing. Setting them by hand produces nonsense fast (try $\alpha=0.9, \beta=0.9$) — which is exactly why the next section automates it.
    """)
    return


@app.cell
def _(ads, currency, plt):
    def double_exponential_smoothing(series, alpha, beta):
        """alpha - smoothing for level; beta - smoothing for trend."""
        result = [series.iloc[0]]
        for n in range(1, len(series) + 1):
            if n == 1:
                level, trend = series.iloc[0], series.iloc[1] - series.iloc[0]
            if n >= len(series):  # forecasting
                value = result[-1]
            else:
                value = series.iloc[n]
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        return result


    def plotDoubleExponentialSmoothing(series, alphas, betas):
        with plt.style.context("bmh"):
            plt.figure(figsize=(20, 8))
            for alpha in alphas:
                for beta in betas:
                    plt.plot(
                        double_exponential_smoothing(series, alpha, beta),
                        label="Alpha {}, beta {}".format(alpha, beta),
                    )
            plt.plot(series.values, label="Actual")
            plt.legend(loc="best")
            plt.axis("tight")
            plt.title("Double Exponential Smoothing")
            plt.grid(True)


    plotDoubleExponentialSmoothing(ads.Ads, alphas=[0.9, 0.02], betas=[0.9, 0.02])
    plotDoubleExponentialSmoothing(
        currency.GEMS_GEMS_SPENT, alphas=[0.9, 0.02], betas=[0.9, 0.02]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Triple exponential smoothing (Holt-Winters)

    Add a third component: **seasonality** $s$, with a season length $L$. One seasonal component per position in the season — 7 for weekly-on-daily data, 24 for daily-on-hourly. Obvious corollary: **don't use this if the series has no seasonality.**

    $$\ell_x = \alpha(y_x - s_{x-L}) + (1-\alpha)(\ell_{x-1} + b_{x-1})$$
    $$b_x = \beta(\ell_x - \ell_{x-1}) + (1-\beta) b_{x-1}$$
    $$s_x = \gamma(y_x - \ell_x) + (1-\gamma) s_{x-L}$$
    $$\hat y_{x+m} = \ell_x + m b_x + s_{x-L+1+(m-1)\bmod L}$$

    The level equation now works on the **deseasonalized** value $y_x - s_{x-L}$, and $\gamma$ smooths the seasonal components themselves.

    The implementation also bolts on **Brutlag's method** for confidence bands: track a smoothed predicted deviation $d_t = \gamma|y_t - \hat y_t| + (1-\gamma)d_{t-1}$ and put the bands at $\hat y_t \pm m \cdot d_{t-L}$. Because the deviation is itself exponentially smoothed, the band widens right after a structural shock and then quickly forgets it — which is what makes this a practical anomaly detector on noisy series with almost no data prep.
    """)
    return


@app.cell
def _(np):
    class HoltWinters:
        """
        Holt-Winters model with anomaly detection by the Brutlag method.

        series          - initial time series
        slen            - length of a season
        alpha, beta, gamma - model coefficients
        n_preds         - prediction horizon
        scaling_factor  - width of the Brutlag confidence interval (usually 2 to 3)
        """

        def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
            self.series = np.asarray(series, dtype=float)  # positional indexing throughout, and the CV loop hands us a numpy array
            self.slen = slen  # anyway -> normalize once here instead of .iloc-ing at every call site
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.n_preds = n_preds
            self.scaling_factor = scaling_factor

        def initial_trend(self):
            total = 0.0
            for i in range(self.slen):
                total = total + float(self.series[i + self.slen] - self.series[i]) / self.slen
            return total / self.slen

        def initial_seasonal_components(self):
            seasonals = {}
            season_averages = []
            n_seasons = int(len(self.series) / self.slen)
            for j in range(n_seasons):
                season_averages.append(sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))
            for i in range(self.slen):
                sum_of_vals_over_avg = 0.0
                for j in range(n_seasons):
                    sum_of_vals_over_avg = sum_of_vals_over_avg + (self.series[self.slen * j + i] - season_averages[j])
                seasonals[i] = sum_of_vals_over_avg / n_seasons
            return seasonals

        def triple_exponential_smoothing(self):
            self.result = []
            self.Smooth = []
            self.Season = []
            self.Trend = []
            self.PredictedDeviation = []
            self.UpperBond = []
            self.LowerBond = []
            seasonals = self.initial_seasonal_components()
            for i in range(len(self.series) + self.n_preds):
                if i == 0:
                    smooth = self.series[0]
                    trend = self.initial_trend()
                    self.result.append(self.series[0])
                    self.Smooth.append(smooth)
                    self.Trend.append(trend)
                    self.Season.append(seasonals[i % self.slen])
                    self.PredictedDeviation.append(0)  # components initialization
                    self.UpperBond.append(self.result[0] + self.scaling_factor * self.PredictedDeviation[0])
                    self.LowerBond.append(self.result[0] - self.scaling_factor * self.PredictedDeviation[0])
                    continue
                if i >= len(self.series):
                    m = i - len(self.series) + 1
                    self.result.append(smooth + m * trend + seasonals[i % self.slen])
                    self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)
                else:
                    val = self.series[i]
                    last_smooth, smooth = (smooth, self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (smooth + trend))
                    trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                    seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                    self.result.append(smooth + trend + seasonals[i % self.slen])
                    self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) + (1 - self.gamma) * self.PredictedDeviation[-1])
                self.UpperBond.append(self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1])
                self.LowerBond.append(self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1])  # predicting
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])  # uncertainty grows on each forecast step  # Brutlag's predicted deviation

    return (HoltWinters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time series cross-validation

    You cannot shuffle time. Random folds destroy the temporal dependencies that are the entire signal, and they let the model peek at the future.

    The fix is **cross-validation on a rolling basis**: train on a prefix of the series, validate on the chunk immediately after it, then extend the prefix and repeat. Every fold trains only on data that precedes its test set. `sklearn.model_selection.TimeSeriesSplit` implements it.

    Wrap that in a loss and you can optimize $\alpha, \beta, \gamma$ automatically instead of guessing.
    """)
    return


@app.cell
def _(HoltWinters, mean_squared_error, np):
    from sklearn.model_selection import TimeSeriesSplit


    def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=24):
        """Mean error across rolling CV folds for a given (alpha, beta, gamma)."""
        errors = []
        values = series.values
        alpha, beta, gamma = params

        tscv = TimeSeriesSplit(n_splits=3)

        for train, test in tscv.split(values):
            model = HoltWinters(
                series=values[train],
                slen=slen,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                n_preds=len(test),
            )
            model.triple_exponential_smoothing()

            predictions = model.result[-len(test) :]
            actual = values[test]
            errors.append(loss_function(predictions, actual))

        return np.mean(np.array(errors))

    return TimeSeriesSplit, timeseriesCVscore


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fitting the parameters

    All three smoothing parameters live in $[0, 1]$, so the optimizer must support **bounds** — hence `method="TNC"` (truncated Newton conjugate gradient) rather than an unconstrained method. Hold out the last 20 hours so the forecast has something honest to be scored against.
    """)
    return


@app.cell
def _(HoltWinters, ads, mean_squared_log_error, minimize, timeseriesCVscore):
    data = ads.Ads[:-20]  # hold out the last 20 hours
    x = [0, 0, 0]
    _opt = minimize(timeseriesCVscore, x0=x, args=(data, mean_squared_log_error), method='TNC', bounds=((0, 1), (0, 1), (0, 1)))  # initial alpha, beta, gamma
    _alpha_final, _beta_final, _gamma_final = _opt.x
    print(_alpha_final, _beta_final, _gamma_final)
    model = HoltWinters(data, slen=24, alpha=_alpha_final, beta=_beta_final, gamma=_gamma_final, n_preds=50, scaling_factor=3)
    model.triple_exponential_smoothing()
    return (model,)


@app.cell
def _(ads, mean_absolute_percentage_error, model, np, plt):
    def plotHoltWinters(series, plot_intervals=False, plot_anomalies=False):
        """Plot the fitted Holt-Winters model against actuals (uses the global `model`)."""
        plt.figure(figsize=(20, 10))
        plt.plot(model.result, label="Model")
        plt.plot(series.values, label="Actual")
        error = mean_absolute_percentage_error(series.values, model.result[: len(series)])
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

        if plot_anomalies:
            anomalies = np.array([np.nan] * len(series))
            anomalies[series.values < model.LowerBond[: len(series)]] = series.values[
                series.values < model.LowerBond[: len(series)]
            ]
            anomalies[series.values > model.UpperBond[: len(series)]] = series.values[
                series.values > model.UpperBond[: len(series)]
            ]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

        if plot_intervals:
            plt.plot(model.UpperBond, "r--", alpha=0.5, label="Up/Low confidence")
            plt.plot(model.LowerBond, "r--", alpha=0.5)
            plt.fill_between(
                x=range(0, len(model.result)),
                y1=model.UpperBond,
                y2=model.LowerBond,
                alpha=0.2,
                color="grey",
            )

        plt.vlines(
            len(series),
            ymin=min(model.LowerBond),
            ymax=max(model.UpperBond),
            linestyles="dashed",
        )
        plt.axvspan(len(series) - 20, len(model.result), alpha=0.3, color="lightgrey")
        plt.grid(True)
        plt.axis("tight")
        plt.legend(loc="best", fontsize=13)


    plotHoltWinters(ads.Ads)
    plotHoltWinters(ads.Ads, plot_intervals=True, plot_anomalies=True)
    return (plotHoltWinters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model catches the daily seasonality, the mild downward trend, and several anomalies. Plotting Brutlag's predicted deviation makes the "reacts sharply, then forgets" behaviour explicit — the band spikes at a structural change and decays back.

    Rerunning the same machinery on the currency series only needs `slen=30` (30-day seasonality) and a different loss. Same code, different season length — that's the point.
    """)
    return


@app.cell
def _(model, plt):
    plt.figure(figsize=(25, 5))
    plt.plot(model.PredictedDeviation)
    plt.grid(True)
    plt.axis("tight")
    plt.title("Brutlag's predicted deviation")
    return


@app.cell
def _(
    HoltWinters,
    currency,
    mean_absolute_percentage_error,
    minimize,
    plotHoltWinters,
    timeseriesCVscore,
):
    data_1 = currency.GEMS_GEMS_SPENT[:-50]
    slen = 30
    _opt = minimize(timeseriesCVscore, x0=[0, 0, 0], args=(data_1, mean_absolute_percentage_error, slen), method='TNC', bounds=((0, 1), (0, 1), (0, 1)))
    _alpha_final, _beta_final, _gamma_final = _opt.x
    print(_alpha_final, _beta_final, _gamma_final)
    model_1 = HoltWinters(data_1, slen=slen, alpha=_alpha_final, beta=_beta_final, gamma=_gamma_final, n_preds=100, scaling_factor=3)
    model_1.triple_exponential_smoothing()
    plotHoltWinters(currency.GEMS_GEMS_SPENT)
    plotHoltWinters(currency.GEMS_GEMS_SPENT, plot_intervals=True, plot_anomalies=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The econometric approach: stationarity

    **Stationarity** is the entry fee for the ARIMA family. A process is stationary if its statistical properties don't drift with time: constant mean, constant variance (**homoscedasticity**), and a covariance that depends only on the *distance* between observations, not on when you look.

    Why it matters: if the properties don't change, they're estimable from the past and extrapolable into the future. Non-stationary series break that promise, so the whole ARIMA workflow is really "beat the series into stationarity, model it, then undo the beating."

    Generate white noise, then feed it back into itself as $x_t = \rho x_{t-1} + e_t$ and sweep $\rho$. At $\rho = 0$ it's white noise. As $\rho$ grows, wider and wider cycles appear. At $\rho = 1$ you have a **random walk** — a unit root, non-stationary, mean wanders freely.

    The **Dickey-Fuller test** is the formal check. Null hypothesis: a unit root is present (non-stationary). A small p-value means you *reject* the null, i.e. the series is stationary. It's printed in each plot title below; watch it stop rejecting as $\rho \to 1$.
    """)
    return


@app.cell
def _(np, plt, sm):
    white_noise = np.random.normal(size=1000)
    with plt.style.context("bmh"):
        plt.figure(figsize=(15, 5))
        plt.plot(white_noise)


    def plotProcess(n_samples=1000, rho=0):
        x = w = np.random.normal(size=n_samples)
        for t in range(n_samples):
            x[t] = rho * x[t - 1] + w[t]

        with plt.style.context("bmh"):
            plt.figure(figsize=(10, 3))
            plt.plot(x)
            plt.title(
                "Rho {}\n Dickey-Fuller p-value: {}".format(
                    rho, round(sm.tsa.stattools.adfuller(x)[1], 3)
                )
            )


    for rho in [0, 0.6, 0.9, 1]:
        plotProcess(rho=rho)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differencing your way to stationarity

    `tsplot` is the workhorse diagnostic: the series, its **ACF** (autocorrelation: correlation with itself at lag $k$), its **PACF** (partial autocorrelation: same, with the intermediate lags' effects removed), and the Dickey-Fuller p-value in the title.

    The recipe:
    1. Look at the raw series. `ads` is already stationary (no trend, stable variance) — Dickey-Fuller rejects the unit root. But seasonality is still there, and seasonality must go before modeling.
    2. Take the **seasonal difference**: subtract the series from itself lagged by one season (`shift(24)`). Seasonality gone; ACF still has too many significant lags.
    3. Take the **first difference** (`shift(1)`). Now it oscillates around zero, Dickey-Fuller is happy, and the significant ACF peaks have collapsed.

    Each difference you take is a `d` (or `D`) in the SARIMA order — and each one you'll have to undo when you forecast.
    """)
    return


@app.cell
def _(ads, pd, plt, sm, smt):
    def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
        """Plot the series, its ACF and PACF, and the Dickey-Fuller p-value."""
        if not isinstance(_y, pd.Series):
            _y = pd.Series(_y)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            layout = (2, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            ts_ax.plot(_y)
            p_value = sm.tsa.stattools.adfuller(_y)[1]
            ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
            smt.graphics.plot_acf(_y, lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(_y, lags=lags, ax=pacf_ax)
            plt.tight_layout()
    tsplot(ads.Ads, lags=60)  # already stationary, but seasonal
    return (tsplot,)


@app.cell
def _(ads, tsplot):
    # 1) seasonal difference: kills the 24-hour seasonality
    ads_diff = ads.Ads - ads.Ads.shift(24)
    tsplot(ads_diff[24:], lags=60)

    # 2) first difference: kills the leftover autocorrelation
    ads_diff = ads_diff - ads_diff.shift(1)
    tsplot(ads_diff[24 + 1 :], lags=60)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SARIMA crash course

    $\text{SARIMA}(p, d, q)(P, D, Q, s)$, letter by letter:

    - **AR($p$)** — autoregression: regress the series on its own past values, up to lag $p$. Read the initial $p$ off the **PACF**: the last significant lag after which most others go quiet.
    - **MA($q$)** — moving average *of the errors*: the current error depends on previous errors up to lag $q$. Read $q$ off the **ACF**, same logic.
    - **I($d$)** — integration: $d$ = how many first differences you took to reach stationarity. ARIMA = AR + I + MA.
    - **$(P, D, Q, s)$** — the same three letters applied at the seasonal lag, with $s$ = season length. $D$ = number of seasonal differences.

    For the differenced `ads` series, read off the final `tsplot`: **p ≈ 4** (last significant PACF lag), **d = 1** (one first difference), **q ≈ 4** (from the ACF), **P ≈ 2** (lags 24 and 48 look significant on the PACF), **D = 1** (one seasonal difference), **Q ≈ 1** (lag 24 significant on the ACF, 48 not).

    Those are starting points, not answers. Grid-search around them and pick by **AIC** (Akaike Information Criterion — fit penalized by parameter count; lower is better).
    """)
    return


@app.cell
def _(ads, pd, product, sm, tqdm):
    # search grid around the values read off ACF/PACF
    ps = range(2, 5)
    d = 1
    qs = range(2, 5)
    Ps = range(0, 3)
    D = 1
    Qs = range(0, 2)
    s = 24  # season length

    parameters_list = list(product(ps, qs, Ps, Qs))
    len(parameters_list)


    def optimizeSARIMA(parameters_list, d, D, s):
        """Return a dataframe of (p, q, P, Q) tuples and their AIC, best first."""
        results = []
        best_aic = float("inf")

        for param in tqdm(parameters_list):
            # try/except because some combinations simply fail to converge
            try:
                model = sm.tsa.statespace.SARIMAX(
                    ads.Ads,
                    order=(param[0], d, param[1]),
                    seasonal_order=(param[2], D, param[3], s),
                ).fit(disp=-1)
            except:
                continue
            aic = model.aic
            if aic < best_aic:
                best_model, best_aic, best_param = model, aic, param
            results.append([param, model.aic])

        result_table = pd.DataFrame(results, columns=["parameters", "aic"])
        return result_table.sort_values(by="aic", ascending=True).reset_index(drop=True)


    result_table = optimizeSARIMA(parameters_list, d, D, s)
    result_table.head()
    return D, d, result_table, s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fit the lowest-AIC model and **check the residuals**. This is the step people skip. If the residuals are stationary with no leftover autocorrelation, the model has extracted the structure and only noise remains — that's what "done" looks like.
    """)
    return


@app.cell
def _(D, ads, d, result_table, s, sm, tsplot):
    p, q, P, Q = result_table.parameters[0]

    best_model = sm.tsa.statespace.SARIMAX(
        ads.Ads, order=(p, d, q), seasonal_order=(P, D, Q, s)
    ).fit(disp=-1)
    print(best_model.summary())

    tsplot(best_model.resid[24 + 1 :], lags=60)  # should look like noise
    return (best_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the `s + d` shift in the plotting function: those first values were never observed by the model because of the differencing, so they're excluded from both the plot and the error.

    The forecast comes out around **4% MAPE** — genuinely very good. And still: the data prep, the stationarity surgery, the parameter grid, and the retraining cost more than the accuracy is usually worth. Hold that thought for the next section.
    """)
    return


@app.cell
def _(ads, best_model, d, mean_absolute_percentage_error, np, pd, plt, s):
    def plotSARIMA(series, model, n_steps):
        """Plot fitted + forecast values against actuals. n_steps - steps to forecast."""
        data = series.copy()
        data.columns = ["actual"]
        data["arima_model"] = model.fittedvalues
        # the first s+d values were unobserved by the model due to differencing
        data.loc[data.index[: s + d], "arima_model"] = np.nan

        forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
        forecast = pd.concat([data.arima_model, forecast])  # fitted line + out-of-sample
        error = mean_absolute_percentage_error(
            data["actual"][s + d :], data["arima_model"][s + d :]
        )

        plt.figure(figsize=(15, 7))
        plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
        plt.plot(forecast, color="r", label="model")
        plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color="lightgrey")
        plt.plot(data.actual, label="actual")
        plt.legend()
        plt.grid(True)


    plotSARIMA(ads, best_model, 50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear (and other) models: turn the series into a table

    The pragmatic move: stop doing econometrics, start doing supervised learning. Extract features from the series and fit a regression. This **breaks assumptions** (Gauss-Markov wants uncorrelated errors; time series errors are anything but) but it is fast, cheap, easy to tune, and it wins Kaggle competitions.

    What can you extract from a 1-D series?
    - **Lags** of the series itself
    - **Window statistics** — max/min/mean/median/variance over a rolling window
    - **Date/time features** — hour of day, day of week, is-it-a-holiday flags, special events
    - **Target encoding** — categories replaced by the target's mean in that category
    - **Forecasts from other models** (at the cost of prediction speed)

    ### Lags

    Shifting the series $n$ steps back aligns $y_t$ with $y_{t-n}$ as a feature. Train on a lag-1 feature and you can forecast 1 step ahead. Want to forecast 6 steps ahead? Your minimum lag must be 6 — and the model then only ever sees data from 6 steps ago, blind to anything that happened since.

    **The tradeoff to internalize:** forecast horizon and forecast quality trade against each other, and the lag range is the dial. Here we use lags 6 through 24 — forecast 6 hours out, with a full day of history.
    """)
    return


@app.cell
def _(ads, pd):
    data_2 = pd.DataFrame(ads.Ads.copy())
    data_2.columns = ['y']
    for i in range(6, 25):
    # lags from 6 steps back up to 24 -> a 6-step-ahead horizon with a day of history
        data_2['lag_{}'.format(i)] = data_2.y.shift(i)
    data_2.tail(7)
    return (data_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train/test split that respects time

    Never `train_test_split(shuffle=True)` here. The test set must be the **tail** of the series — everything the model trains on has to precede everything it's scored on. That's what this small helper enforces, and skipping it is the single easiest way to fool yourself.
    """)
    return


@app.cell
def _(TimeSeriesSplit, data_2):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    tscv = TimeSeriesSplit(n_splits=5)

    def timeseries_train_test_split(X, y, test_size):
        """Train/test split that respects the time ordering: test is the tail."""
        test_index = int(len(_X) * (1 - test_size))
        return (_X.iloc[:test_index], _X.iloc[test_index:], _y.iloc[:test_index], _y.iloc[test_index:])
    _y = data_2.dropna().y
    _X = data_2.dropna().drop(['y'], axis=1)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(_X, _y, test_size=0.3)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return (
        LinearRegression,
        X_test,
        X_train,
        cross_val_score,
        lr,
        timeseries_train_test_split,
        tscv,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plain lags + linear regression lands at ~6.29% MAPE against SARIMA's ~3.94% — worse, but in the same ballpark for two lines of modeling and none of the stationarity surgery. That's the whole argument for this section.

    `plotModelResults` builds its prediction interval from **rolling CV** (`cross_val_score` with `cv=tscv`), not from the training residuals: band = `prediction ± (MAE + 1.96·σ)` across folds. Anything the actual value does outside that band gets flagged as an anomaly — same trick as the moving-average detector, better model underneath.
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    cross_val_score,
    lr,
    mean_absolute_percentage_error,
    np,
    pd,
    plt,
    tscv,
    y_test,
    y_train,
):
    def plotModelResults(
        model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False
    ):
        """Plot predictions vs actuals, with CV-derived intervals and anomalies."""
        prediction = model.predict(X_test)

        plt.figure(figsize=(15, 7))
        plt.plot(prediction, "g", label="prediction", linewidth=2.0)
        plt.plot(y_test.values, label="actual", linewidth=2.0)

        if plot_intervals:
            cv = cross_val_score(
                model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error"
            )
            mae = cv.mean() * (-1)
            deviation = cv.std()

            scale = 1.96
            lower = prediction - (mae + scale * deviation)
            upper = prediction + (mae + scale * deviation)

            plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
            plt.plot(upper, "r--", alpha=0.5)

            if plot_anomalies:
                anomalies = np.array([np.nan] * len(y_test))
                anomalies[y_test < lower] = y_test[y_test < lower]
                anomalies[y_test > upper] = y_test[y_test > upper]
                plt.plot(anomalies, "o", markersize=10, label="Anomalies")

        error = mean_absolute_percentage_error(y_test, prediction)
        plt.title("Mean absolute percentage error {0:.2f}%".format(error))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)


    def plotCoefficients(model):
        """Plot model coefficients sorted by absolute value."""
        coefs = pd.DataFrame(model.coef_, X_train.columns)
        coefs.columns = ["coef"]
        coefs["abs"] = coefs.coef.apply(np.abs)
        coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

        plt.figure(figsize=(15, 7))
        coefs.coef.plot(kind="bar")
        plt.grid(True, axis="y")
        plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles="dashed")


    plotModelResults(lr, plot_intervals=True)
    plotCoefficients(lr)
    return plotCoefficients, plotModelResults


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Date features, and why you must scale

    Add `hour`, `weekday`, and an `is_weekend` boolean — this is where the `DatetimeIndex` from the very first load pays off.

    Then **scale**. Lag features are in the tens of thousands; `hour` is in the tens. Coefficient magnitudes are meaningless across those scales, and regularization (next section) would effectively only penalize the lags. `StandardScaler` — fit on train, `transform` on test, never `fit_transform` on test.

    Result: test error drops a little, and the coefficient plot shows `weekday` and `is_weekend` genuinely earning their keep.
    """)
    return


@app.cell
def _(
    LinearRegression,
    data_2,
    pd,
    plotCoefficients,
    plotModelResults,
    timeseries_train_test_split,
):
    from sklearn.preprocessing import StandardScaler
    data_2.index = pd.to_datetime(data_2.index)
    data_2['hour'] = data_2.index.hour
    data_2['weekday'] = data_2.index.weekday
    data_2['is_weekend'] = data_2.weekday.isin([5, 6]) * 1
    scaler = StandardScaler()
    _y = data_2.dropna().y
    _X = data_2.dropna().drop(['y'], axis=1)
    X_train_1, X_test_1, y_train_1, y_test_1 = timeseries_train_test_split(_X, _y, test_size=0.3)
    X_train_scaled = scaler.fit_transform(X_train_1)
    X_test_scaled = scaler.transform(X_test_1)
    lr_1 = LinearRegression()
    lr_1.fit(X_train_scaled, y_train_1)
    plotModelResults(lr_1, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
    plotCoefficients(lr_1)
    return (scaler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Target encoding — and the leak that comes with it

    One-hot encoding `hour` explodes the feature space; treating it as a number implies "0 hours < 23 hours", which is false for a cyclic variable. **Target encoding** sidesteps both: replace each category with the mean of the target within that category. Every hour of the day becomes "the average number of ads watched at that hour".

    **The rule that makes it safe:** compute the means on the **training set only** (or the current CV fold only). Compute them on the full data and you have handed the model the future. `prepareData` does this correctly — note `code_mean(data[:test_index], ...)`.
    """)
    return


@app.cell
def _(data_2, pd, plt):
    def code_mean(data, cat_feature, real_feature):
        """{category: mean of real_feature within that category}"""
        return dict(data.groupby(cat_feature)[real_feature].mean())
    average_hour = code_mean(data_2, 'hour', 'y')
    plt.figure(figsize=(7, 5))
    plt.title('Hour averages')
    pd.DataFrame.from_dict(average_hour, orient='index')[0].plot()
    plt.grid(True)
    return (code_mean,)


@app.cell
def _(
    LinearRegression,
    ads,
    code_mean,
    pd,
    plotCoefficients,
    plotModelResults,
    scaler,
    timeseries_train_test_split,
):
    def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
        """
        lag_start / lag_end - range of lags to build (lag_start = the forecast horizon)
        test_size           - test fraction, split respecting time order
        target_encoding     - if True, add target averages computed on train only
        """
        data = pd.DataFrame(series.copy())
        data.columns = ['y']
        for i in range(lag_start, lag_end):
            data['lag_{}'.format(i)] = data.y.shift(i)
        data.index = pd.to_datetime(data.index)
        data['hour'] = data.index.hour
        data['weekday'] = data.index.weekday
        data['is_weekend'] = data.weekday.isin([5, 6]) * 1
        if target_encoding:
            test_index = int(len(data.dropna()) * (1 - test_size))
            data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', 'y').get, data.weekday))
            data['hour_average'] = list(map(code_mean(data[:test_index], 'hour', 'y').get, data.hour))
            data.drop(['hour', 'weekday'], axis=1, inplace=True)
        _y = data.dropna().y
        _X = data.dropna().drop(['y'], axis=1)
        return timeseries_train_test_split(_X, _y, test_size=test_size)
    X_train_2, X_test_2, y_train_2, y_test_2 = prepareData(ads.Ads, lag_start=6, lag_end=25, test_size=0.3, target_encoding=True)
    X_train_scaled_1 = scaler.fit_transform(X_train_2)
    X_test_scaled_1 = scaler.transform(X_test_2)
    lr_2 = LinearRegression()
    lr_2.fit(X_train_scaled_1, y_train_2)
    plotModelResults(lr_2, X_train=X_train_scaled_1, X_test=X_test_scaled_1, plot_intervals=True, plot_anomalies=True)
    plotCoefficients(lr_2)
    return (prepareData,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Overfitting, live.** `hour_average` was so good on the training set that the model dumped all its weight onto it — and test quality got *worse*. Even leak-free target encoding can overfit if the encoded feature is too informative in-sample.

    Fixes: compute the encoding over a rolling **window** rather than the whole train set (recent encodings describe the current state better), or just drop the feature. Here we drop it and move on.
    """)
    return


@app.cell
def _(ads, prepareData, scaler):
    X_train_3, X_test_3, y_train_3, y_test_3 = prepareData(ads.Ads, lag_start=6, lag_end=25, test_size=0.3, target_encoding=False)
    X_train_scaled_2 = scaler.fit_transform(X_train_3)
    X_test_scaled_2 = scaler.transform(X_test_3)
    return X_test_scaled_2, X_train_3, X_train_scaled_2, y_train_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularization and feature selection

    Lag features are brutally **multicollinear** — plot `X_train.corr()` and the heatmap glows. Correlated features let coefficients blow up in opposite directions and cancel; the fit looks fine, generalization doesn't.

    Regularization adds a penalty on coefficient size to the loss:
    - **Ridge (L2)** — penalty $\propto \sum \beta_j^2$. Shrinks coefficients toward zero but never *to* zero. Higher bias, lower variance.
    - **Lasso (L1)** — penalty $\propto \sum |\beta_j|$. Can drive coefficients exactly to zero, so it does **feature selection** for free.

    `RidgeCV` / `LassoCV` pick the penalty strength by cross-validation — and you pass `cv=tscv` so the selection also respects time order. Here Lasso is the more aggressive one: it zeroes out `lag_11`, `lag_17` and `lag_21` outright, and prediction quality *improves* (MAPE 5.60% vs Ridge's 6.06%). Fewer features, better model.
    """)
    return


@app.cell
def _(X_train_3, plt, sns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_train_3.corr())  # lags are heavily collinear
    return


@app.cell
def _(
    X_test_scaled_2,
    X_train_scaled_2,
    plotCoefficients,
    plotModelResults,
    tscv,
    y_train_3,
):
    from sklearn.linear_model import LassoCV, RidgeCV
    ridge = RidgeCV(cv=tscv)
    ridge.fit(X_train_scaled_2, y_train_3)
    plotModelResults(ridge, X_train=X_train_scaled_2, X_test=X_test_scaled_2, plot_intervals=True, plot_anomalies=True)
    plotCoefficients(ridge)
    lasso = LassoCV(cv=tscv)
    lasso.fit(X_train_scaled_2, y_train_3)
    plotModelResults(lasso, X_train=X_train_scaled_2, X_test=X_test_scaled_2, plot_intervals=True, plot_anomalies=True)
    plotCoefficients(lasso)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boosting

    XGBoost on the same feature matrix gives the **lowest test error of any model built on the feature table** (~4.18% MAPE, vs ~5.60% for Lasso — still a shade behind SARIMA's ~3.94%). And you should still be suspicious.

    Tree models handle **trends** badly. A tree splits on thresholds; it can only ever predict values it saw in training, so it cannot extrapolate a rising line — it flatlines at the edge of the training range. `ads` has no real trend, so XGBoost looks great. On a trending series it would quietly fail.

    **The rule:** detrend or difference into stationarity *before* boosting. Or model the trend with a linear model and let the booster fit the residual on top. "Throw XGBoost at it" is not a time series strategy.
    """)
    return


@app.cell
def _(X_test_scaled_2, X_train_scaled_2, plotModelResults, y_train_3):
    from xgboost import XGBRegressor
    xgb = XGBRegressor(verbosity=0)
    xgb.fit(X_train_scaled_2, y_train_3)
    plotModelResults(xgb, X_train=X_train_scaled_2, X_test=X_test_scaled_2, plot_intervals=True, plot_anomalies=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Time ordering is a constraint, not a detail.** No shuffling: use `TimeSeriesSplit` for CV and split the test set off the tail of the series. Violating this is the easiest way to publish a fake result.
    - **Start with the naive baseline.** $\hat y_t = y_{t-1}$ is a serious contender and is sometimes unbeatable. Everything else must justify itself against it.
    - **Moving averages smooth, they don't forecast.** Match the window to the seasonal period you want to average out; the ± band around the rolling mean is a free anomaly detector.
    - **Unmodelled seasonality reads as anomalies.** The 7-day band on the currency series flagged every monthly peak. False positives are a symptom of a missing component.
    - **Exponential smoothing scales by components.** Single = level; double (Holt) = + trend; triple (Holt-Winters) = + seasonality. Don't use triple on a series with no season.
    - **Tune $\alpha, \beta, \gamma$ by optimization, not by hand.** They're bounded to $[0,1]$, so use a bounded optimizer (`TNC`) over a rolling-CV loss.
    - **Stationarity is the ARIMA entry fee.** Difference seasonally, then difference again; confirm with Dickey-Fuller (small p = stationary) and by watching the ACF peaks collapse.
    - **Read $p$ off the PACF and $q$ off the ACF**, then grid-search around them by AIC — and always check that the residuals look like noise.
    - **SARIMA is accurate and expensive.** ~4% MAPE, bought with hours of data surgery and retraining. Rarely the right production trade.
    - **Feature-engineering + linear regression gets in the same ballpark, for almost nothing.** Lags + date features land ~5.6-6.3% MAPE vs SARIMA's ~3.9%, at a fraction of the effort. Assumption-breaking but effective.
    - **Lag range = forecast horizon.** Minimum lag 6 means forecasting 6 steps out, blind to the last 6 steps. Quality and horizon trade against each other.
    - **Scale before you regularize or read coefficients.** Fit the scaler on train, `transform` test.
    - **Target encoding must be computed on train only** — and it can still overfit even when leak-free. Use a rolling window or drop it.
    - **Lasso beats Ridge when features are junk.** L1 zeroes coefficients out entirely; here dropping features *improved* the score.
    - **XGBoost wins the feature-table bake-off for a suspicious reason.** ~4.18% MAPE, best of the regressions — but trees can't extrapolate trends. Detrend first, or model the trend separately and boost the residual.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    This part climbs from the naive "tomorrow = today" baseline through moving/weighted averages (smoothing and a cheap anomaly detector), exponential smoothing in its single/double/triple forms up to Holt-Winters with Brutlag confidence bands, whose $\alpha, \beta, \gamma$ get fitted by a bounded optimizer over rolling-basis cross-validation — the one CV scheme that doesn't leak the future. It then takes the econometric route: stationarity and the Dickey-Fuller test, seasonal + first differencing, reading $p$ and $q$ off the PACF and ACF, and an AIC grid search for SARIMA that lands ~4% MAPE at a cost most projects can't justify. The pragmatic finale converts the series into a feature table — lags, hour/weekday/is_weekend, target encoding (train-set means only, and still overfitting) — and shows that a scaled linear regression gets within ~1.7 points of SARIMA's MAPE for a fraction of the effort, that Lasso improves things by deleting features outright, and that XGBoost's winning score hides its inability to extrapolate a trend. There is no single right method; pick by the balance of quality against the cost of building and maintaining it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **time series** = data points indexed in time order, where the ordering itself carries information
    - **stationarity** = statistical properties (mean, variance, covariance structure) do not change over time; the entry requirement for ARIMA-family models
        - **homoscedasticity** = the constant-variance part of stationarity
    - **unit root** = the non-stationarity that makes a series a random walk ($\rho = 1$)
    - **random walk** = $x_t = x_{t-1} + e_t$; non-stationary, mean wanders freely
    - **white noise** = i.i.d. draws with constant mean and variance; the stationary reference process
    - **Dickey-Fuller test** = hypothesis test for a unit root; null = non-stationary, so a small p-value means stationary
    - **differencing** = subtracting the series from a lagged copy of itself to remove structure
    - **first difference** = `series - series.shift(1)`; removes trend/leftover autocorrelation, sets $d$
    - **seasonal difference** = `series - series.shift(s)`; removes seasonality, sets $D$
    - **ACF** = autocorrelation function; correlation of the series with itself at each lag — read $q$ off it
    - **PACF** = partial autocorrelation function; same but with intermediate lags' effects removed — read $p$ off it
    - **lag** = the series shifted $n$ steps back, used as a feature; the minimum lag fixes the forecast horizon
    - **season length (`slen` / `s`)** = the period after which the pattern repeats (24 for hourly data with a daily cycle, 30 for the currency series)
    - **moving average** = forecast/smoother using the mean of the last $k$ observations
    - **rolling window** = the $k$-observation span; `DataFrame.rolling(window).mean()` in pandas
    - **weighted average** = moving average with hand-picked weights summing to 1, heaviest on the most recent value
    - **exponential smoothing** = $\hat y_t = \alpha y_t + (1-\alpha)\hat y_{t-1}$; weights all history with exponentially decaying weight
        - **alpha ($\alpha$)** = smoothing factor for the level; how fast the model forgets
    - **double exponential smoothing (Holt)** = exponential smoothing of level + trend
        - **beta ($\beta$)** = smoothing factor for the trend component
    - **triple exponential smoothing (Holt-Winters)** = level + trend + seasonality
        - **gamma ($\gamma$)** = smoothing factor for the seasonal components
    - **level ($\ell$)** = the intercept component: where the series currently is
    - **trend ($b$)** = the slope component: which way the series is heading
    - **Brutlag method** = confidence bands from an exponentially smoothed predicted deviation; widens on shocks, then forgets
    - **SARIMA(p,d,q)(P,D,Q,s)** = seasonal autoregressive integrated moving average model
        - **AR(p)** = autoregression: the series regressed on its own past values up to lag $p$
        - **I(d)** = integration: the number of first differences taken to reach stationarity
        - **MA(q)** = moving average of the *errors*: current error depends on past errors up to lag $q$
    - **SARIMAX** = the statsmodels implementation of SARIMA (`sm.tsa.statespace.SARIMAX`)
    - **AIC** = Akaike Information Criterion; goodness of fit penalized by parameter count, lower is better — used to pick the SARIMA order
    - **residuals** = actual minus fitted values; should be stationary and uncorrelated if the model captured the structure
    - **cross-validation on a rolling basis** = train on a prefix, validate on the chunk right after, extend, repeat — the only CV that respects time
    - **TimeSeriesSplit** = the sklearn implementation of rolling-basis cross-validation
    - **forecast horizon** = how many steps ahead you predict; bounded below by your minimum lag
    - **MAE** = mean absolute error; same units as the series
    - **MedAE** = median absolute error; robust to outliers
    - **MSE** = mean squared error; penalizes large errors hardest
    - **MSLE** = MSE on $\log(1+y)$; for exponentially growing series
    - **MAPE** = mean absolute percentage error; interpretable, but undefined at $y=0$ and explosive near it
    - **$R^2$** = coefficient of determination; fraction of variance explained, $(-\infty, 1]$
    - **TNC** = truncated Newton conjugate gradient; the bounded optimizer used to fit $\alpha, \beta, \gamma$ within $[0,1]$
    - **target encoding** = replacing a category with the mean of the target in that category; must be computed on the training set only
    - **StandardScaler** = zero-mean/unit-variance rescaling; required before regularization or coefficient comparison
    - **multicollinearity** = features correlated with each other (lags badly so), which destabilizes coefficients
    - **Ridge (L2)** = regularization penalizing $\sum \beta_j^2$; shrinks coefficients toward zero but never to zero
    - **Lasso (L1)** = regularization penalizing $\sum |\beta_j|$; drives coefficients exactly to zero, doing feature selection
    - **anomaly** = an observation falling outside the model's confidence band
    - **prediction interval** = the band around a forecast, here built from rolling-CV MAE and its spread
    """)
    return


if __name__ == "__main__":
    app.run()
