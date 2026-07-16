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
    # Topic 9 — Part 2: Predicting the Future with Facebook Prophet

    **Prophet** is Facebook's forecasting library, built for *business* time series: the kind with human-driven weekly/yearly cycles, holiday dips, trend breaks when a product ships, and outliers. Its selling point isn't accuracy records — it's that a non-specialist can get an analyst-quality forecast out of the defaults, and then improve it by turning knobs that actually mean something (changepoints, seasonalities, holiday lists) instead of staring at ACF plots picking ARIMA orders.

    The paper is called *Forecasting at Scale*, but "scale" here means **many analysts × many forecasting problems × automated quality checks**, not big-data infrastructure. Keep that framing: Prophet trades a bit of statistical rigor for accessibility and volume.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model: an additive decomposition

    Prophet is not autoregressive. It never looks at $y_{t-1}$ to predict $y_t$. It's a **curve-fitting regression on time itself**:

    $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

    - $g(t)$ — **trend**: the non-periodic drift.
    - $s(t)$ — **seasonality**: the periodic part (weekly, yearly, daily).
    - $h(t)$ — **holidays**: irregular but *known-in-advance* spikes/dips you hand it.
    - $\epsilon_t$ — noise, assumed normal, absorbing everything the three terms missed.

    This is why Prophet has no stationarity requirement, tolerates gaps and missing days, and can extrapolate arbitrarily far into the future — and also why it can be badly wrong in ways an ARIMA wouldn't: it has no memory of recent errors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trend $g(t)$: saturating vs. piecewise linear

    Two choices:

    1. **Nonlinear saturating growth** — a logistic curve
       $$g(t) = \frac{C}{1 + e^{-k(t-m)}}$$
       with carrying capacity $C$ (the ceiling), growth rate $k$ (steepness), offset $m$. Use it when growth *must* level off — market saturation, app installs bounded by the addressable population. $C$ and $k$ need not be constant; Prophet lets them shift at changepoints.
    2. **Piecewise linear** — the default. Constant growth rate between **changepoints**, where the rate is allowed to jump.

    The key mechanism in both is **changepoints**. Prophet auto-selects candidate dates by fitting history, but you can pass them yourself — and you should, when you *know* the dates (a launch, a pricing change, a redesign). That's the one place domain knowledge buys the most accuracy per unit of effort.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seasonality $s(t)$ and holidays $h(t)$

    **Weekly** seasonality is dummy variables — six of them (Mon–Sat). Sunday is deliberately left out: including all seven makes them a linear combination of the intercept and the fit degenerates. This is just the standard dummy-variable trap.

    **Yearly** seasonality is a **Fourier series** — a small sum of sines/cosines. That's what makes the yearly curve smooth and lets you control its flexibility with a single integer (number of terms) rather than 365 dummies.

    Since v0.2 Prophet also handles **sub-daily** series and daily seasonality.

    **Holidays** are the component you have to supply: a table of dates you already know are abnormal (Christmas, Black Friday, your own release days). Prophet can't infer these — irregular-schedule events look like outliers to a seasonality term. Supplying them is often the single biggest win.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the benchmarks were measured: MAPE and MAE

    Prophet's paper compares against `naive` (predict the last observed value forever), `snaive` (predict the same point one season back), and other standard baselines, scoring with **MAPE**.

    With actual $y_i$ and forecast $\hat{y}_i$:

    - forecast error $e_i = y_i - \hat{y}_i$
    - relative error $p_i = e_i / y_i$
    - $\text{MAPE} = \text{mean}(|p_i|)$ — error as a **percentage**, so it's comparable across datasets of different scale.
    - $\text{MAE} = \text{mean}(|e_i|)$ — error in the **original units**, so you know what the mistake actually costs.

    Report both. MAPE alone lies when $y$ gets near zero (it explodes); MAE alone tells you nothing about whether 70 is a lot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    `pip install prophet` (the old package name was `fbprophet`). Prophet compiles a Stan model under the hood, so the install is heavier than a typical pip package.
    """)
    return


@app.cell
def _():
    import logging
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt

    from prophet import Prophet

    # Prophet/cmdstanpy is chatty during fit
    logging.getLogger().setLevel(logging.ERROR)
    return Prophet, np, pd, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset: daily posts published on Medium

    The task: forecast the number of posts published on Medium per day. The raw file is one row per post, with a `published` timestamp and a `url` that identifies it.

    Two cleaning steps worth naming:

    - **`published` arrives as a string.** `pd.to_datetime` is mandatory before any time-series operation — pandas won't resample a string column.
    - **The dates are dirty.** Medium launched 2012-08-15, yet rows carry 1970 epoch-ish timestamps. Those are junk, not history. Trim the series to a window you believe in rather than letting the model fit garbage at the left edge.
    """)
    return


@app.cell
def _(pd):
    # tab-separated dump of Medium posts; adjust the path if you keep the file locally
    DATA_PATH = "medium_posts.csv"
    df = pd.read_csv(DATA_PATH, sep="\t")

    # keep only the time dimension and the post identifier
    df = df[["published", "url"]].dropna().drop_duplicates()
    df["published"] = pd.to_datetime(df["published"])

    # trim to the period we trust: launch date -> end of data
    df = df[(df["published"] > "2012-08-15") & (df["published"] < "2017-06-26")].sort_values(
        by=["published"]
    )
    df.head(3)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sub-daily → daily: resampling

    Counting posts per timestamp gives a **sub-daily** series — irregular intervals, one row per publication instant. We want one row per day.

    **Resampling** = re-binning a time index to a different frequency. Going to a *coarser* frequency is **downsampling**. `resample("D").sum()` does it: bin by calendar day, add up the counts inside each bin. `resample` only works on a proper DatetimeIndex — that's why the `to_datetime` step above wasn't optional.
    """)
    return


@app.cell
def _(df):
    aggr_df = df.groupby("published")[["url"]].count()
    aggr_df.columns = ["posts"]

    daily_df = aggr_df.resample("D").sum()
    daily_df.head(3)
    return (daily_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory visual analysis

    Always plot the whole range first — a long window is where seasonality and structural breaks announce themselves.

    Plotting the **daily** series as-is is nearly useless: the high-frequency noise buries everything except an obvious accelerating upward trend. So downsample to **weekly** bins purely for the eyeballing. Binning is the crudest noise-reduction tool; moving-average and exponential smoothing are the alternatives.

    Keep the weekly frame in a *separate* variable — the model still trains on the daily series.
    """)
    return


@app.cell
def _(daily_df, plt):
    weekly_df = daily_df.resample('W').sum()
    _fig, _axes = plt.subplots(2, 1, figsize=(12, 7))
    daily_df.plot(ax=_axes[0], title='Posts on Medium (daily)', legend=False)
    weekly_df.plot(ax=_axes[1], title='Posts on Medium (weekly)', legend=False)
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trimming the early years

    Drop everything before 2015. Two reasons, and they're the general rule for this move:

    1. The 2012–2014 volumes are tiny — they add nothing to a 2017 forecast.
    2. Worse, they *hurt*. Least-squares weights every point equally, so the model burns capacity fitting an era whose dynamics no longer exist, at the cost of the recent data you actually care about.

    Verdict from the visual pass: **non-stationary, strong growing trend, weekly + yearly seasonality, a handful of abnormal days per year.** That's exactly Prophet's home turf.
    """)
    return


@app.cell
def _(daily_df):
    daily_df_1 = daily_df.loc[daily_df.index >= '2015-01-01']
    daily_df_1.head(3)
    return (daily_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Making a forecast

    Prophet's API mirrors sklearn: construct → `fit` → `predict`. The one non-negotiable is the input contract — `fit` wants a DataFrame with exactly two columns named:

    - **`ds`** — the datestamp (`date`/`datetime`).
    - **`y`** — the numeric value to predict.

    Rename anything else and it errors. Also strip the timezone: Prophet chokes on tz-aware datetimes.

    The authors want **at least several months, ideally >1 year** of history. We have 2.5 years.

    For evaluation, hold out the **last 30 days**. With time series you never shuffle-split — the future must stay in the future, or you leak.
    """)
    return


@app.cell
def _(daily_df_1):
    df_1 = daily_df_1.reset_index()
    df_1.columns = ['ds', 'y']
    df_1['ds'] = df_1['ds'].dt.tz_convert(None)  # Prophet needs tz-naive datetimes
    prediction_size = 30
    train_df = df_1[:-prediction_size]
    train_df.tail(3)
    return df_1, prediction_size, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit, then build the future frame

    `make_future_dataframe(periods=n)` returns **history + n future dates**. Feeding the whole thing to `predict` gives you the out-of-sample forecast *and* an in-sample fit for the history — useful, since you need the historical rows to plot actual-vs-fitted anyway.

    The output of `predict` is a wide frame: `trend`, `weekly`, `yearly`, each with `_lower`/`_upper` intervals. The forecast itself is **`yhat`**, bracketed by `yhat_lower`/`yhat_upper`.
    """)
    return


@app.cell
def _(Prophet, prediction_size, train_df):
    m = Prophet()
    m.fit(train_df)

    future = m.make_future_dataframe(periods=prediction_size)
    forecast = m.predict(future)
    forecast[["ds", "trend", "weekly", "yearly", "yhat_lower", "yhat", "yhat_upper"]].tail(3)
    return forecast, m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The two built-in plots

    - **`m.plot(forecast)`** — every point plus the forecast band. On a noisy series like this it's close to useless: all you learn is that the model treats a lot of points as outliers.
    - **`m.plot_components(forecast)`** — the one that earns its keep. It decomposes the fit into trend, weekly, yearly (and holidays, if you supplied them), each on its own axis.

    The components plot is where Prophet's interpretability lives: trend picks up the late-2016 acceleration, weekly shows the weekend dip in posting, yearly shows a sharp Christmas trough. Every one of those is a claim you can sanity-check against reality — which is exactly what a black-box forecaster won't give you.
    """)
    return


@app.cell
def _(forecast, m):
    m.plot(forecast);
    m.plot_components(forecast);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecast quality evaluation

    `forecast` has everything except the ground truth, so join it back onto the actuals on `ds`. Two small helpers, reused for both models below.

    Then compute MAPE and MAE **on the held-out tail only** — scoring on the in-sample history would flatter the model badly.
    """)
    return


@app.cell
def _(df_1, forecast, np, prediction_size):
    def make_comparison_dataframe(historical, forecast):
        """Join history with forecast -> columns yhat, yhat_lower, yhat_upper, y."""
        return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))

    def calculate_forecast_errors(df, prediction_size):
        """MAPE and MAE over the last `prediction_size` rows."""
        df = df.copy()
        df['e'] = df['y'] - df['yhat']
        df['p'] = 100 * df['e'] / df['y']
        predicted_part = df[-prediction_size:]
        error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))  # forecast error
        return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}  # relative error, %
    cmp_df = make_comparison_dataframe(df_1, forecast)
    calculate_forecast_errors(cmp_df, prediction_size)  # only the held-out window
    return calculate_forecast_errors, cmp_df, make_comparison_dataframe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the result

    Out of the box: **MAPE ≈ 22.7%**, **MAE ≈ 70 posts**. Translation: the mean level is roughly right, but we're off by ~70 posts on a typical day.

    Plotting actuals against the forecast band over the predicted month exposes the actual failure: **many real values fall outside the confidence interval**, and the model under-fits the *growing amplitude* of the weekly swing. The weekly seasonality is additive and constant in size, but in reality the peak-to-trough gap widens as the series grows.

    That's the diagnosis worth remembering: **Prophet's default additive model assumes constant variance.** When your series' fluctuations scale with its level, the defaults will systematically under-cover.
    """)
    return


@app.cell
def _(cmp_df, plt, prediction_size):
    def show_forecast(cmp_df, num_predictions, num_values, title, ax):
        """Plot the forecast band over the predicted window against actuals."""
        pred = cmp_df.tail(num_predictions)
        act = cmp_df.tail(num_values)
        ax.fill_between(pred.index, pred['yhat_lower'], pred['yhat_upper'], color='gray', alpha=0.3, label='Confidence interval')
        ax.plot(pred.index, pred['yhat'], color='tab:blue', label='Forecast')
        ax.plot(act.index, act['y'], color='red', label='Actual')
        ax.set_title(title)
        ax.set_ylabel('Posts')
        ax.legend()
    _fig, ax = plt.subplots(figsize=(12, 5))
    show_forecast(cmp_df, prediction_size, 100, 'New posts on Medium', ax)
    return (show_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Box–Cox transformation

    The fix for level-dependent variance, and it costs nothing in model tuning: **transform the target, fit on the transformed scale, invert the predictions.**

    The one-parameter Box–Cox transform:

    $$\text{boxcox}_\lambda(y_i) = \begin{cases} \dfrac{y_i^{\lambda} - 1}{\lambda}, & \lambda \neq 0 \\[6pt] \ln(y_i), & \lambda = 0 \end{cases}$$

    It's monotonic (so order is preserved) and **stabilizes variance** — it compresses the large values more than the small ones, which is exactly what a series with growing swings needs. `scipy.stats.boxcox` picks the $\lambda$ that maximizes the log-likelihood for you.

    To get back to posts-per-day you need the inverse:

    $$\text{invboxcox}_\lambda(y_i) = \begin{cases} e^{\ln(\lambda y_i + 1)/\lambda}, & \lambda \neq 0 \\[6pt] e^{y_i}, & \lambda = 0 \end{cases}$$

    **Trap:** you must invert `yhat_lower` and `yhat_upper` too, not just `yhat`. Forget them and your interval is silently on the wrong scale. (And because the transform is nonlinear, the inverted mean is no longer strictly the mean — a technicality worth knowing, not worth worrying about here.)
    """)
    return


@app.cell
def _(np):
    def inverse_boxcox(y, lambda_):
        return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)

    return (inverse_boxcox,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Same model, transformed target

    Nothing about the Prophet call changes — same defaults, same 30-day horizon. Only `y` is different.
    """)
    return


@app.cell
def _(
    Prophet,
    calculate_forecast_errors,
    df_1,
    inverse_boxcox,
    make_comparison_dataframe,
    prediction_size,
    stats,
    train_df,
):
    train_df2 = train_df.copy().set_index('ds')
    train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])
    train_df2.reset_index(inplace=True)
    m2 = Prophet()
    m2.fit(train_df2)
    future2 = m2.make_future_dataframe(periods=prediction_size)
    forecast2 = m2.predict(future2)
    for column in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast2[column] = inverse_boxcox(forecast2[column], lambda_prophet)
    # invert the transform for the point forecast AND both interval bounds
    cmp_df2 = make_comparison_dataframe(df_1, forecast2)
    calculate_forecast_errors(cmp_df2, prediction_size)
    return (cmp_df2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Result: MAPE 22.7% → 11.8%, MAE 70 → 40

    Error roughly halved, with **zero model tuning** — we only changed the scale we fit on. Side by side, the transformed model tracks the weekly swings far more closely, because on the Box–Cox scale the amplitude *is* roughly constant, which is what the additive seasonality term assumes.

    Rule of thumb: **if the fluctuations of your series grow with its level, transform before you tune.** Trying to fix it with Prophet's hyperparameters first is fighting the wrong battle. (Prophet's `seasonality_mode="multiplicative"` is the other route to the same idea.)
    """)
    return


@app.cell
def _(cmp_df, cmp_df2, plt, prediction_size, show_forecast):
    _fig, _axes = plt.subplots(2, 1, figsize=(12, 9))
    show_forecast(cmp_df, prediction_size, 100, 'No transformations', _axes[0])
    show_forecast(cmp_df2, prediction_size, 100, 'Box-Cox transformation', _axes[1])
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Prophet is regression on time, not autoregression.** $y = g(t) + s(t) + h(t) + \epsilon$. No stationarity requirement, tolerates missing days, extrapolates freely — but it never looks at the previous value.
    - **The input contract is rigid:** a DataFrame with columns named exactly `ds` (tz-naive datetime) and `y`. Most first-run errors are this.
    - **Changepoints and holidays are where your domain knowledge pays.** Prophet guesses changepoints; it *cannot* guess your Black Fridays and release dates — hand them over.
    - **Never shuffle a time-series split.** Hold out the tail, score only on the tail.
    - **Report MAPE and MAE together.** MAPE is scale-free but blows up near zero; MAE is in real units but meaningless without context.
    - **Trim history you don't believe in.** Junk timestamps and a low-volume early era both drag the fit; least squares weights every point equally.
    - **Defaults assume constant variance.** Values falling outside the confidence band plus under-fit seasonal amplitude = your cue to transform.
    - **Box–Cox is the cheapest win available.** `stats.boxcox` finds $\lambda$; it roughly halved the error here without touching a hyperparameter. Just remember to invert `yhat_lower`/`yhat_upper` too.
    - **`plot_components` over `plot`.** The decomposition is the interpretability; the raw forecast plot on a noisy series tells you almost nothing.
    - **No free lunch.** Out-of-the-box Prophet isn't magic — it's a good, honest, tunable baseline that a human still has to steer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Prophet models a business time series as an additive decomposition of trend, seasonality, holidays, and noise, fit by curve-fitting against time rather than past values — which buys robustness to gaps, outliers, and non-stationarity, plus a components plot you can actually argue with. Forecasting daily Medium posts shows the full loop: clean and `to_datetime` the timestamps, downsample sub-daily events into daily bins with `resample`, trim both the junk early history and the irrelevant low-volume years, rename to `ds`/`y`, hold out the last 30 days, then `fit` / `make_future_dataframe` / `predict`. Default settings give MAPE ≈ 22.7% and MAE ≈ 70 posts, and the diagnostic plot names the culprit: the weekly amplitude grows with the level, but additive seasonality assumes it doesn't, so actuals spill outside the confidence band. Applying a Box–Cox transform to `y`, refitting the identical model, and inverting `yhat` along with both bounds cuts the error roughly in half (MAPE ≈ 11.8%, MAE ≈ 40) — the lesson being that fixing the *scale* of your target beats tuning the model, and that Prophet's value is a fast, interpretable, steerable baseline, not an oracle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **Prophet** = Facebook's open-source forecasting library for business time series, tuned for accessibility over statistical rigor
        - **`fit(df)`** = trains on a DataFrame with exactly two columns, `ds` and `y`
        - **`make_future_dataframe(periods=n)`** = builds a frame of all history dates plus `n` future dates to feed `predict`
        - **`predict(future)`** = returns a wide frame of components and their intervals; the forecast lives in `yhat`
        - **`plot(forecast)`** = built-in plot of all points plus the forecast band
        - **`plot_components(forecast)`** = built-in plot decomposing the fit into trend, weekly, yearly, holidays
    - **additive regression model** = $y(t) = g(t) + s(t) + h(t) + \epsilon_t$, the decomposition Prophet fits
    - **`ds`** = Prophet's required column name for the datestamp; must be tz-naive `date`/`datetime`
    - **`y`** = Prophet's required column name for the numeric target
    - **`yhat`** = the point forecast column, bracketed by `yhat_lower` and `yhat_upper`
    - **trend $g(t)$** = the non-periodic drift component
    - **piecewise linear model** = the default trend: constant growth rate between changepoints, jumps allowed at them
    - **nonlinear saturating growth** = the logistic trend $g(t)=C/(1+e^{-k(t-m)})$, for growth that must level off
        - **carrying capacity ($C$)** = the ceiling the logistic curve approaches
        - **growth rate ($k$)** = the steepness of the logistic curve
    - **changepoint** = a date where the trend's growth rate is permitted to change; auto-detected or supplied by the analyst
    - **seasonality $s(t)$** = the periodic component; weekly via dummy variables, yearly via a Fourier series, optionally daily
    - **Fourier series** = a sum of sines and cosines used to model smooth yearly seasonality with few parameters
    - **dummy variable** = a 0/1 indicator column; six are used for weekdays, the seventh omitted to avoid a linear dependence that breaks the fit
    - **holidays $h(t)$** = predictable abnormal days on irregular schedules; must be supplied by the analyst as a custom list
    - **error term $\epsilon_t$** = whatever the model didn't capture, assumed normally distributed
    - **sub-daily time series** = observations at intervals finer than a day, at irregular timestamps
    - **resampling** = re-binning a time index to a different frequency; requires a DatetimeIndex
        - **downsampling** = resampling to a coarser frequency (e.g. events → daily bins) via `resample("D").sum()`
    - **binning** = the crudest noise-reduction technique: aggregate into coarser time buckets
    - **forecast error ($e_i$)** = $y_i - \hat{y}_i$
    - **relative forecast error ($p_i$)** = $e_i / y_i$
    - **MAPE** = mean absolute percentage error, $\text{mean}(|p_i|)$; scale-free and comparable across datasets, but explodes as $y \to 0$
    - **MAE** = mean absolute error, $\text{mean}(|e_i|)$; error in the original units
    - **naive** = baseline that predicts every future value as the last observed value
    - **snaive** = seasonal naive baseline that predicts the value from one season earlier
    - **Box–Cox transformation** = a monotonic one-parameter transform of $y$ that stabilizes variance; `scipy.stats.boxcox` returns the transformed series and the maximum-likelihood $\lambda$
        - **$\lambda$** = the Box–Cox parameter; $\lambda = 0$ reduces the transform to $\ln(y)$
        - **inverse Box–Cox** = the back-transform returning predictions to the original scale; apply it to `yhat_lower` and `yhat_upper`, not just `yhat`
    - **non-stationary** = a series whose statistical properties (level, variance) change over time — fine for Prophet, fatal for a raw ARMA
    - **confidence interval** = the `yhat_lower`–`yhat_upper` band; actuals routinely falling outside it signals unmodeled variance
    - **holdout** = the tail slice reserved for scoring; time series are split chronologically, never shuffled
    """)
    return


if __name__ == "__main__":
    app.run()
