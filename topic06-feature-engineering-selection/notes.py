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
    # Topic 6: Feature Engineering and Feature Selection

    **Feature engineering** is turning whatever raw mess you were handed — text, images, coordinates, timestamps, log lines — into a numeric matrix a model can actually learn from. **Feature selection** is throwing part of that matrix back out. Both are the unglamorous majority of real ML work: model choice is a one-line decision, features are where the domain knowledge lives, and a mediocre model on good features beats a good model on bad ones almost every time.

    Two habits worth adopting up front: every trick here is a *hypothesis about the data*, not a rule — verify it with cross-validation instead of trusting it; and none of it is a silver bullet, including the parts that look like best practice.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    The running example is the **Renthop** dataset (Two Sigma Connect: Rental Listing Inquiries) — New York apartment listings with price, bedrooms, coordinates, and a creation timestamp. Rich enough to demonstrate geo, date, and interaction features on one table.
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path
    from pprint import pprint

    import numpy as np
    import pandas as pd

    DATA_PATH = Path("./data")
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    FILE_NAME = "renthop_train.json.gz"
    FILE_URL = "https://drive.google.com/uc?id=1N8eav5HWd6A7rG0ZCRo7rNFryqZS4tVg"

    # requires `gdown` on PATH
    if not (DATA_PATH / FILE_NAME).exists():
        os.system(f"gdown {FILE_URL} -O {DATA_PATH / FILE_NAME}")

    df = pd.read_json(
        DATA_PATH / FILE_NAME, compression="gzip", convert_dates=["created"]
    )
    df.shape
    return df, np, pprint


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Extraction

    Data almost never arrives as a ready-made matrix. Feature extraction is the step that manufactures one. What follows is a tour by data type — text, images, geo, datetime, web — because the techniques are type-specific and mostly don't transfer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Text: bag of words

    The pipeline is always the same three steps:

    1. **Tokenize** — split text into units (usually words). Naive splitting already loses meaning: "Santa Barbara" is one token, "rock'n'roll" isn't two. Off-the-shelf tokenizers help but still fail on slang, typos, and domain jargon.
    2. **Normalize** — **stemming** (chop to a crude root) or **lemmatization** (map to the real dictionary form) so that *cat* and *cats* stop being different features.
    3. **Vectorize** — turn the token sequence into numbers.

    The simplest vectorizer is **Bag of Words**: build a vocabulary of every word in the corpus, then represent each document as a vector of length `len(vocabulary)` counting occurrences. Here's the whole idea in a dozen lines — nobody writes it this way in practice (no stop words, no vocabulary cap, dense storage), but it makes the mechanics concrete.
    """)
    return


@app.cell
def _(np):
    texts = ["i have a cat", "you have a dog", "you and i have a cat and a dog"]

    vocabulary = list(
        enumerate(set(word for sentence in texts for word in sentence.split()))
    )
    print("Vocabulary:", vocabulary)

    def vectorize(tokens):
        vector = np.zeros(len(vocabulary))
        for i, word in vocabulary:
            vector[i] = sum(1 for w in tokens if w == word)
        return vector

    for sentence in texts:
        print(vectorize(sentence.split()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The trap: **bag of words throws away word order**. "i have no cows" and "no, i have cows" vectorize identically despite meaning the opposite. The fix is to change what counts as a token — use **N-grams**, sequences of N consecutive tokens. `CountVectorizer(ngram_range=(1, 2))` keeps unigrams *and* bigrams, so `no have` and `have no` become separate features and the two sentences finally differ.

    Cost: vocabulary size explodes with N. That's the whole tradeoff — order information bought with dimensionality.
    """)
    return


@app.cell
def _():
    from sklearn.feature_extraction.text import CountVectorizer

    _vect = CountVectorizer(ngram_range=(1, 1))
    # unigrams only: the two sentences are indistinguishable
    print(_vect.fit_transform(["no i have cows", "i have no cows"]).toarray())
    print(_vect.vocabulary_)
    _vect = CountVectorizer(ngram_range=(1, 2))
    print(_vect.fit_transform(["no i have cows", "i have no cows"]).toarray())
    # add bigrams: now they separate
    print(_vect.vocabulary_)
    return (CountVectorizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tokens don't have to be words. **Character N-grams** (`analyzer="char_wb"`) capture sub-word similarity, which makes them robust to typos and good for fuzzy matching of names. Below, *andersen* and *petersen* land closer together than either does to *smith* — the vectorizer never saw a dictionary, it just noticed shared letter triples.
    """)
    return


@app.cell
def _(CountVectorizer):
    from scipy.spatial.distance import euclidean

    _vect = CountVectorizer(ngram_range=(3, 3), analyzer="char_wb")
    n1, n2, n3, n4 = _vect.fit_transform(
        ["andersen", "petersen", "petrov", "smith"]
    ).toarray()
    (euclidean(n1, n2), euclidean(n2, n3), euclidean(n3, n4))
    return (euclidean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Text: TF-IDF and embeddings

    Raw counts overweight words that are common everywhere ("the") and underweight the rare ones that actually identify a document. **TF-IDF** fixes the weighting: scale each term's frequency in a document by how rare it is across the whole corpus.

    $$\text{idf}(t, D) = \log \frac{|D|}{\text{df}(d, t) + 1}$$

    $$\text{tfidf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$$

    Bag-of-words thinking generalizes past text — **bag of sites** for browsing sessions, **bag of apps** for mobile users. Any sequence of discrete tokens works.

    The modern alternative is **embeddings** (Word2Vec, GloVe, FastText): dense low-dimensional vectors learned so that words in similar contexts sit near each other, which gives you the famous `king − man + woman ≈ queen` arithmetic. Embeddings can be extended to sentences and documents (doc2vec), but note they are trained on a corpus — a general-purpose model may be useless on specialized text. Rule of thumb: **TF-IDF plus a linear model is the baseline to beat**, and it is embarrassingly often good enough.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Images

    Before deep learning, image features were hand-built: corners, edges, region borders, color histograms (`skimage` and friends). Today the default is to grab a **pretrained** CNN, chop off the classification head, and use the network's penultimate activations as a feature vector — or **fine-tune** it, retraining the last layers (or all of them with a small learning rate) on your data. Which one depends on how much labelled data you have and how far your images are from ImageNet.

    But hand-made features didn't die. For predicting rental-listing popularity, "average pixel value" is a legitimate feature — bright apartments get more clicks. And if the signal is *text printed on the image*, OCR beats a CNN: `pytesseract` reads it directly.
    """)
    return


@app.cell
def _():
    # requires the `tesseract` binary installed on the system
    from io import BytesIO

    import requests
    from PIL import Image
    import pytesseract

    def ocr(url):
        return pytesseract.image_to_string(
            Image.open(BytesIO(requests.get(url).content))
        )

    # a clean logo: reads perfectly
    print(ocr("http://ohscurrent.org/wp-content/uploads/2015/09/domus-01-google.jpg"))

    # an apartment floor plan: room labels come out mangled — OCR is not a silver bullet
    print(
        ocr("https://habrastorage.org/webt/mj/uv/6o/mjuv6olsh1x9xxe1a6zjy79u1w8.jpeg")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Don't forget the **metadata** no network can see: image **EXIF** carries camera make and model, resolution, whether the flash fired, GPS coordinates, and the editing software used. Sometimes that's the entire signal — "photo taken on a phone with no flash" says a lot about a listing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Geospatial data

    Geo data shows up as addresses or `(latitude, longitude)` pairs, and you'll want both directions: **geocoding** (address → point) and **reverse geocoding** (point → address). Google Maps and OpenStreetMap expose both through APIs, wrapped conveniently by `geopy`. At volume you hit rate limits and HTTP latency, so a local OSM instance becomes the answer. For small data and modest needs, `reverse_geocoder` works entirely offline.
    """)
    return


@app.cell
def _(df, pprint):
    import reverse_geocoder as revgc

    results = revgc.search(list(zip(df.latitude, df.longitude)))
    pprint(results[:2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Quality caveats worth internalizing: addresses have typos, so cleaning is mandatory; coordinates have fewer typos but drift from GPS noise in tunnels and downtown canyons. Worse, mobile devices often locate via **WiFi SSID/MAC lookup**, which teleports users — a router's firmware standardized across cities (or a company moving offices with its routers) can put someone in Manhattan and Chicago on consecutive rows.

    The real value is what surrounds the point. Distance to the nearest subway, number of floors in the building, ATMs within 500m, distance to the nearest store — invent these from domain knowledge and pull them from external sources. Outside cities, think elevation and terrain instead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Date and time

    Day of week is easy: seven dummies via one-hot, plus a binary `is_weekend` on top since the weekday/weekend split usually carries more signal than the individual days.
    """)
    return


@app.cell
def _(df):
    df["dow"] = df["created"].apply(lambda x: x.date().weekday())
    df["is_weekend"] = df["dow"].isin((5, 6)).astype(int)
    df[["created", "dow", "is_weekend"]].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hour is the trap.** Treat it as a real number and you assert `0 < 23`, which is false in the way that matters: 23:00 and 00:00 are one hour apart, not twenty-three. One-hot it instead and you breed 24 features while destroying all proximity — 22 and 23 become as unrelated as 22 and 7.

    The fix is **cyclical encoding**: project time onto a circle and keep both coordinates.

    $$x = \cos\left(\frac{2\pi v}{T}\right), \qquad y = \sin\left(\frac{2\pi v}{T}\right)$$

    Now 23 and 1 are genuinely close, and — as the checks below show — exactly as close as 9 and 11, while 9 and 21 sit at the maximum distance of 2.0 (opposite sides of the circle). This matters most for anything distance-based: kNN, SVM, k-means. Honest caveat: on real problems the gap between encodings often shows up in the third decimal of the metric. Know the trick, don't build a religion around it.

    Calendar features deserve their own mention: pay days drive cash withdrawals, month starts drive transit-pass purchases. When working with time series, keep a calendar of public holidays, abnormal weather, and one-off events (Chinese New Year, a city marathon, an inauguration) — they are all anomalies your model would otherwise be blamed for.
    """)
    return


@app.cell
def _(euclidean, np):
    def make_harmonic_features(value, period=24):
        value *= 2 * np.pi / period
        return np.cos(value), np.sin(value)

    print(
        euclidean(make_harmonic_features(23), make_harmonic_features(1))
    )  # adjacent hours across midnight
    print(
        euclidean(make_harmonic_features(9), make_harmonic_features(11))
    )  # same gap, mid-day: identical distance
    print(
        euclidean(make_harmonic_features(9), make_harmonic_features(21))
    )  # opposite: 2.0, the maximum
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time series and web data

    For time series, `tsfresh` generates hundreds of candidate features automatically — a reasonable starting point when you have no domain intuition.

    Web data hides a lot in the **User Agent** string: parse it for OS family, browser, `is_mobile`, `is_bot`. Then get creative — "how far behind the latest browser version is this user?" is a real feature about how a person maintains their machine. Also mine the referrer, `http_accept_language`, and the IP address (country, city, ISP, mobile vs fixed — noisy, thanks to proxies and stale databases). Combining signals is where it gets interesting: a Chilean proxy IP with an `ru_RU` browser locale earns an `is_traveler_or_proxy_user` flag.
    """)
    return


@app.cell
def _():
    # pip install pyyaml ua-parser user-agents
    import user_agents

    ua = user_agents.parse(
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Ubuntu Chromium/56.0.2924.76 Chrome/56.0.2924.76 Safari/537.36"
    )

    print("Is a bot?", ua.is_bot)
    print("Is mobile?", ua.is_mobile)
    print("Is PC?", ua.is_pc)
    print("OS:", ua.os.family, ua.os.version)
    print("Browser:", ua.browser.family, ua.browser.version)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature transformations

    ### Normalization and changing distribution

    **Monotonic transformations** (log, scaling) are critical for some algorithms and completely irrelevant to others. Trees only care about the *order* of values, so random forests and gradient boosting are immune — one of the quiet reasons for their popularity. Everything else has requirements: parametric methods want roughly symmetric, unimodal distributions; kNN outputs nonsense if one feature spans (0, 1) and another spans (0, 1000), because the second feature is now the distance metric. There's also the pure engineering use: `np.log` keeps huge numbers inside `float64`.

    **StandardScaler** (a.k.a. Z-score) subtracts the mean and divides by the standard deviation, giving mean 0 and variance 1.

    The critical misconception to kill right now: **standardization does not make data normal**. The Shapiro–Wilk p-value below is unchanged after scaling — a tiny p-value means we reject the null hypothesis of normality both times. Scaling shifts and stretches; it does not reshape.
    """)
    return


@app.cell
def _():
    from scipy.stats import beta, lognorm, shapiro
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    _data = beta(1, 10).rvs(1000).reshape(-1, 1)
    print(shapiro(_data))
    print(
        shapiro(StandardScaler().fit_transform(_data))
    )  # not normal  # still not normal — same statistic
    return MinMaxScaler, StandardScaler, lognorm, shapiro


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What standardization *does* buy you is some protection from outliers: a value of 100 among single digits gets squashed to ~3.16 instead of dominating on the raw scale. The transform is no mystery — it's exactly `(data - data.mean()) / data.std()`.

    **MinMaxScaler** is the other common choice, mapping everything into a fixed interval (usually (0, 1)):

    $$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

    Rule of thumb: they're largely interchangeable, but **use StandardScaler when the algorithm computes distances** and MinMax when you need a bounded range (e.g. scaling to (0, 255) for visualization). Note MinMax is *not* outlier-safe — one extreme value compresses everything else into a sliver near zero, exactly as below.
    """)
    return


@app.cell
def _(MinMaxScaler, StandardScaler, np):
    _data = (
        np.array([1, 1, 0, -1, 2, 1, 2, 3, -2, 4, 100])
        .reshape(-1, 1)
        .astype(np.float64)
    )
    print(StandardScaler().fit_transform(_data).flatten())
    print(((_data - _data.mean()) / _data.std()).flatten())  # the 100 becomes ~3.16
    print(MinMaxScaler().fit_transform(_data).flatten())  # identical: no magic
    print(
        ((_data - _data.min()) / (_data.max() - _data.min())).flatten()
    )  # the 100 pins to 1.0, everything else ~0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If a feature is **log-normally** distributed, taking the log makes it normal outright — watch the Shapiro p-value jump from ~1e-44 (definitively not normal) to ~0.63 (no reason to reject normality). Log-normal describes a lot of real quantities: salaries, security prices, city populations, comment counts.

    The distribution doesn't have to be truly log-normal for this to help. **Any heavy right tail is a candidate.** For more general cases, reach for **Box-Cox** (log is a special case) or **Yeo-Johnson** (also handles negatives), or just try `np.log(x + const)` when zeros are in the way.
    """)
    return


@app.cell
def _(lognorm, np, shapiro):
    _data = lognorm(s=1).rvs(1000)
    print(shapiro(_data))
    print(shapiro(np.log(_data)))  # p ~ 1e-44  # p ~ 0.63 — normal after logging
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Shapiro–Wilk is the formal test; the **Q-Q plot** is the informal one, and on real data it's the more useful of the two. Plot your data's quantiles against a normal distribution's: a straight diagonal means normal, and the shape of the deviation tells you *how* it's non-normal.

    Run the four plots below and the lesson is visual and immediate: the raw `price` curves away from the diagonal; **StandardScaler and MinMaxScaler change nothing about the shape** — the axis numbers move, the curve doesn't; only the log straightens it out. Scaling and distribution-fixing are different jobs.
    """)
    return


@app.cell
def _(MinMaxScaler, StandardScaler, df, np):
    import statsmodels.api as sm

    price = df.price[(df.price <= 20000) & (df.price > 500)]
    # trim the most extreme prices by hand for clarity
    price_log = np.log(price)
    price_z = (
        StandardScaler()
        .fit_transform(price.values.reshape(-1, 1).astype(np.float64))
        .flatten()
    )
    price_mm = (
        MinMaxScaler()
        .fit_transform(price.values.reshape(-1, 1).astype(np.float64))
        .flatten()
    )
    for _name, x in [
        ("raw", price),
        ("z-scored", price_z),
        ("min-max", price_mm),
        ("logged", price_log),
    ]:
        sm.qqplot(np.asarray(x), loc=np.mean(x), scale=np.std(x)).suptitle(
            f"Q-Q: {_name}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactions

    Less math, more domain sense. Rental listings carry `price` and `bedrooms`, but a renter compares **price per bedroom** — that ratio is the quantity with meaning, and a linear model cannot construct it on its own (it can add features, not divide them). Building it by hand is the entire point of interaction features.

    Note the `max(x, 0.5)` guard: studios have `bedrooms == 0`, and 0.5 is an arbitrary-but-reasonable stand-in that avoids dividing by zero.

    Restraint matters here. With few features you can generate every pairwise interaction and let the selection methods below prune them — that's what `PolynomialFeatures` does. But those generated features have no physical meaning and destroy interpretability, so it's a trade, not a free win.
    """)
    return


@app.cell
def _(df):
    rooms = df["bedrooms"].apply(
        lambda x: max(x, 0.5)
    )  # studios are 0 bedrooms; 0.5 avoids div-by-zero
    df["price_per_bedroom"] = df["price"] / rooms
    df[["price", "bedrooms", "price_per_bedroom"]].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filling in the missing values

    Most algorithms refuse missing values; most real data has them. No creativity required — `pandas.DataFrame.fillna` and `sklearn.impute.SimpleImputer` cover it, and the strategies are few:

    - **Separate blank category** (`"n/a"`) for categoricals — makes "missing" itself a signal.
    - **Most probable value** — mean or median for numerics, mode for categoricals.
    - **An extreme sentinel value** — counterintuitive but good for trees, which can then split missing away from non-missing.
    - **Adjacent value** (forward/backward fill) — for ordered data like time series.

    The trap is `df = df.fillna(0)` on autopilot. Data preparation takes longer than modeling for a reason: thoughtless filling **hides bugs in your processing pipeline** and quietly poisons the model. Look at *why* each gap exists first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature selection

    Why delete features you worked to build? Two reasons:

    1. **Cost** — more columns, more compute. Irrelevant on toy data, very tangible in a loaded production system with hundreds of extra features.
    2. **Overfitting** — some algorithms take noise as signal. Non-informative features actively make models worse, not just slower.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistical approaches

    The dumbest useful heuristic: a feature that never changes carries zero information, so **low variance is a proxy for low usefulness**. `VarianceThreshold` cuts everything below a cutoff. Note the numbers below — the threshold is on raw variance, so it is scale-dependent and completely unaware of the target. It's a cheap first pass, not a strategy.
    """)
    return


@app.cell
def _():
    from sklearn.datasets import make_classification
    from sklearn.feature_selection import VarianceThreshold

    x_data_generated, y_data_generated = make_classification(random_state=17)
    print(x_data_generated.shape)

    for t in (0.7, 0.8, 0.9):
        print(t, VarianceThreshold(t).fit_transform(x_data_generated).shape)
    return VarianceThreshold, x_data_generated, y_data_generated


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Better: **univariate selection**, which does look at the target. `SelectKBest(f_classif, k=5)` scores each feature against `y` with an ANOVA F-test and keeps the top k.

    Both selections below beat the full feature set on `neg_log_loss` (higher = better, since it's negated). That's the headline: **fewer features, better model** — `make_classification` seeds mostly-noise columns, and the classifier was drowning in them. Caveat: this data is synthetic and the effect is by construction. On real data, always confirm with cross-validation instead of assuming.
    """)
    return


@app.cell
def _(VarianceThreshold, x_data_generated, y_data_generated):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    logit = LogisticRegression(solver="lbfgs", random_state=17)

    x_data_kbest = SelectKBest(f_classif, k=5).fit_transform(
        x_data_generated, y_data_generated
    )
    x_data_varth = VarianceThreshold(0.9).fit_transform(x_data_generated)

    def score(x):
        return cross_val_score(
            logit, x, y_data_generated, scoring="neg_log_loss", cv=5
        ).mean()

    print("all features: ", score(x_data_generated))
    print("SelectKBest:  ", score(x_data_kbest))
    print("VarianceThresh:", score(x_data_varth))
    return cross_val_score, logit, score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Selection by modeling

    Let a model tell you which features matter. Two families are standard: a **tree ensemble** (random forest — read off `feature_importances_`) or a **linear model with Lasso (L1) regularization**, which drives weak features' weights to exactly zero. The logic is that if a feature is useless to a simple model, it's not worth dragging into a complex one.

    `SelectFromModel` wraps this and drops into a pipeline. Below, logistic regression *with* RF-based selection beats both plain logistic regression and the random forest itself — the RF finds the informative columns, the linear model exploits them cleanly.

    **Always do this inside a `Pipeline`, never before the split.** Selecting features using the whole dataset and then cross-validating leaks the target into the selection step and inflates your score. The pipeline re-runs selection inside each fold, which is the only honest way to measure it.
    """)
    return


@app.cell
def _(cross_val_score, logit, score, x_data_generated, y_data_generated):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.pipeline import make_pipeline

    rf = RandomForestClassifier(n_estimators=10, random_state=17)
    pipe = make_pipeline(SelectFromModel(estimator=rf), logit)

    print("LR:            ", score(x_data_generated))
    print(
        "RF:            ",
        cross_val_score(
            rf, x_data_generated, y_data_generated, scoring="neg_log_loss", cv=5
        ).mean(),
    )
    print(
        "LR + selection:",
        cross_val_score(
            pipe, x_data_generated, y_data_generated, scoring="neg_log_loss", cv=5
        ).mean(),
    )
    return SelectFromModel, make_pipeline, rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same comparison with scaling folded in, as you'd actually build it. It still comes out ahead here — but this is not guaranteed, and selection can absolutely make a model worse. Measure, don't assume.
    """)
    return


@app.cell
def _(
    SelectFromModel,
    StandardScaler,
    cross_val_score,
    logit,
    make_pipeline,
    rf,
    x_data_generated,
    y_data_generated,
):
    pipe1 = make_pipeline(StandardScaler(), SelectFromModel(estimator=rf), logit)
    pipe2 = make_pipeline(StandardScaler(), logit)
    for _name, est in [
        ("LR + selection: ", pipe1),
        ("LR:             ", pipe2),
        ("RF:             ", rf),
    ]:
        print(
            _name,
            cross_val_score(
                est, x_data_generated, y_data_generated, scoring="neg_log_loss", cv=5
            ).mean(),
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Grid search over feature subsets

    The most reliable method, and the most expensive: just try the subsets. Train on a subset, record the score, repeat, take the best.

    - **Exhaustive Feature Selection** — every possible combination. Correct, and combinatorially hopeless beyond a handful of features.
    - **Sequential Feature Selection (forward)** — find the best combination of N features, fix it, then try each remaining feature as the (N+1)th. Greedy, so it can miss the true optimum, but it makes the search tractable.
    - **Sequential Feature Selection (backward)** — the same thing reversed: start with everything, drop features one at a time while quality holds up.

    Stop when the score plateaus or you hit your feature budget. `mlxtend` implements both directions; sklearn ships `SequentialFeatureSelector` too.
    """)
    return


@app.cell
def _():
    # pip install mlxtend
    # from mlxtend.feature_selection import SequentialFeatureSelector
    #
    # selector = SequentialFeatureSelector(
    #     logit, scoring="neg_log_loss", verbose=2, k_features=3, forward=False, n_jobs=-1
    # )
    # selector.fit(x_data_generated, y_data_generated)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Features beat models.** Model choice is one line; features are where domain knowledge enters and where most of the score lives.
    - **Extraction is type-specific.** Text → tokenize, normalize, vectorize (BoW → N-grams → TF-IDF → embeddings). Images → pretrained CNN, but don't overlook OCR, pixel stats, and EXIF. Geo → reverse geocode, then engineer *surroundings*. Web → parse the User Agent and IP, then combine signals.
    - **N-grams buy word order with dimensionality.** Character N-grams handle typos and name matching.
    - **TF-IDF + linear model is the text baseline.** Beat it before reaching for anything neural.
    - **Scaling ≠ normalizing.** StandardScaler and MinMaxScaler do not change distribution shape; only log/Box-Cox/Yeo-Johnson do. The Q-Q plot shows this in one glance.
    - **StandardScaler for distance-based algorithms**, MinMax for bounded ranges. Trees don't care about either — they only see order.
    - **Cyclical time needs cyclical encoding.** Hour-as-a-number claims 23 and 0 are far apart; project onto a circle instead.
    - **Interactions encode domain logic** a linear model can't derive — price per bedroom, not price and bedrooms.
    - **Don't `fillna(0)` reflexively.** It hides pipeline bugs. Decide per column, and let trees exploit a sentinel value.
    - **Select features to fight both compute cost and overfitting.** Escalate: variance threshold → univariate (`SelectKBest`) → model-based (`SelectFromModel`) → sequential search.
    - **Selection goes inside the Pipeline.** Selecting on the full dataset before cross-validation leaks the target and flatters your score.
    - **Nothing here is a silver bullet.** Every transformation is a hypothesis; cross-validation is the referee.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    This module covers the two halves of turning raw data into a model-ready matrix. **Extraction** is a type-by-type toolkit: text gets tokenized, stemmed/lemmatized, and vectorized via bag of words, N-grams, TF-IDF, or embeddings; images get a pretrained CNN's activations, or OCR and EXIF when the signal isn't visual; geospatial data gets reverse-geocoded and enriched with invented neighborhood features; datetimes get one-hot day-of-week plus cyclical encoding for hours; web logs get User Agent and IP parsing. **Transformation** covers scaling (StandardScaler for distance-based models, MinMax for bounded ranges — neither changes distribution shape), distribution-fixing via log/Box-Cox/Yeo-Johnson verified with Shapiro–Wilk or a Q-Q plot, hand-built interaction features that encode domain logic, and deliberate missing-value strategies. **Selection** then prunes the result — variance thresholds, univariate F-tests, model-based importances, and sequential search — to cut both compute cost and overfitting, always inside a Pipeline so cross-validation stays honest. The through-line: every technique is a hypothesis about the data, tree models are immune to most of the transformations, and cross-validation is the only thing that settles an argument.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **feature engineering** = manufacturing model-ready numeric features from raw data
    - **feature extraction** = building a feature matrix out of non-tabular data (text, images, geo, time)
    - **feature selection** = discarding features to cut compute cost and overfitting
    - **tokenization** = splitting text into units (tokens, usually words)
    - **stemming** = chopping a word to a crude root
    - **lemmatization** = mapping a word to its dictionary form
    - **Bag of Words** = vector of per-word counts over a fixed vocabulary; discards word order
        - **N-gram** = sequence of N consecutive tokens, used to recover word order
        - **character N-gram** = N-gram of letters; robust to typos and good for name matching
    - **stop words** = ultra-common words usually dropped before vectorizing
    - **`CountVectorizer`** = sklearn's Bag of Words / N-gram vectorizer
    - **TF-IDF** = term frequency scaled by inverse document frequency; upweights rare, document-specific words
    - **embedding** = dense low-dimensional learned word vector (Word2Vec, GloVe, FastText)
    - **fine-tuning** = retraining a pretrained network's layers on your own data
    - **OCR** = reading text printed inside an image (`pytesseract`)
    - **EXIF** = image metadata: camera, resolution, flash, GPS, software
    - **geocoding** = address → coordinates
    - **reverse geocoding** = coordinates → address
    - **one-hot encoding** = one binary column per category value
    - **cyclical encoding** = projecting periodic values onto a circle via cos/sin so wraparound distances are correct
    - **monotonic transformation** = order-preserving transform; invisible to tree models
    - **StandardScaler** = subtract mean, divide by std (Z-score); default for distance-based algorithms
    - **MinMaxScaler** = rescale into a fixed interval, usually (0, 1); not outlier-safe
    - **log transform** = fixes heavy right tails; turns a log-normal feature normal
    - **Box-Cox** = family of power transforms generalizing the log
    - **Yeo-Johnson** = Box-Cox extended to negative values
    - **Shapiro–Wilk test** = formal normality test; small p-value ⇒ not normal
    - **Q-Q plot** = quantiles vs. a reference distribution; a straight diagonal means normal
    - **interaction feature** = a feature combining others by domain logic (e.g. price per bedroom)
    - **`PolynomialFeatures`** = auto-generates all polynomial interactions; uninterpretable but useful for linear models
    - **imputation** = filling in missing values (mean/median/mode, sentinel, or forward-fill)
    - **`VarianceThreshold`** = drops features with variance below a cutoff; ignores the target
    - **univariate selection** = scoring each feature against the target independently (`SelectKBest`, `f_classif`)
    - **`SelectFromModel`** = keeps features a fitted model rates as important
    - **Lasso (L1) regularization** = penalty that zeroes out weak features' weights
    - **Exhaustive Feature Selection** = evaluate every possible feature subset
    - **Sequential Feature Selection** = greedy add (forward) or drop (backward) of one feature at a time
    - **`Pipeline`** = chains transforms and an estimator so cross-validation refits each step per fold, preventing leakage
    - **data leakage** = letting information from outside a training fold influence it, inflating the measured score
    """)
    return


if __name__ == "__main__":
    app.run()
