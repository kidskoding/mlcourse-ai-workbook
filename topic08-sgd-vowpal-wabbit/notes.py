import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Topic 8: SGD, the Hashing Trick, and Vowpal Wabbit

    Everything so far assumed the dataset fits in RAM. This topic is about what you do when it doesn't. Two ideas carry the whole thing:

    - **Stochastic gradient descent (SGD)** — update the model from *one* example at a time, so training data never has to be loaded at once.
    - **The hashing trick** — map feature names to column indices with a hash function, so you never have to build or store a vocabulary.

    Together they give you **Vowpal Wabbit** (VW), a command-line learner that trains linear models on gigabytes of text in seconds. It's the "boring old linear model, but industrially fast" tool.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gradient descent, briefly

    Gradient descent minimizes a function by stepping in the direction of steepest *decrease*. The gradient $\nabla f = \left(\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right)^T$ points toward fastest *growth*, so you move against it (the antigradient) and the function drops as fast as it can locally.

    Concretely, for simple linear regression on one feature — predict height $y_i$ from weight $x_i$ as $\hat{y}_i = w_0 + w_1 x_i$ — the loss is squared error:

    $$SE(w_0, w_1) = \frac{1}{2}\sum_{i=1}^{\ell}\left(y_i - (w_0 + w_1 x_i)\right)^2 \rightarrow \min_{w_0, w_1}$$

    Take partial derivatives and you get the update rules, stepping by a small **learning rate** $\eta$:

    $$w_0^{(t+1)} = w_0^{(t)} + \eta \sum_{i=1}^{\ell}\left(y_i - w_0^{(t)} - w_1^{(t)} x_i\right)$$
    $$w_1^{(t+1)} = w_1^{(t)} + \eta \sum_{i=1}^{\ell}\left(y_i - w_0^{(t)} - w_1^{(t)} x_i\right) x_i$$

    **The trap is that sum.** Every single weight update walks all $\ell$ training rows. At a few thousand rows that's fine. At 100 GB it's fatal — you can't even hold $X$ in memory, let alone sweep it once per step.
    """)
    return


@app.cell
def _():
    import os
    import re

    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_20newsgroups, load_files
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    import matplotlib.pyplot as plt

    DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"

    # scratch dir for the .vw files and the IMDB tarball; lives next to this notebook,
    # inside the repo. Nothing here is worth committing.
    PATH_TO_WRITE_DATA = "data/"
    os.makedirs(PATH_TO_WRITE_DATA, exist_ok=True)
    return (
        DATA_URL,
        LabelEncoder,
        LogisticRegression,
        OneHotEncoder,
        PATH_TO_WRITE_DATA,
        accuracy_score,
        classification_report,
        confusion_matrix,
        fetch_20newsgroups,
        load_files,
        np,
        os,
        pd,
        plt,
        re,
        roc_auc_score,
        roc_curve,
        train_test_split,
    )


@app.cell
def _(DATA_URL, pd, plt):
    # the toy regression setup: predict height from weight
    data_demo = pd.read_csv(DATA_URL + "weights_heights.csv")

    plt.scatter(data_demo["Weight"], data_demo["Height"])
    plt.xlabel("Weight in lb")
    plt.ylabel("Height in inches")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SGD: drop the sum

    **Stochastic gradient descent** does the obvious surgery — throw away the summation and update the weights from a *single* randomly-chosen example:

    $$w_0^{(t+1)} = w_0^{(t)} + \eta \left(y_i - w_0^{(t)} - w_1^{(t)} x_i\right)$$
    $$w_1^{(t+1)} = w_1^{(t)} + \eta \left(y_i - w_0^{(t)} - w_1^{(t)} x_i\right) x_i$$

    Each step is now a noisy, cheap estimate of the true gradient instead of an exact, expensive one. The path to the minimum wobbles instead of running straight downhill — but you take *thousands* of wobbly steps in the time batch GD takes one, and you converge much faster in wall-clock terms.

    **Why this is the whole ballgame:** an SGD update touches exactly one row. So the data can live on disk, be read one line at a time, and never be held in RAM. That's the crack that lets you train on more data than your machine has memory for.

    ### Online learning

    Formalize that and you get **online learning**: stream $(X, y)$ off disk (or off a live feed), read one object, update the weights, discard it, repeat. One full sweep through the file is an **epoch**; you typically need several dozen epochs before the loss settles.

    The rules of thumb worth internalizing:
    - A **decaying learning rate** helps — big steps early to cover ground, small steps late to actually settle into the minimum rather than bouncing around it. VW exposes this as `--power_t`.
    - Loss goes down per-epoch, not per-step. Individual steps can make things worse; that's expected and fine.
    - Online learning is also the natural fit for genuinely streaming data (clicks, logs) where "the dataset" never stops arriving.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical features: three encodings, each fixing the last

    Linear models want vectors of real numbers. Real data hands you `job=student`, `month=jun`, `marital=single`. Bridging that gap is where most of the practical damage happens.

    The UCI bank marketing dataset is mostly categorical, so it makes the point well.
    """)
    return


@app.cell
def _(DATA_URL, pd):
    df = pd.read_csv(DATA_URL + "bank_train.csv")
    labels = pd.read_csv(DATA_URL + "bank_train_target.csv", header=None)

    df.head()
    return df, labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Label encoding — and why it lies

    `LabelEncoder` maps each distinct value to an integer: `fit` collects the unique values and builds the mapping (exposed afterwards as `classes_`), `transform` applies it. `university.degree` → 6, `basic.4y` → 0, and so on.

    **The trap:** this invents an ordering that doesn't exist. Once `job` is an integer you can compute `df.loc[1].job - df.loc[2].job` and get `-1.0` — a number that means nothing. Worse, a linear model *believes* it: it fits one coefficient per column, so it's forced to assume `blue-collar` sits exactly halfway between `admin.` and `technician` on some imaginary jobs axis.

    The result is a model that hits 89% accuracy and is completely useless — it never predicts class 1 at all. Recall on the minority class is ~0.00. It just learned to always say "no", because label-encoded nonsense gave it nothing better.

    Label encoding is fine for **tree** models (they split on thresholds, so arbitrary ordering costs little) and actively harmful for **linear** ones.
    """)
    return


@app.cell
def _(LabelEncoder, df):
    label_encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(include=["object", "str"]).columns.union(
        ["education"]
    )
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    df.head()
    return (categorical_columns,)


@app.cell
def _(df):
    # the invented algebra: this subtraction is meaningless but perfectly legal
    df.loc[1].job - df.loc[2].job
    return


@app.cell
def _(
    LogisticRegression,
    categorical_columns,
    classification_report,
    df,
    labels,
    train_test_split,
):
    def logistic_regression_accuracy_on(dataframe, labels):
        train_features, test_features, train_labels, test_labels = train_test_split(
            dataframe, labels
        )
        logit = LogisticRegression()
        logit.fit(train_features, train_labels.values.ravel())
        return classification_report(test_labels, logit.predict(test_features))

    # label-encoded: high accuracy, ~zero recall on class 1. A useless model.
    print(logistic_regression_accuracy_on(df[categorical_columns], labels))
    return (logistic_regression_accuracy_on,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. One-hot encoding — correct, but it doesn't scale

    A feature with 10 distinct values becomes 10 new binary columns, exactly one of which is 1. No fake ordering: every value gets its own free coefficient.

    `OneHotEncoder` returns a **sparse matrix** by default, which is the right call — nearly every entry is a zero and storing them densely wastes RAM. (`sparse_output=False` below only because this example is small enough to eyeball.)

    Our 10 categorical columns explode into **53** columns. And the fix works: recall on class 1 jumps from ~0.00 to ~0.18, so real signal finally appears. (The split above has no `random_state`, so that recall wobbles a point or two run to run.) Same model, same data, just an honest encoding.

    **The two traps that kill it at scale:**
    - **Width.** Columns = total distinct values across all features. One `user_id` column with 30M values = 30M columns.
    - **Brittleness.** The encoder must see the whole dataset up front and hold the mapping in memory. A value that shows up for the first time in production has no column — the model has no idea what to do with it.
    """)
    return


@app.cell
def _(
    OneHotEncoder,
    categorical_columns,
    df,
    labels,
    logistic_regression_accuracy_on,
    pd,
):
    onehot_encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical_columns = pd.DataFrame(
        onehot_encoder.fit_transform(df[categorical_columns])
    )
    print(encoded_categorical_columns.shape)  # 53 columns from 10 original features

    print(logistic_regression_accuracy_on(encoded_categorical_columns, labels))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. The hashing trick — give up on the vocabulary

    Both problems above trace to the same root: keeping an explicit *mapping* from value → column. So delete the mapping. Instead, compute the column index directly:

    ```
    column_index = hash(feature_string) % hash_space
    ```

    `hash_space` is a fixed number you pick (VW's default is $2^{18}$). No `fit` pass, no dictionary in memory, no notion of "unseen value" — a brand new category on row 40 billion just hashes to *some* column, same as everything else. This is what makes true single-pass online learning possible.

    **Hash the `name=value` pair, not the bare value.** Otherwise `housing=no` and `loan=no` both hash on `"no"` and collapse into the same column — two different features silently sharing a coefficient.
    """)
    return


@app.cell
def _(pd):
    hash_space = 25

    # a single unmarried student called on a Monday -> one sparse vector, fixed width
    hashing_example = pd.DataFrame([{i: 0.0 for i in range(hash_space)}])
    for s in ("job=student", "marital=single", "day_of_week=mon"):
        print(s, "->", hash(s) % hash_space)
        hashing_example.loc[0, hash(s) % hash_space] = 1
    hashing_example
    return


@app.cell
def _():
    # hash the name=value pair so different features with the same value stay distinct
    assert hash("no") == hash("no")
    assert hash("housing=no") != hash("loan=no")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **"But collisions!"** Yes — two different features can land in the same column and share a coefficient. Three reasons not to panic:

    1. With a large enough `hash_space`, collisions are rare relative to the number of features.
    2. When they do happen, metrics barely move. Colliding features act like a crude **regularizer** — random weight-sharing that mostly costs you nothing.
    3. What's the alternative at 30M features? There often isn't one.

    It feels wrong and it works anyway. Booking.com published a good empirical study of how collision rates track feature-space vs. hash-space size ("Don't be tricked by the Hashing Trick") if you want the numbers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vowpal Wabbit

    VW is SGD + the hashing trick, in C++, driven from the shell. It's a linear model — no trees, no nets — but it trains on data far larger than RAM and it is very, very fast. For text it's an excellent default.

    ### The input format

    VW eats its own line-oriented format from a file or stdin:

    ```
    [Label] [Importance] [Tag]|Namespace Features |Namespace Features ...
    ```

    - **Label** — the target. `1`/`-1` for classification, any float for regression. Omit it for test data.
    - **Importance** — per-sample weight during training. This is your lever for imbalanced data.
    - **Tag** — a name for the sample, echoed back on prediction. Start it with `'` so VW can tell it apart from Importance.
    - **Namespace** — a named feature group (lets you cross whole groups of features later).
    - **Features** — `string` or `string:value` inside a namespace. Weight defaults to 1.0.

    A valid line looks like:

    ```
    1 1.0 |Subject WHAT car is this |Organization University of Maryland:0.5 College Park
    ```
    """)
    return


@app.cell
def _(subprocess):
    #! vw --help | head
    subprocess.call(["vw", "--help", "|", "head"])

    # feed one hand-written example through VW to sanity-check the format
    #! echo '1 1.0 |Subject WHAT car is this |Organization University of Maryland:0.5 College Park' | vw
    subprocess.call(
        [
            "echo",
            "1 1.0 |Subject WHAT car is this |Organization University of Maryland:0.5 College Park",
            "|",
            "vw",
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 20 newsgroups: binary classification

    The task: pick out `rec.autos` posts from the other 19 newsgroups. The conversion to VW format is deliberately crude — lowercase, keep tokens of 3+ characters, drop everything else. No stemming, no lemmatization, no stopword list. That's the point: VW gets away with it.

    Test rows are written **without** a label.
    """)
    return


@app.cell
def _(PATH_TO_WRITE_DATA, fetch_20newsgroups, re):
    newsgroups = fetch_20newsgroups(data_home=PATH_TO_WRITE_DATA)

    def to_vw_format(document, label=None):
        return (
            str(label or "")
            + " |text "
            + " ".join(re.findall(r"\w{3,}", document.lower()))
            + "\n"
        )

    all_documents = newsgroups["data"]
    all_targets = [
        1 if newsgroups["target_names"][target] == "rec.autos" else -1
        for target in newsgroups["target"]
    ]

    print(to_vw_format(all_documents[0], all_targets[0]))
    return all_documents, all_targets, newsgroups, to_vw_format


@app.cell
def _(
    PATH_TO_WRITE_DATA,
    all_documents,
    all_targets,
    os,
    to_vw_format,
    train_test_split,
):
    train_documents, test_documents, train_labels, test_labels = train_test_split(
        all_documents, all_targets, random_state=7
    )

    with open(
        os.path.join(PATH_TO_WRITE_DATA, "20news_train.vw"), "w"
    ) as vw_train_data:
        for text, target in zip(train_documents, train_labels):
            vw_train_data.write(to_vw_format(text, target))

    with open(os.path.join(PATH_TO_WRITE_DATA, "20news_test.vw"), "w") as vw_test_data:
        for text in test_documents:  # no label on test rows
            vw_test_data.write(to_vw_format(text))
    return (test_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now train. `-d` is the data, `--loss_function hinge` makes it a linear SVM, `-f` saves the model.

    The diagnostic table VW prints while training is worth reading (`--quiet` suppresses it). The key column is **average loss**, and the key detail is *how* it's computed: VW scores each example **before** learning from it — a progressive validation loss on data the model has genuinely never seen. So that number is an honest running estimate of generalization, not a training-set fit.

    Predicting is the same binary with `-i` (load model), `-t` (test mode: ignore labels, don't learn), `-p` (write predictions).
    """)
    return


@app.cell
def _(subprocess):
    #! vw -d $PATH_TO_WRITE_DATA/20news_train.vw --loss_function hinge -f $PATH_TO_WRITE_DATA/20news_model.vw
    subprocess.call(
        [
            "vw",
            "-d",
            "$PATH_TO_WRITE_DATA/20news_train.vw",
            "--loss_function",
            "hinge",
            "-f",
            "$PATH_TO_WRITE_DATA/20news_model.vw",
        ]
    )
    return


@app.cell
def _(subprocess):
    #! vw -i $PATH_TO_WRITE_DATA/20news_model.vw -t -d $PATH_TO_WRITE_DATA/20news_test.vw -p $PATH_TO_WRITE_DATA/20news_test_predictions.txt
    subprocess.call(
        [
            "vw",
            "-i",
            "$PATH_TO_WRITE_DATA/20news_model.vw",
            "-t",
            "-d",
            "$PATH_TO_WRITE_DATA/20news_test.vw",
            "-p",
            "$PATH_TO_WRITE_DATA/20news_test_predictions.txt",
        ]
    )
    return


@app.cell
def _(PATH_TO_WRITE_DATA, os, plt, roc_auc_score, roc_curve, test_labels):
    with open(
        os.path.join(PATH_TO_WRITE_DATA, "20news_test_predictions.txt")
    ) as pred_file:
        test_prediction = [float(label) for label in pred_file.readlines()]

    auc = roc_auc_score(test_labels, test_prediction)
    fpr, tpr, _ = roc_curve(test_labels, test_prediction)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("test AUC = %f" % auc)
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multiclass: `--oaa`

    All 20 classes at once via `--oaa K` ("one against all" — VW trains K binary models under the hood and takes the argmax).

    **VW is picky about labels here:** it wants `1..K`, not `0..K-1`. `LabelEncoder` gives you `0..K-1`, so add 1. Getting this wrong doesn't error — it silently trains a garbage model, which is a nasty way to lose an afternoon.

    A confusion matrix row tells you *what* a class gets mistaken for, which is more useful than a scalar accuracy. Here `alt.atheism` bleeds into `misc.forsale`, `talk.religion.misc`, and `sci.crypt` — the last one being a hint that this crude tokenizer is picking up on writing style, not topic.
    """)
    return


@app.cell
def _(
    LabelEncoder,
    PATH_TO_WRITE_DATA,
    all_documents,
    newsgroups,
    os,
    to_vw_format,
    train_test_split,
):
    topic_encoder = LabelEncoder()
    all_targets_mult = (
        topic_encoder.fit_transform(newsgroups["target"]) + 1
    )  # VW wants 1..K
    train_documents_1, test_documents_1, train_labels_mult, test_labels_mult = (
        train_test_split(all_documents, all_targets_mult, random_state=7)
    )
    with open(os.path.join(PATH_TO_WRITE_DATA, "20news_train_mult.vw"), "w") as f:
        for text_1, target_1 in zip(train_documents_1, train_labels_mult):
            f.write(to_vw_format(text_1, target_1))
    with open(os.path.join(PATH_TO_WRITE_DATA, "20news_test_mult.vw"), "w") as f:
        for text_1 in test_documents_1:
            f.write(to_vw_format(text_1))
    return (test_labels_mult,)


@app.cell
def _(subprocess):
    #! vw --oaa 20 $PATH_TO_WRITE_DATA/20news_train_mult.vw -f $PATH_TO_WRITE_DATA/20news_model_mult.vw --loss_function=hinge
    subprocess.call(
        [
            "vw",
            "--oaa",
            "20",
            "$PATH_TO_WRITE_DATA/20news_train_mult.vw",
            "-f",
            "$PATH_TO_WRITE_DATA/20news_model_mult.vw",
            "--loss_function=hinge",
        ]
    )
    #! vw -i $PATH_TO_WRITE_DATA/20news_model_mult.vw -t -d $PATH_TO_WRITE_DATA/20news_test_mult.vw -p $PATH_TO_WRITE_DATA/20news_test_predictions_mult.txt
    subprocess.call(
        [
            "vw",
            "-i",
            "$PATH_TO_WRITE_DATA/20news_model_mult.vw",
            "-t",
            "-d",
            "$PATH_TO_WRITE_DATA/20news_test_mult.vw",
            "-p",
            "$PATH_TO_WRITE_DATA/20news_test_predictions_mult.txt",
        ]
    )
    return


@app.cell
def _(
    PATH_TO_WRITE_DATA,
    accuracy_score,
    confusion_matrix,
    newsgroups,
    np,
    os,
    test_labels_mult,
):
    with open(
        os.path.join(PATH_TO_WRITE_DATA, "20news_test_predictions_mult.txt")
    ) as pred_file_1:
        test_prediction_mult = [float(label) for label in pred_file_1.readlines()]
    print(accuracy_score(test_labels_mult, test_prediction_mult))
    M = confusion_matrix(test_labels_mult, test_prediction_mult)
    for i in np.where(M[0, :] > 0)[0][1:]:
        # what does atheism get confused with?
        print(newsgroups["target_names"][i], M[0, i])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The knobs that matter

    Four VW parameters carry most of the tuning weight:

    - **`-l`** (learning rate, default 0.5) — how far each SGD step moves the weights.
    - **`--power_t`** (learning-rate decay, default 0.5) — how fast the rate shrinks as steps accumulate. Decay is what lets SGD actually settle rather than bounce.
    - **`--loss_function`** — `hinge` (SVM), `logistic`, `squared`, `quantile`. This *is* the training algorithm; changing it changes the model, not just its speed.
    - **`--l1`** / **`--l2`** (regularization) — note the *double* dash: `-l1` is not this, it parses as `-l 1`, a learning rate of 1. **Note the scale**: VW applies regularization **per example**, not per pass. So sane values are tiny, around $10^{-20}$, not the ~$10^{-2}$ you'd use in sklearn. Copying sklearn's values here will flatten your model to zero.

    Beyond hand-tuning, VW pairs well with Hyperopt for automated search.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### IMDB reviews: where the speed shows

    Binary sentiment on 25k train / 25k test movie reviews — same dataset as topic 4. Watch the wall-clock: VW trains this in about a second.

    This one is a ~80MB tarball, not in the mlcourse.ai data repo, so the cell below downloads and extracts it into the local `data/` dir next to this notebook (same dir the `.vw` files go to).

    70/30 train/hold-out split, then write three `.vw` files (train, valid, test).
    """)
    return


@app.cell
def _(PATH_TO_WRITE_DATA, load_files, np, os):
    import tarfile
    from io import BytesIO

    import requests

    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    def load_imdb_dataset(extract_path=PATH_TO_WRITE_DATA, overwrite=False):
        if (
            os.path.isfile(os.path.join(extract_path, "aclImdb", "README"))
            and not overwrite
        ):
            print("IMDB dataset is already in place.")
            return
        print("Downloading the dataset from: ", url)
        response = requests.get(url)
        tar = tarfile.open(mode="r:gz", fileobj=BytesIO(response.content))
        tar.extractall(extract_path)

    load_imdb_dataset()

    PATH_TO_IMDB = PATH_TO_WRITE_DATA + "aclImdb"
    reviews_train = load_files(
        os.path.join(PATH_TO_IMDB, "train"), categories=["pos", "neg"]
    )
    text_train, y_train = reviews_train.data, reviews_train.target

    reviews_test = load_files(
        os.path.join(PATH_TO_IMDB, "test"), categories=["pos", "neg"]
    )
    text_test, y_test = reviews_test.data, reviews_test.target

    print(len(text_train), np.bincount(y_train))  # 25000, perfectly balanced
    return (y_test,)


app._unparsable_cell(
    r"""
    train_share = int(0.7 * len(text_train))
    train, valid = text_train[:train_share], text_train[train_share:]
    train_labels, valid_labels = y_train[:train_share], y_train[train_share:]

    with open(os.path.join(PATH_TO_WRITE_DATA, "movie_reviews_train.vw"), "w") as f:
        for text, target in zip(train, train_labels):
            f.write(to_vw_format(str(text), 1 if target == 1 else -1))   # 0/1 -> -1/+1
    with open(os.path.join(PATH_TO_WRITE_DATA, "movie_reviews_valid.vw"), "w") as f:
        for text, target in zip(valid, valid_labels):
            f.write(to_vw_format(str(text), 1 if target == 1 else -1))
    with open(os.path.join(PATH_TO_WRITE_DATA, "movie_reviews_test.vw"), "w") as f:
        for text in text_test:
            f.write(to_vw_format(str(text)))

    !head -2 $PATH_TO_WRITE_DATA/movie_reviews_train.vw
    """,
    name="_",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Train, predict on the hold-out, score.

    **Reading VW's predictions:** with `hinge` loss VW writes the raw decision-function value, roughly in $[-1, 1]$ — *not* a probability, despite looking like one. For a hard class label, threshold at 0: `pred > 0` → class 1. Feed the raw scores (not the thresholded labels) to `roc_auc_score`, since AUC needs the ranking.

    Unigrams get ~88.5% accuracy / 0.942 AUC on the hold-out.
    """)
    return


app._unparsable_cell(
    r"""
    def score_vw_predictions(pred_path, true_labels):
        with open(os.path.join(PATH_TO_WRITE_DATA, pred_path)) as pred_file:
            pred = [float(label) for label in pred_file.readlines()]
        print("Accuracy:", round(accuracy_score(true_labels, [int(p > 0) for p in pred]), 3))
        print("AUC:", round(roc_auc_score(true_labels, pred), 3))
        return pred


    !vw -d $PATH_TO_WRITE_DATA/movie_reviews_train.vw --loss_function hinge \
    -f $PATH_TO_WRITE_DATA/movie_reviews_model.vw --quiet

    #! vw -i $PATH_TO_WRITE_DATA/movie_reviews_model.vw -t -d $PATH_TO_WRITE_DATA/movie_reviews_valid.vw -p $PATH_TO_WRITE_DATA/movie_valid_pred.txt --quiet
    subprocess.call(['vw', '-i', '$PATH_TO_WRITE_DATA/movie_reviews_model.vw', '-t', '-d', '$PATH_TO_WRITE_DATA/movie_reviews_valid.vw', '-p', '$PATH_TO_WRITE_DATA/movie_valid_pred.txt', '--quiet'])

    score_vw_predictions("movie_valid_pred.txt", valid_labels)   # ~0.885 / 0.942
    """,
    name="_",
)


@app.cell
def _(score_vw_predictions, subprocess, y_test):
    #! vw -i $PATH_TO_WRITE_DATA/movie_reviews_model.vw -t -d $PATH_TO_WRITE_DATA/movie_reviews_test.vw -p $PATH_TO_WRITE_DATA/movie_test_pred.txt --quiet
    subprocess.call(
        [
            "vw",
            "-i",
            "$PATH_TO_WRITE_DATA/movie_reviews_model.vw",
            "-t",
            "-d",
            "$PATH_TO_WRITE_DATA/movie_reviews_test.vw",
            "-p",
            "$PATH_TO_WRITE_DATA/movie_test_pred.txt",
            "--quiet",
        ]
    )

    score_vw_predictions(
        "movie_test_pred.txt", y_test
    )  # ~0.88 / 0.94 — hold-out was honest
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bigrams for free: `--ngram 2`

    One flag adds all adjacent word pairs as features. This matters enormously for sentiment, where `not good` and `good` are opposites that a bag-of-unigrams cannot tell apart.

    Hold-out goes 0.885 → **0.894** accuracy, 0.942 → **0.954** AUC; test lands at 0.888 / 0.952. And because bigrams are hashed into the *same* fixed weight vector, the feature count exploding costs no extra memory — that's the hashing trick paying rent.
    """)
    return


@app.cell
def _(score_vw_predictions, subprocess, valid_labels):
    #! vw -d $PATH_TO_WRITE_DATA/movie_reviews_train.vw --loss_function hinge --ngram 2 -f $PATH_TO_WRITE_DATA/movie_reviews_model2.vw --quiet
    subprocess.call(
        [
            "vw",
            "-d",
            "$PATH_TO_WRITE_DATA/movie_reviews_train.vw",
            "--loss_function",
            "hinge",
            "--ngram",
            "2",
            "-f",
            "$PATH_TO_WRITE_DATA/movie_reviews_model2.vw",
            "--quiet",
        ]
    )

    #! vw -i $PATH_TO_WRITE_DATA/movie_reviews_model2.vw -t -d $PATH_TO_WRITE_DATA/movie_reviews_valid.vw -p $PATH_TO_WRITE_DATA/movie_valid_pred2.txt --quiet
    subprocess.call(
        [
            "vw",
            "-i",
            "$PATH_TO_WRITE_DATA/movie_reviews_model2.vw",
            "-t",
            "-d",
            "$PATH_TO_WRITE_DATA/movie_reviews_valid.vw",
            "-p",
            "$PATH_TO_WRITE_DATA/movie_valid_pred2.txt",
            "--quiet",
        ]
    )

    score_vw_predictions("movie_valid_pred2.txt", valid_labels)  # ~0.894 / 0.954
    return


@app.cell
def _(score_vw_predictions, subprocess, y_test):
    #! vw -i $PATH_TO_WRITE_DATA/movie_reviews_model2.vw -t -d $PATH_TO_WRITE_DATA/movie_reviews_test.vw -p $PATH_TO_WRITE_DATA/movie_test_pred2.txt --quiet
    subprocess.call(
        [
            "vw",
            "-i",
            "$PATH_TO_WRITE_DATA/movie_reviews_model2.vw",
            "-t",
            "-d",
            "$PATH_TO_WRITE_DATA/movie_reviews_test.vw",
            "-p",
            "$PATH_TO_WRITE_DATA/movie_test_pred2.txt",
            "--quiet",
        ]
    )

    score_vw_predictions("movie_test_pred2.txt", y_test)  # ~0.888 / 0.952
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **SGD is what makes big-data training possible.** Dropping the sum over $\ell$ examples turns each update into a one-row operation — noisy gradients, but the data can stay on disk. Many cheap wobbly steps beat few exact expensive ones.
    - **Online learning = stream, update, discard.** One pass over the file is an epoch; expect dozens. Decay the learning rate so it settles instead of bouncing.
    - **Label encoding is a lie for linear models.** It invents an ordering that doesn't exist; the model dutifully believes it and collapses to predicting the majority class. Fine for trees, poison for linear.
    - **One-hot is correct but doesn't scale** — width grows with total distinct values, and it needs the whole vocabulary in memory before it can encode anything.
    - **The hashing trick deletes the vocabulary.** `hash(name=value) % hash_space` gives the column index directly: no fit pass, no dictionary, no unseen-value problem. Hash the `name=value` pair, never the bare value.
    - **Collisions are fine.** Rare with a big hash space, and when they happen they act as regularization. Don't over-think it.
    - **VW = SGD + hashing, from the shell.** `-d` data, `-f` save model, `-i` load model, `-t` test mode, `-p` predictions, `--quiet` shut up, `--oaa K` multiclass, `--ngram 2` bigrams.
    - **VW's average loss is progressive validation** — it scores each example before learning from it, so the running number is an honest generalization estimate.
    - **VW's `--l1` scale is not sklearn's.** Regularization is applied per example, so values live around $10^{-20}$. And it's `--l1`, not `-l1` — the latter parses as `-l 1`.
    - **`--oaa` wants labels 1..K, not 0..K-1.** It fails silently otherwise.
    - **`--ngram 2` is the cheapest accuracy win on text.** Bigrams hash into the same weight vector, so the memory cost is zero.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    This topic assembles the two ideas that let a linear model train on data larger than RAM. SGD replaces batch gradient descent's sum-over-everything update with a single-example update, making training a streaming operation and enabling online learning; the hashing trick replaces the value→column dictionary with `hash(name=value) % hash_space`, eliminating the fit pass, the memory-resident vocabulary, and the unseen-category problem, at the price of occasional collisions that behave like regularization anyway. Vowpal Wabbit is both ideas shipped as a fast shell tool: its line format carries label, importance weight, tag, and namespaced features; `--oaa` handles multiclass (labels 1..K); its average-loss diagnostic is progressive validation, so it's honest by construction. The worked examples run the arc from why label encoding wrecks a logistic regression (89% accuracy, ~0.00 recall on the minority class), through one-hot fixing that but exploding to 53 columns from 10 features, to VW hitting 0.88 accuracy / 0.94 AUC on 25k IMDB reviews in about a second — and 0.888 / 0.952 by adding `--ngram 2`, a one-flag improvement that costs no memory because bigrams hash into the same fixed weight vector.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **gradient descent** = minimize a function by stepping against the gradient
        - **antigradient** = the negative gradient; direction of steepest decrease
        - **learning rate** = step size per update ($\eta$)
        - **learning-rate decay** = shrinking $\eta$ as steps accumulate so the optimizer settles (VW: `--power_t`)
    - **batch gradient descent** = each update sums over all $\ell$ training rows; exact but needs the data in RAM
    - **SGD** = each update uses one example; noisy gradient, but the row can be read off disk and discarded
    - **online learning** = stream examples off disk/feed, update, discard; never loads the dataset
    - **epoch** = one full pass over the training file
    - **progressive validation loss** = VW's average loss; each example is scored before it's learned from, so it estimates generalization
    - **label encoding** = map each category to an integer; invents a false ordering that breaks linear models
    - **one-hot encoding** = one binary column per distinct value; correct but width scales with vocabulary size
    - **sparse matrix** = storage format keeping only non-zero entries; OneHotEncoder's default output
    - **hashing trick** = column index = `hash(name=value) % hash_space`; no vocabulary, no fit pass, no unseen-value problem
        - **hash space** = the fixed number of columns hashed into (VW default $2^{18}$)
        - **hash collision** = two features landing in the same column; rare at scale and acts as regularization
    - **Vowpal Wabbit (VW)** = shell-driven linear learner built on SGD + hashing
        - **label** = VW's target field; `1`/`-1` for classification, omitted on test rows
        - **importance** = VW's per-sample training weight; the lever for imbalanced data
        - **tag** = VW's per-sample name echoed back at prediction; prefix with `'`
        - **namespace** = a named feature group inside a VW line
        - **`--oaa K`** = one-against-all multiclass; requires labels 1..K
        - **`--ngram 2`** = add adjacent word pairs as features, hashed into the same weight vector
        - **`-t`** = test mode: ignore labels, don't update weights
    - **hinge loss** = the linear-SVM loss; VW outputs a raw decision value, not a probability — threshold at 0
    """)
    return


if __name__ == "__main__":
    app.run()
