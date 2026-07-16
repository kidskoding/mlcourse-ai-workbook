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
    # Topic 4 — Part 2: Linear Classification

    Part 1 fit lines to *numbers*. This part fits a line to *classes*: draw a hyperplane through
    feature space, call one side "+", the other "−". The real content here isn't the hyperplane —
    it's answering three questions honestly:

    1. How do you squeeze a linear score $w^Tx \in \mathbb{R}$ into a **probability** in $[0,1]$?
    2. Where does the **logistic loss** actually come from? (Not handed down — derived from maximum likelihood.)
    3. What does **regularization** look like once the loss is a log-likelihood?

    It's a derivation-heavy part. The math *is* the lesson, so the math stays.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The linear classifier

    Take binary labels encoded as $y \in \{-1, +1\}$ — not $\{0,1\}$. This looks cosmetic and isn't;
    it's what makes the whole derivation below collapse into one line.

    The simplest linear classifier just takes the sign of a linear regression:

    $$a(x) = \mathrm{sign}(w^T x)$$

    where $x$ is the feature vector (with a 1 appended so the bias $w_0$ rides along inside $w$), and
    $w^T x = 0$ defines the **separating hyperplane**. If some hyperplane classifies the training set
    with zero errors, the data is **linearly separable** — which real data almost never is, and that's
    fine: we don't need zero errors, we need a good boundary.

    The problem with `sign`: it returns a hard verdict and nothing else. No confidence, no ranking, no
    "how sure are you". That's the gap logistic regression fills.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why you want a probability, not a verdict

    Credit scoring is the canonical case and it makes the argument better than any abstract one.
    A bank doesn't want "will default / won't default". It wants $p_+$, so it can:

    - **Rank** applicants worst-to-best and set a threshold $p^*$ wherever its risk appetite sits
      (approve below 0.15, decline above) — and *move* that number when the economy changes,
      without retraining.
    - Multiply $p_+ \times \text{loan amount}$ to get **expected loss** in currency — a number the
      business side can actually act on.

    A `sign` classifier gives you none of this. It bakes the threshold into the model permanently.
    So: keep the linear score, but map it to a probability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Odds, log-odds, and why the sigmoid is forced on us

    The sigmoid isn't an arbitrary squashing function someone liked the shape of. Work backwards from
    ranges and it's the only thing that fits.

    We need $f: \mathbb{R} \to [0,1]$, because $w^Tx$ lives in $\mathbb{R}$ but a probability doesn't.
    Ladder up through three quantities that carry **identical information** but live in different ranges:

    | quantity | definition | range |
    |---|---|---|
    | probability | $P(X)$ | $[0, 1]$ |
    | **odds** | $OR(X) = \dfrac{P(X)}{1 - P(X)}$ | $[0, \infty)$ |
    | **log-odds** | $\log OR(X)$ | $\mathbb{R}$ |

    Log-odds range over all of $\mathbb{R}$ — exactly the range of $w^Tx$. So don't force the linear
    model to predict a probability. **Let it predict the log-odds**, where it's already at home.

    Prediction is then three steps:

    1. Compute the linear score $w^Tx$.
    2. Declare it the log-odds: $\log(OR_+) = w^Tx$.
    3. Invert back to a probability:

    $$p_+ = \frac{OR_+}{1 + OR_+} = \frac{\exp(w^Tx)}{1 + \exp(w^Tx)} = \frac{1}{1 + \exp(-w^Tx)} = \sigma(w^Tx)$$

    The sigmoid falls out of step 3 algebraically. It was never a design choice — it's the inverse of
    the log-odds link. Hence:

    $$p_+(x_i) = P(y_i = 1 \mid x_i, w) = \sigma(w^T x_i)$$
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    def sigma(z):
        return 1 / (1 + np.exp(-z))

    z = np.linspace(-8, 8, 200)
    plt.figure(figsize=(6, 3))
    plt.plot(z, sigma(z))
    plt.axhline(0.5, ls="--", c="gray", lw=1)
    plt.axvline(0, ls="--", c="gray", lw=1)
    plt.xlabel("$w^Tx$ (log-odds)")
    plt.ylabel("$p_+$")
    plt.title("sigmoid: log-odds -> probability")
    plt.show()

    # the inverse-link symmetry the derivation leans on
    assert np.allclose(sigma(-z), 1 - sigma(z))
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The margin, and the $\pm 1$ trick

    Now the payoff for labelling classes $\pm 1$. The two class probabilities are:

    $$p_+(x_i) = \sigma(w^T x_i), \qquad p_-(x_i) = 1 - \sigma(w^T x_i) = \sigma(-w^T x_i)$$

    The second equality is the symmetry $1 - \sigma(z) = \sigma(-z)$ (asserted in the cell above).
    Because $y_i$ is literally $\pm 1$, the sign it carries folds straight into the argument and both
    cases become **one expression**:

    $$P(y = y_i \mid x_i, w) = \sigma(y_i w^T x_i)$$

    That inner quantity gets a name — the **margin** of object $x_i$:

    $$M(x_i) = y_i w^T x_i$$

    (Not the SVM "margin" = gap between classes. Same word, different thing. Standard trap.)

    Sign and magnitude each mean something specific:

    - **sign** — positive ⇒ the predicted class matches $y_i$ (correct); negative ⇒ misclassified.
    - **magnitude** — from the point-to-plane distance $\rho(x_A, w^Tx=0) = \frac{w^T x_A}{\|w\|}$, a
      large $|w^Tx_i|$ means $x_i$ sits **far from the boundary**. So $|M|$ is *confidence*.

    Three cases worth internalizing:

    | margin | reading |
    |---|---|
    | large positive | correct and far from the boundary — confidently right |
    | large negative | wrong and far from the boundary — confidently wrong ⇒ **suspect a mislabel or an anomaly** |
    | small (either sign) | hugging the boundary; the model is basically guessing |

    The margin is only defined on the training set — it needs the true $y_i$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deriving logistic loss from maximum likelihood

    The logistic loss is not a definition to memorize. It's what you get when you write down "make the
    observed data as likely as possible" and simplify.

    Assume the objects are **i.i.d.** — a genuinely strong assumption, and the one that licenses the
    product below. The likelihood of the whole dataset:

    $$P(y \mid X, w) = \prod_{i=1}^{\ell} P(y = y_i \mid x_i, w)$$

    Take logs, because sums optimize and products don't (products underflow, and their derivatives
    turn into product rules):

    $$\log P(y \mid X, w) = \sum_{i=1}^{\ell} \log \sigma(y_i w^T x_i)
    = \sum_{i=1}^{\ell} \log \frac{1}{1 + \exp(-y_i w^T x_i)}
    = -\sum_{i=1}^{\ell} \log \left(1 + \exp(-y_i w^T x_i)\right)$$

    Maximizing that is minimizing its negation — the **logistic loss**:

    $$\mathcal{L}_{\log}(X, y, w) = \sum_{i=1}^{\ell} \log \left(1 + \exp(-y_i w^T x_i)\right)$$

    Same move as part 1: there, MLE under Gaussian noise handed us squared error. Here, MLE under a
    sigmoid/Bernoulli model hands us log-loss. **The loss function is a consequence of the noise model
    you assumed, never a free choice.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why not just minimize the error count?

    What we actually care about is **zero-one loss** — count the mistakes:

    $$\mathcal{L}_{1/0}(M) = [M < 0]$$

    It's the honest objective and it's useless to optimize. It's a step function: flat everywhere
    (gradient zero, nothing to descend) with an undefined jump at zero. Gradient methods have no
    signal to follow.

    So substitute a **smooth upper bound** and minimize that instead. As a function of the margin,
    logistic loss is $L(M) = \log(1 + e^{-M})$, and it sits above the step everywhere:

    $$\mathcal{L}_{1/0}(X, y, w) = \sum_{i=1}^{\ell} [M(x_i) < 0] \;\le\; \sum_{i=1}^{\ell} \log_2\left(1 + \exp(-y_i w^T x_i)\right)$$

    (The bound is stated with a base-2 log; the base is only a constant factor and doesn't move the
    argmin.) Push the ceiling down and the error count underneath it gets dragged down too. That's the
    whole logic — **you don't optimize what you want, you optimize a differentiable thing that caps it.**

    Worth noticing in the plot: log-loss keeps penalizing points with $M > 0$ that are *already
    correct* but close to the boundary. It isn't satisfied by "correct" — it wants correct *and*
    confident.
    """)
    return


@app.cell
def _(np, plt):
    M = np.linspace(-3, 3, 300)

    plt.figure(figsize=(6, 3.5))
    plt.plot(M, (M < 0).astype(int), label="zero-one loss  $[M<0]$", lw=2)
    plt.plot(
        M, np.log2(1 + np.exp(-M)), label="logistic loss  $\\log_2(1+e^{-M})$", lw=2
    )
    plt.xlabel("margin  $M = y \\cdot w^Tx$")
    plt.ylabel("loss")
    plt.legend()
    plt.title("logistic loss upper-bounds the error count")
    plt.show()

    # the bound: base-2 logistic loss is never below zero-one loss
    assert np.all(np.log2(1 + np.exp(-M)) >= (M < 0).astype(int))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L2-regularization and the C convention

    Identical idea to ridge in part 1 — penalize big weights:

    $$J(X, y, w) = \mathcal{L}_{\log}(X, y, w) + \lambda \|w\|_2$$

    One wrinkle that trips everyone up in scikit-learn: logistic regression conventionally uses an
    **inverse** regularization coefficient $C = \frac{1}{\lambda}$, and it multiplies the *loss* rather
    than the penalty:

    $$\hat{w} = \arg\min_{w} \left( C \sum_{i=1}^{\ell} \log\left(1 + \exp(-y_i w^T x_i)\right) + \|w\|_2 \right)$$

    So the direction runs **backwards from $\lambda$**, and this is the single most common config bug:

    - **small $C$** ⇒ strong regularization ⇒ weights shrink toward 0 ⇒ underfit
    - **large $C$** ⇒ weak regularization ⇒ the loss term dominates ⇒ overfit

    Intuition for why shrinking weights helps: $p_+ = \sigma(w^Tx)$, and $\sigma$ **saturates**. Big
    weights push $|w^Tx|$ into the flat tails, so the model emits probabilities pinned at 0.99 / 0.01 —
    confident everywhere, including where it's wrong. Small weights keep scores near the sigmoid's
    linear middle, so predictions stay hedged. **Regularization is a cap on how cocky the model is
    allowed to be**, and $C$ is where you set it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Encode labels as $\pm 1$, not $\{0,1\}$.** It's what lets $p_+$ and $p_-$ merge into the single expression $\sigma(y_i w^T x_i)$, and it's what makes the margin exist.
    - **The sigmoid is derived, not chosen.** Have the linear model predict log-odds (range $\mathbb{R}$, matching $w^Tx$); inverting the link gives $\sigma$ for free.
    - **Probability beats verdict.** It lets you rank, move the threshold $p^*$ without retraining, and convert to expected loss in currency.
    - **Margin $M = y_i w^T x_i$**: sign = correct or not, magnitude = distance from the boundary = confidence. A large-negative margin is a mislabel flag, not just an error.
    - **Don't confuse this margin with the SVM margin.** Same word, unrelated meaning.
    - **Log-loss comes from MLE**, exactly as squared error did in part 1. Your loss is downstream of your assumed noise model.
    - **Zero-one loss is unoptimizable** (flat gradient, jump at 0). Minimize a smooth upper bound instead and drag the error count down with it.
    - **$C = 1/\lambda$ runs backwards.** Small $C$ = strong regularization = underfit; large $C$ = overfit. Check the direction every single time.
    - **Regularization limits overconfidence**: small weights keep $w^Tx$ off the sigmoid's saturated tails.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Linear classification splits feature space with the hyperplane $w^Tx = 0$, but plain $\mathrm{sign}(w^Tx)$ throws away everything except the verdict. Logistic regression recovers the lost information by having the linear model predict **log-odds** — the one transform of probability whose range is all of $\mathbb{R}$, matching $w^Tx$ — and inverting that link, which produces the sigmoid $p_+ = \sigma(w^Tx)$ algebraically rather than by fiat. Encoding classes as $\pm 1$ collapses both class probabilities into $P(y = y_i \mid x_i, w) = \sigma(M(x_i))$ with the **margin** $M(x_i) = y_i w^T x_i$, whose sign says correct/incorrect and whose magnitude measures distance from the boundary, i.e. confidence — so a large negative margin flags a probable mislabel rather than an ordinary mistake. Applying maximum likelihood to the i.i.d. dataset, taking logs to turn the product into a sum, and negating yields the **logistic loss** $\sum_i \log(1 + e^{-M_i})$, mirroring part 1 where MLE under Gaussian noise produced squared error, and confirming that the loss is a consequence of the assumed noise model rather than a free choice. We minimize it because the objective we actually want — the zero-one error count — is a step function with no usable gradient, while logistic loss is a smooth upper bound on it, so pushing the bound down pushes the errors down. Finally, L2-regularization adds $\|w\|_2$ under scikit-learn's inverse convention $C = 1/\lambda$ multiplying the loss (small $C$ shrinks weights and underfits, large $C$ overfits), which amounts to keeping $w^Tx$ out of the sigmoid's saturated tails and stopping the model from being confidently wrong.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **linear classifier** = classify by which side of the hyperplane $w^Tx = 0$ a point falls on
        - **separating hyperplane** = the boundary surface $w^Tx = 0$
        - **signum function** = $\mathrm{sign}(\cdot)$, returns the sign of its argument
    - **linearly separable** = some hyperplane classifies the training set with zero errors
    - **logistic regression** = linear classifier that outputs $p_+ = \sigma(w^Tx)$ instead of a hard label
    - **odds** = $OR(X) = P(X) / (1 - P(X))$, range $[0, \infty)$
        - **log-odds** = $\log OR(X)$, range $\mathbb{R}$ — what the linear model actually predicts
    - **sigmoid** = $\sigma(z) = 1 / (1 + e^{-z})$, the inverse of the log-odds link
    - **saturation** = the sigmoid's flat tails, where large $|w^Tx|$ yields probabilities pinned at 0 or 1
    - **margin** = $M(x_i) = y_i w^T x_i$; sign = correctness, magnitude = confidence
    - **i.i.d.** = objects drawn independently from one distribution; the assumption licensing the likelihood product
    - **likelihood** = $P(y \mid X, w)$, the probability of the observed labels under weights $w$
        - **MLE** = choose $w$ maximizing the likelihood
        - **log-likelihood** = its logarithm; turns the product into an optimizable sum
    - **logistic loss** = $\sum_i \log(1 + \exp(-y_i w^T x_i))$, the negative log-likelihood
    - **zero-one loss** = $[M < 0]$, the raw error count; the correct objective, but no usable gradient
    - **surrogate loss** = a smooth function sitting above the target loss, minimized in its place
    - **L2-regularization** = add $\|w\|_2$ to the objective to shrink weights
        - **C** = inverse regularization strength $1/\lambda$; small $C$ = strong regularization = underfit
    """)
    return


if __name__ == "__main__":
    app.run()
