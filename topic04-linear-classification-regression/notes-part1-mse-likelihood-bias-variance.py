# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
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
    # Topic 4 — Part 1: OLS, Maximum Likelihood, and the Bias–Variance Decomposition

    Part 1 of Topic 4 is the theory floor under every linear model that follows. Three questions, answered in order:

    1. **What is OLS and what does it solve?** A closed-form fit, optimal under a specific set of assumptions.
    2. **Why squared error and not, say, absolute error?** Because minimizing MSE *is* maximum likelihood when the noise is Gaussian. The loss isn't arbitrary — it encodes an assumption about the noise.
    3. **Where does prediction error actually come from?** Bias² + variance + irreducible noise. This decomposition is the mental model for every over/underfitting argument in the rest of the course.

    This part is all math — no code. The code shows up in Parts 3–5, where regularization and validation get exercised on real data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The linear model

    A linear model predicts the target as a weighted sum of features:

    $$y = w_0 + \sum_{i=1}^{m} w_i x_i$$

    The trick that makes the notation clean: bolt on a fake feature $x_0 = 1$ (the **bias** / **intercept** term). Now the standalone $w_0$ folds into the sum and everything is one dot product:

    $$y = \sum_{i=0}^{m} w_i x_i = \mathbf{w}^T \mathbf{x}$$

    Stack $n$ observations as rows and you get the matrix form:

    $$\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$$

    - $\mathbf{w} \in \mathbb{R}^{m+1}$ — the **weights** (parameters) to learn.
    - $\mathbf{X} \in \mathbb{R}^{n \times (m+1)}$ — the design matrix, with that column of ones on the left. Assumed **full column rank**: $\text{rank}(\mathbf{X}) = m+1$.
    - $\boldsymbol{\epsilon} \in \mathbb{R}^{n}$ — random **noise** / error.
    - $\mathbf{y} \in \mathbb{R}^{n}$ — the target.

    Note where the noise lives: in the *data-generating process*, not in the model. We assume the world hands us a deterministic signal plus irreducible jitter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Gauss–Markov assumptions

    The whole "OLS is optimal" claim rests on three restrictions on the noise. Learn them, because every later complaint about linear regression ("heteroscedastic residuals", "autocorrelated errors") is one of these breaking:

    - **Zero mean:** $\forall i: \mathbb{E}[\epsilon_i] = 0$ — the noise doesn't systematically push predictions one way.
    - **Homoscedasticity:** $\forall i: \text{Var}(\epsilon_i) = \sigma^2 < \infty$ — same finite noise variance everywhere. (The opposite, noise that grows with $x$, is *heteroscedasticity*.)
    - **Uncorrelated errors:** $\forall i \neq j: \text{Cov}(\epsilon_i, \epsilon_j) = 0$ — no leakage between observations.

    Two definitions the theorem is stated in terms of:

    - An estimator $\hat{w}_i$ is **linear** if it's a linear combination of the targets, $\hat{w}_i = \omega_{1i} y_1 + \cdots + \omega_{ni} y_n$, where the $\omega_{ki}$ depend only on $\mathbf{X}$ — never on the unknown true $\mathbf{w}$. The OLS solution turns out to have exactly this shape, which is *why the model is called linear regression*.
    - An estimator is **unbiased** if it's right on average: $\mathbb{E}[\hat{w}_i] = w_i$.

    Notice that **normality of the errors is NOT on this list.** It shows up later as an *extra* assumption for the likelihood argument. Conflating the two is the classic trap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ordinary Least Squares

    OLS picks the weights that minimize mean squared error between actual and predicted:

    $$\mathcal{L}(\mathbf{X}, \mathbf{y}, \mathbf{w}) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 = \frac{1}{2n} \lVert \mathbf{y} - \mathbf{X}\mathbf{w} \rVert_2^2 = \frac{1}{2n} (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w})$$

    (The $\tfrac{1}{2}$ is pure convenience — it cancels the 2 from the derivative.)

    Expand, differentiate, set to zero:

    $$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{\partial}{\partial \mathbf{w}} \frac{1}{2n}\left(\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{w} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w}\right) = \frac{1}{2n}\left(-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w}\right)$$

    $$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \iff \mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y} \iff \boxed{\;\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\;}$$

    That last line is the **normal equation** — a closed-form solution, no gradient descent needed. It's also the source of every practical problem with OLS, because it requires inverting $\mathbf{X}^T\mathbf{X}$ (see regularization below).

    Matrix-derivative cheat sheet used above (if you don't trust it, redo it in scalar sums — tedious but convincing):

    $$\frac{\partial}{\partial \mathbf{X}}\mathbf{X}^T\mathbf{A} = \mathbf{A} \qquad \frac{\partial}{\partial \mathbf{X}}\mathbf{X}^T\mathbf{A}\mathbf{X} = (\mathbf{A} + \mathbf{A}^T)\mathbf{X} \qquad \frac{\partial}{\partial \mathbf{A}}\mathbf{X}^T\mathbf{A}\mathbf{y} = \mathbf{X}^T\mathbf{y}$$

    **Gauss–Markov theorem:** under the three assumptions above, the OLS estimator has the *lowest variance among all linear unbiased estimators*. It is BLUE — Best Linear Unbiased Estimator. Read the qualifiers carefully: only *linear* estimators, only *unbiased* ones. Give up unbiasedness and you can do better — which is precisely the deal regularization offers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why squared error? Maximum likelihood

    Why minimize the *square* of the residual rather than its absolute value? You could — but you'd fall outside the Gauss–Markov conditions and lose the optimality guarantee. The deeper answer is that **MSE is not a design choice; it's a consequence of assuming Gaussian noise.**

    **Maximum likelihood estimation (MLE)** is the machinery: pick the parameter value that makes the data you actually observed most probable.

    ### Warm-up: the Bernoulli case

    Survey 400 people on whether they know the formula for methanol; 117 do. The intuitive estimate of "probability the next person knows it" is $117/400 \approx 29\%$. Let's show that intuition *is* the MLE.

    A **Bernoulli** random variable takes value 1 with probability $\theta$ and 0 with probability $1-\theta$:

    $$p(\theta, x) = \theta^x (1-\theta)^{(1-x)}, \quad x \in \{0, 1\}$$

    The **likelihood** of the whole (independent) sample is the product of per-observation probabilities:

    $$p(\mathbf{x}; \theta) = \prod_{i=1}^{400} \theta^{x_i}(1-\theta)^{(1-x_i)} = \theta^{117}(1-\theta)^{283}$$

    Maximize the **log-likelihood** instead — log is monotonic, so the argmax is unchanged, but products become sums and the algebra collapses:

    $$\log p(\mathbf{x}; \theta) = 117\log\theta + 283\log(1-\theta)$$

    $$\frac{\partial \log p}{\partial \theta} = \frac{117}{\theta} - \frac{283}{1-\theta} = 0 \implies \theta = \frac{117}{400}$$

    The intuitive fraction is exactly the maximum likelihood estimate. **Take the log** is the reflex to internalize here — it shows up in every likelihood derivation for the rest of the course.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same argument for linear regression

    Same model, $\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$, but now add one **extra** assumption that Gauss–Markov never required — the errors are normally distributed:

    $$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

    Then each target is itself normal, centered on the model's prediction:

    $$p(y_i \mid \mathbf{X}; \mathbf{w}) = \mathcal{N}\Big(\sum_{j=1}^{m} w_j X_{ij},\; \sigma^2\Big)$$

    Observations are independent (uncorrelated errors), so the joint likelihood is a product of densities, and the log-likelihood is a sum:

    $$\log p(\mathbf{y} \mid \mathbf{X}; \mathbf{w}) = -\frac{n}{2}\log 2\pi\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

    Now maximize over $\mathbf{w}$ — and throw away every term that doesn't contain $\mathbf{w}$:

    $$\mathbf{w}_{ML} = \arg\max_{\mathbf{w}} \log p(\mathbf{y} \mid \mathbf{X}; \mathbf{w}) = \arg\max_{\mathbf{w}} -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 = \arg\min_{\mathbf{w}} \mathcal{L}(\mathbf{X}, \mathbf{y}, \mathbf{w})$$

    The $-\frac{n}{2}\log 2\pi\sigma^2$ is constant in $\mathbf{w}$, and the $\frac{1}{2\sigma^2}$ is a positive scale factor — neither moves the argmax. Flipping the sign turns max into min and out drops **exactly the OLS objective**.

    **The punchline:** maximizing likelihood under Gaussian noise ≡ minimizing squared error. If you believed the noise were Laplacian instead, the same derivation would hand you *absolute* error. The loss function is a statement about the noise you think you have.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The bias–variance decomposition

    This section is framed around linear regression but the result holds for **any** model of the form $y = f(\mathbf{x}) + \epsilon$. It's the single most reusable idea in the course.

    The setup, restated:

    - The target is a deterministic function plus noise: $y = f(\mathbf{x}) + \epsilon$, with $\epsilon \sim \mathcal{N}(0, \sigma^2)$, so $y \sim \mathcal{N}(f(\mathbf{x}), \sigma^2)$.
    - We approximate the unknown $f$ with $\hat{f}(\mathbf{x})$, fit from a training set. **Crucially, $\hat{f}$ is itself a random variable** — draw a different training set, get a different $\hat{f}$. That's what makes "variance" meaningful.

    Expand the expected squared error at a point $\mathbf{x}$ (dropping the argument for readability):

    $$\text{Err}(\mathbf{x}) = \mathbb{E}\big[(y - \hat{f})^2\big] = \mathbb{E}[y^2] + \mathbb{E}[\hat{f}^2] - 2\,\mathbb{E}[y\hat{f}]$$

    Handle the pieces with $\text{Var}(z) = \mathbb{E}[z^2] - \mathbb{E}[z]^2$. Since $\mathbb{E}[y] = \mathbb{E}[f + \epsilon] = f$ and $\text{Var}(y) = \mathbb{E}[(f + \epsilon - f)^2] = \mathbb{E}[\epsilon^2] = \sigma^2$:

    $$\mathbb{E}[y^2] = \text{Var}(y) + \mathbb{E}[y]^2 = \sigma^2 + f^2 \qquad\qquad \mathbb{E}[\hat{f}^2] = \text{Var}(\hat{f}) + \mathbb{E}[\hat{f}]^2$$

    The cross term uses independence of noise and estimator, plus $\mathbb{E}[\epsilon] = 0$:

    $$\mathbb{E}[y\hat{f}] = \mathbb{E}[(f + \epsilon)\hat{f}] = f\,\mathbb{E}[\hat{f}] + \mathbb{E}[\epsilon]\mathbb{E}[\hat{f}] = f\,\mathbb{E}[\hat{f}]$$

    Put it together and the squares collapse into a perfect square:

    $$\text{Err}(\mathbf{x}) = \sigma^2 + f^2 + \text{Var}(\hat{f}) + \mathbb{E}[\hat{f}]^2 - 2f\,\mathbb{E}[\hat{f}] = \underbrace{(f - \mathbb{E}[\hat{f}])^2}_{\text{Bias}^2} + \underbrace{\text{Var}(\hat{f})}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible}}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the decomposition

    $$\text{Err}(\mathbf{x}) = \text{Bias}(\hat{f})^2 + \text{Var}(\hat{f}) + \sigma^2$$

    - **Bias²** — how far the *average* model (over all possible training sets) sits from the truth. Systematic error. A model too weak to represent $f$ is biased: it learns something reliably offset from the right answer.
    - **Variance** — how much $\hat{f}$ jumps around when you re-train on a different sample. Instability. A model with enough free parameters to memorize the training set gets thrown by small data changes — that's **overfitting**.
    - **Irreducible error $\sigma^2$** — the noise floor. No model, no amount of data, no cleverness removes it. If someone reports test error below the noise floor, they have a leak.

    The dartboard picture people draw: bias = your cluster is off-center; variance = your cluster is spread out. You want tight and centered, but you usually have to trade.

    **The rule of thumb:** as model capacity grows (more free parameters), **variance goes up and bias goes down**. As capacity shrinks, the reverse. You can only influence the first two terms, and mostly by trading one against the other. Everything later in the course — regularization, bagging, boosting, learning curves — is a strategy for managing this trade.

    And note where Gauss–Markov lands on this map: it guarantees OLS has the minimum variance *among unbiased linear estimators*. It says nothing about estimators willing to accept some bias. That's the loophole.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularization: buying stability with bias

    Sometimes you **want** more bias, deliberately, to cut $\text{Var}(\hat{f})$. Here's the concrete motivation.

    **The failure mode.** Gauss–Markov assumed $\mathbf{X}$ has full column rank. If it doesn't, $\mathbf{X}^T\mathbf{X}$ is **singular**, the inverse doesn't exist, and the OLS solution simply isn't defined — an **ill-posed problem**.

    **The realistic version is worse than the clean failure.** Real data has **multicollinearity**: two or more features nearly linearly dependent. Classic example — predicting house prices from both "area with balcony" and "area without balcony". Formally $\mathbf{X}^T\mathbf{X}$ is still invertible, so nothing errors out. But some of its eigenvalues sit near zero, and since the inverse has eigenvalues $1/\lambda_i$, the inverse has *enormous* ones. Result: wildly unstable weights. Add a handful of new observations and the fitted coefficients change completely. The math didn't complain; the answer is just garbage.

    **The fix.** Make $\mathbf{X}^T\mathbf{X}$ well-conditioned again — make it *regular*, which is where the word *regularization* comes from. **Tikhonov regularization** adds a penalty term to the loss:

    $$\mathcal{L}(\mathbf{X}, \mathbf{y}, \mathbf{w}) = \frac{1}{2}\lVert \mathbf{y} - \mathbf{X}\mathbf{w}\rVert_2^2 + \lVert \Gamma \mathbf{w} \rVert^2$$

    (The $\tfrac{1}{n}$ is dropped here — it only rescales $\lambda$, so carrying it just clutters the algebra.)

    Take the Tikhonov matrix to be a scaled identity, $\Gamma = \sqrt{\tfrac{\lambda}{2}}\,E$. Then $\lVert\Gamma\mathbf{w}\rVert^2 = \tfrac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ — the penalty is now a plain constraint on the **L2 norm** of $\mathbf{w}$. Watch the square root: it has to survive the squaring inside the penalty, so $\Gamma = \tfrac{\lambda}{2}E$ would leave you with $\tfrac{\lambda^2}{4}\lVert\mathbf{w}\rVert^2$ and the wrong knob entirely. Differentiate:

    $$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{X}^T\mathbf{X}\mathbf{w} - \mathbf{X}^T\mathbf{y} + \lambda\mathbf{w}$$

    Set to zero and solve — still closed form:

    $$\boxed{\;\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda E)^{-1}\mathbf{X}^T\mathbf{y}\;}$$

    This is **ridge regression**. The "ridge" is literally the diagonal $\lambda E$ laid along $\mathbf{X}^T\mathbf{X}$ to lift its small eigenvalues away from zero and guarantee a regular matrix.

    **The cost.** Because the penalty also minimizes $\lVert\mathbf{w}\rVert$, the solution gets pulled toward the origin — it is now **biased**. Crank $\lambda$ up and the weights shrink further toward zero; set $\lambda = 0$ and you're back to plain OLS. That knob is exactly the bias–variance trade made explicit and tunable. You give up the Gauss–Markov unbiasedness guarantee and buy variance reduction with it — and total error often *drops*, because Err only cares about the sum of the three terms, not which one you paid.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Linear model = one dot product.** Append $x_0 = 1$ so the intercept folds into $\mathbf{w}^T\mathbf{x}$. The matrix form $\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$ puts noise in the world, not the model.
    - **OLS has a closed form:** $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. No iteration required — but it needs that inverse to exist and be well-conditioned.
    - **Gauss–Markov needs exactly three things:** zero-mean noise, homoscedasticity, uncorrelated errors. **Normality is not one of them.**
    - **"OLS is optimal" is heavily qualified.** Best *among linear, unbiased* estimators. Both qualifiers are escape hatches.
    - **MSE isn't arbitrary — it's Gaussian noise in disguise.** Maximizing likelihood under $\epsilon \sim \mathcal{N}(0,\sigma^2)$ derives the OLS objective exactly. Choosing a loss means asserting a noise model.
    - **Log-likelihood, always.** Log is monotonic so the argmax survives, products become sums, and constants in the parameter drop out.
    - **Err = Bias² + Variance + σ².** Holds for any $y = f(\mathbf{x}) + \epsilon$ model, not just linear ones. $\hat{f}$ is random because the training set is random.
    - **Capacity up → variance up, bias down.** Overfitting is the high-variance end (memorizing); underfitting is the high-bias end (too weak to learn the pattern).
    - **σ² is a floor.** You cannot beat it. Test error below the noise floor means a bug or a leak.
    - **Multicollinearity is the quiet killer.** Near-dependent columns leave $\mathbf{X}^T\mathbf{X}$ technically invertible but with near-zero eigenvalues → huge inverse eigenvalues → unstable weights. Nothing throws an error.
    - **Ridge trades bias for stability on purpose.** $(\mathbf{X}^T\mathbf{X} + \lambda E)^{-1}\mathbf{X}^T\mathbf{y}$ — add a diagonal ridge, shrink weights toward zero, cut variance. $\lambda$ is the trade-off dial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Part 1 builds linear regression from the ground up three times over. First mechanically: define $\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$, minimize squared error, differentiate, and read off the normal equation $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ — which the Gauss–Markov theorem certifies as minimum-variance among linear unbiased estimators, given zero-mean, homoscedastic, uncorrelated noise (and notably *not* given normality). Second probabilistically: maximum likelihood, warmed up on a Bernoulli survey where the intuitive fraction turns out to be the MLE, then applied to regression under the extra assumption of Gaussian errors — where maximizing the log-likelihood reduces term-by-term to minimizing MSE, revealing that the squared-error loss was an assumption about noise all along. Third decompositionally: expanding $\mathbb{E}[(y - \hat{f})^2]$ splits any model's error into squared bias, variance, and an irreducible $\sigma^2$ floor, with capacity trading the first two against each other. That decomposition then justifies regularization: when multicollinearity makes $\mathbf{X}^T\mathbf{X}$ nearly singular and OLS weights swing wildly, ridge regression adds a diagonal $\lambda E$ to restore conditioning, accepting deliberate bias — shrinking weights toward zero — in exchange for variance low enough that total error falls.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **linear regression** = model predicting the target as a weighted sum of features; named for its estimator being linear in $\mathbf{y}$
    - **weights** = the model parameters $\mathbf{w}$ being fit
    - **bias term** = the fake constant feature $x_0 = 1$ that lets the intercept $w_0$ fold into the dot product (also called *intercept*; unrelated to statistical bias)
    - **design matrix** = the $n \times (m+1)$ matrix $\mathbf{X}$ of observations, including the leading column of ones
    - **full column rank** = columns are linearly independent; required for $(\mathbf{X}^T\mathbf{X})^{-1}$ to exist
    - **noise** = the random error term $\boldsymbol{\epsilon}$ in $\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$
    - **OLS** = ordinary least squares; fitting $\mathbf{w}$ by minimizing mean squared error
        - **normal equation** = the closed-form OLS solution $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
    - **MSE** = mean squared error, $\frac{1}{2n}\lVert\mathbf{y} - \mathbf{X}\mathbf{w}\rVert_2^2$
    - **estimator** = a rule computing a parameter guess from data
    - **linear estimator** = one that is a linear combination of the targets, with coefficients depending only on $\mathbf{X}$
    - **unbiased estimator** = one whose expectation equals the true parameter, $\mathbb{E}[\hat{w}_i] = w_i$
    - **Gauss–Markov theorem** = OLS is minimum-variance among linear unbiased estimators, given its three noise assumptions
        - **homoscedasticity** = all errors share the same finite variance $\sigma^2$
    - **heteroscedasticity** = the violation of homoscedasticity; error variance changes across observations
    - **BLUE** = Best Linear Unbiased Estimator; the property Gauss–Markov grants OLS
    - **MLE** = maximum likelihood estimation; choose the parameter making the observed data most probable
        - **likelihood** = probability of the observed data as a function of the parameter
        - **log-likelihood** = log of the likelihood; monotonic, so same argmax, but turns products into sums
    - **Bernoulli distribution** = distribution over $\{0, 1\}$ with $p(x) = \theta^x(1-\theta)^{1-x}$
    - **normal distribution** = the Gaussian $\mathcal{N}(\mu, \sigma^2)$; assuming it for the noise is what makes MSE the ML objective
    - **bias–variance decomposition** = $\text{Err}(\mathbf{x}) = \text{Bias}(\hat{f})^2 + \text{Var}(\hat{f}) + \sigma^2$
        - **bias** = $(f - \mathbb{E}[\hat{f}])^2$; systematic offset of the average model from the truth
        - **variance** = $\text{Var}(\hat{f})$; how much the fit changes across different training sets
        - **irreducible error** = the $\sigma^2$ noise floor no model can beat
    - **overfitting** = high-variance regime; the model memorizes the training set instead of generalizing
    - **underfitting** = high-bias regime; the model is too weak to capture the pattern
    - **multicollinearity** = two or more features nearly linearly dependent, pushing eigenvalues of $\mathbf{X}^T\mathbf{X}$ toward zero and destabilizing the weights
    - **ill-posed problem** = one where $\mathbf{X}^T\mathbf{X}$ is singular, so no OLS solution exists
    - **regularization** = adding a penalty term to make the problem regular/stable, trading bias for lower variance
    - **Tikhonov regularization** = the general penalized form, adding $\lVert\Gamma\mathbf{w}\rVert^2$ to the loss for some matrix $\Gamma$
    - **ridge regression** = Tikhonov with $\Gamma = \sqrt{\lambda/2}\,E$, giving $\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda E)^{-1}\mathbf{X}^T\mathbf{y}$
    - **L2 norm** = $\lVert\mathbf{w}\rVert_2$; the quantity ridge penalizes, shrinking weights toward zero
    - **lambda** = the regularization strength $\lambda$; 0 recovers OLS, larger shrinks weights further
    """)
    return


if __name__ == "__main__":
    app.run()
