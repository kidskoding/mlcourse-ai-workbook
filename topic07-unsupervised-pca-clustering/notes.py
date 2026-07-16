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
    # Topic 7: Unsupervised Learning — PCA and Clustering

    **Unsupervised learning** = learning structure from data with *no labels*. Two consequences follow, and both matter more than the algorithms themselves:

    - **Upside: data is cheap.** Nobody has to hand-label anything, so you can throw far more data at it.
    - **Downside: there is no scorecard.** With no ground truth there is no accuracy to optimize or report. "Is this good?" becomes a judgement call backed by proxy metrics.

    This topic covers the two workhorses: **PCA** (compress many features into a few) and **clustering** (group similar points). PCA earns its keep in three ways — plotting high-dimensional data in 2D, killing multicollinearity, and shrinking the feature space before a supervised model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA: rotate to where the variance is

    **Principal Component Analysis** projects the data onto a new orthogonal basis, chosen so that the first axis captures as much variance as possible, the second captures as much of what's left (while staying perpendicular to the first), and so on. Keep the top-$k$ axes, throw away the rest, and you've dropped from $n$ dimensions to $k$.

    The mental picture: your point cloud is an ellipsoid sitting in feature space. PCA finds the ellipsoid's own axes and re-expresses every point in terms of them. Because the new axes are orthogonal, correlated features get merged — that's the multicollinearity fix for free.

    The greedy part is the whole trick: pick the direction of maximum variance, then the next, and so on. Directions with tiny variance are assumed to be noise or redundancy and get cut.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The math (short version)

    Covariance of two features, where $\mu_i = E[X_i]$:

    $$\operatorname{cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)] = E[X_i X_j] - \mu_i\mu_j$$

    Stack these into the **covariance matrix** $\Sigma = E[(X - E[X])(X - E[X])^T]$. It's symmetric, with each feature's variance on the diagonal and the pairwise covariances off it. For centered data it's just $X^TX$ (up to a constant).

    Now the payoff: an **eigenvector** $w_i$ of a matrix $M$ is a direction that only stretches — it doesn't rotate — under $M$, and its **eigenvalue** $\lambda_i$ is the stretch factor: $Mw_i = \lambda_i w_i$. By the Rayleigh quotient, the direction of maximum variance in the sample is the eigenvector of $\Sigma$ with the largest eigenvalue.

    So: **the principal components are the top-$k$ eigenvectors of the covariance matrix, ranked by eigenvalue.** Multiply $X$ by them and you have your projection. Keep $k < n$ and you lose information — the eigenvalues tell you exactly how much.

    **Trap: center the data first** (subtract the column means). PCA measures variance around the origin; if the cloud isn't centered, the first component just points at the mean. `sklearn`'s `PCA` centers internally, but if you hand-roll it, or want to be explicit, subtract the mean yourself. Scaling matters too — a feature measured in millimetres will dominate one measured in metres purely by unit choice.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 1: iris — does PCA help a weak model?

    Baseline first: a decision tree capped at `max_depth=2` on all 4 raw iris features. Depth 2 is deliberately too shallow to fit the data properly — that's the point, we want a model with room to improve.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import datasets, decomposition
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    sns.set(style='white')
    iris = datasets.load_iris()
    X, y = (iris.data, iris.target)
    _X_train, _X_test, _y_train, _y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    _clf = DecisionTreeClassifier(max_depth=2, random_state=42).fit(_X_train, _y_train)
    _preds = _clf.predict_proba(_X_test)
    print('Accuracy: {:.5f}'.format(accuracy_score(_y_test, _preds.argmax(axis=1))))
    return (
        DecisionTreeClassifier,
        X,
        accuracy_score,
        datasets,
        decomposition,
        iris,
        np,
        pd,
        plt,
        train_test_split,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now compress 4 features to 2 principal components and refit the same depth-2 tree. Accuracy goes from **0.889 → 0.911** — a modest bump here, because iris is 4-dimensional and already easy. The lesson generalizes: on genuinely high-dimensional data, PCA before a tree (or a tree ensemble) can help a lot, because trees split on one axis at a time and PCA hands them axes that actually carry the signal.
    """)
    return


@app.cell
def _(X, decomposition, plt, y):
    pca = decomposition.PCA(n_components=2)
    X_centered = X - X.mean(axis=0)
    X_pca = pca.fit_transform(X_centered)

    plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], "bo", label="Setosa")
    plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], "go", label="Versicolour")
    plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], "ro", label="Virginica")
    plt.legend(loc=0)
    plt.title("Iris, 4D -> 2D via PCA");
    return X_pca, pca


@app.cell
def _(DecisionTreeClassifier, X_pca, accuracy_score, train_test_split, y):
    _X_train, _X_test, _y_train, _y_test = train_test_split(X_pca, y, test_size=0.3, stratify=y, random_state=42)
    _clf = DecisionTreeClassifier(max_depth=2, random_state=42).fit(_X_train, _y_train)
    _preds = _clf.predict_proba(_X_test)
    print('Accuracy: {:.5f}'.format(accuracy_score(_y_test, _preds.argmax(axis=1))))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the components

    Two attributes carry everything you need to interpret a fitted PCA:

    - **`explained_variance_ratio_`** — the fraction of total variance each component captures. For iris: PC1 = 92.5%, PC2 = 5.3%. Two components hold ~98% of the information in four features.
    - **`components_`** — the **loadings**: each component written as a weighted combination of the original features. PC1 for iris is dominated by petal length (0.857), which matches the botany — petal size is what separates the species.

    Loadings are how you tell a stakeholder what a component *means*. They're also the honest answer to "PCA is a black box": it isn't, the components are literally linear formulas over your columns.
    """)
    return


@app.cell
def _(iris, pca):
    for _i, component in enumerate(pca.components_):
        print('{} component: {}% of initial variance'.format(_i + 1, round(100 * pca.explained_variance_ratio_[_i], 2)))
        print(' + '.join(('%.3f x %s' % (value, name) for value, name in zip(component, iris.feature_names))))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 2: handwritten digits — 64D down to 2D

    The digits dataset is 8×8 grayscale images, each flattened to a 64-number vector. That's 64 features. Project to 2 and plot: the digit classes already fall into visible clumps, and PCA never saw a single label. That's the visualization use-case in one picture.
    """)
    return


@app.cell
def _(datasets, decomposition, plt):
    digits = datasets.load_digits()
    X_1, y_1 = (digits.data, digits.target)
    pca_1 = decomposition.PCA(n_components=2)
    X_reduced = pca_1.fit_transform(X_1)
    print('Projecting %d-dimensional data to 2D' % X_1.shape[1])
    plt.figure(figsize=(12, 10))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_1, edgecolor='none', alpha=0.7, s=40, cmap=plt.get_cmap('nipy_spectral', 10))
    plt.colorbar()
    plt.title('Digits. PCA projection')
    return X_1, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCA vs t-SNE

    **t-SNE** is the other standard way to get a 2D picture, and it separates the digit clusters visibly better. Why: **PCA is linear** — it can only rotate and project, so structure that curves gets flattened badly. t-SNE is nonlinear and optimizes to keep near neighbours near.

    The trade: t-SNE is much slower even on a dataset this small, its output is not a reusable transform (it's a one-off embedding, not a matrix you can apply to new points), and inter-cluster distances in a t-SNE plot mean essentially nothing. **Rule of thumb: PCA for preprocessing, t-SNE for looking.**
    """)
    return


@app.cell
def _(X_1, plt, y_1):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    from sklearn.manifold import TSNE
    X_tsne = TSNE(random_state=17).fit_transform(X_1)
    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_1, edgecolor='none', alpha=0.7, s=40, cmap=plt.get_cmap('nipy_spectral', 10))
    plt.colorbar()
    plt.title('Digits. t-SNE projection')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How many components to keep?

    The practical recipe: **fit PCA with all components, plot the cumulative `explained_variance_ratio_`, and keep enough components to reach ~90% of the variance.** For digits that's 21 components — 64 features down to 21 with a 10% information haircut. No guessing, no grid search.
    """)
    return


@app.cell
def _(X_1, decomposition, np, plt):
    pca_2 = decomposition.PCA().fit(X_1)
    plt.figure(figsize=(10, 7))
    plt.plot(np.cumsum(pca_2.explained_variance_ratio_), color='k', lw=2)
    plt.xlabel('Number of components')
    plt.ylabel('Total explained variance')
    plt.xlim(0, 63)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axvline(21, c='b')
    plt.axhline(0.9, c='r')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clustering

    The premise: the points look like they fall into groups; name the groups, and assign new points to the right one. No labels involved. The algorithms below are the common ones, not an exhaustive list — they differ mainly in **whether you must specify the number of clusters up front** and **what shape of cluster they can find**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### K-means

    The default choice, and the simplest:

    1. Pick $k$, the number of clusters.
    2. Drop $k$ **centroids** at random positions in the data space.
    3. Assign each point to its nearest centroid.
    4. Move each centroid to the mean of the points assigned to it.
    5. Repeat 3–4 until the centroids stop moving (or you hit a step limit).

    That's it. Euclidean distance is the default but any metric works. Below: three synthetic Gaussian blobs, then three hand-rolled iterations so you can watch the centroids walk into place.
    """)
    return


@app.cell
def _(np, plt):
    X_2 = np.zeros((150, 2))
    np.random.seed(seed=42)
    X_2[:50, 0] = np.random.normal(loc=0.0, scale=0.3, size=50)
    X_2[:50, 1] = np.random.normal(loc=0.0, scale=0.3, size=50)
    X_2[50:100, 0] = np.random.normal(loc=2.0, scale=0.5, size=50)
    X_2[50:100, 1] = np.random.normal(loc=-1.0, scale=0.2, size=50)
    X_2[100:150, 0] = np.random.normal(loc=-1.0, scale=0.2, size=50)
    X_2[100:150, 1] = np.random.normal(loc=2.0, scale=0.5, size=50)
    plt.figure(figsize=(5, 5))
    plt.plot(X_2[:, 0], X_2[:, 1], 'bo')
    plt.title('Three synthetic blobs')
    return (X_2,)


@app.cell
def _(X_2, np, plt):
    from scipy.spatial.distance import cdist
    np.random.seed(seed=42)
    centroids = np.random.normal(loc=0.0, scale=1.0, size=6).reshape((3, 2))
    cent_history = [centroids]
    for _i in range(3):
        distances = cdist(X_2, centroids)
        labels = distances.argmin(axis=1)
        centroids = centroids.copy()
        for c in range(3):
            centroids[c, :] = np.mean(X_2[labels == c, :], axis=0)
        cent_history.append(centroids)
    plt.figure(figsize=(8, 8))
    for _i in range(4):
        labels = cdist(X_2, cent_history[_i]).argmin(axis=1)
        plt.subplot(2, 2, _i + 1)
        plt.plot(X_2[labels == 0, 0], X_2[labels == 0, 1], 'bo', label='cluster #1')
        plt.plot(X_2[labels == 1, 0], X_2[labels == 1, 1], 'co', label='cluster #2')
        plt.plot(X_2[labels == 2, 0], X_2[labels == 2, 1], 'mo', label='cluster #3')
        plt.plot(cent_history[_i][:, 0], cent_history[_i][:, 1], 'rX')
        plt.legend(loc=0)
        plt.title('Step {:}'.format(_i + 1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choosing $k$: the elbow

    K-means minimizes the sum of squared distances from points to their own centroid — **inertia**, exposed as `kmeans.inertia_`:

    $$J(C) = \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2 \rightarrow \min_C$$

    **The trap:** $J$ is monotonically decreasing in $k$. Minimize it naively and you get $k = n$ — every point its own cluster, inertia zero, insight zero. $J$ can never pick $k$ for you.

    **The fix — the elbow method:** plot $J$ against $k$ and take the point where the curve stops falling fast. Formally, minimize the ratio of successive drops:

    $$D(k) = \frac{|J(C_k) - J(C_{k+1})|}{|J(C_{k-1}) - J(C_k)|} \rightarrow \min_k$$

    On our blobs the curve bends hard at $k=3$ and flattens after — which is exactly how we generated the data.
    """)
    return


@app.cell
def _(X_2, np, plt):
    from sklearn.cluster import KMeans
    inertia = []
    for k in range(1, 8):
        kmeans = KMeans(n_clusters=k, random_state=1, n_init='auto').fit(X_2)
        inertia.append(np.sqrt(kmeans.inertia_))
    plt.plot(range(1, 8), inertia, marker='s')
    plt.xlabel('$k$')
    plt.ylabel('$J(C_k)$')
    return (KMeans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### K-means gotchas

    - **It's NP-hard.** Exact solution costs $O(n^{dk+1})$ for $n$ points, $d$ dimensions, $k$ clusters. What you actually run is a heuristic that finds *a* local optimum, not *the* optimum.
    - **Initialization decides the answer.** Different random centroid seeds → different clusters. The mitigation is baked in: `n_init` reruns the whole thing from several seeds and keeps the best; these runs parallelize.
    - **Big data:** MiniBatch K-means updates centroids from random batches instead of the full dataset. Much faster, slightly worse clusters.
    - **Shape assumption:** minimizing squared distance to a centre means K-means wants round, similarly-sized blobs. Elongated or nested clusters break it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Affinity Propagation

    **No need to pick $k$ up front** — the algorithm decides how many clusters there are. It works on a **similarity** $s(x_i, x_j)$ (a natural choice is negative squared distance, $s(x_i,x_j) = -\|x_i - x_j\|^2$) and has points negotiate over which of them should serve as **exemplars** ("role models").

    Two matrices, both starting at zero, updated until they settle:

    - **responsibility** $r_{i,k}$ — how well-suited point $k$ is as exemplar for point $i$, relative to all rival exemplars;
    - **availability** $a_{i,k}$ — how appropriate it would be for $i$ to pick $k$ as its exemplar, given $k$'s support from everyone else.

    $$r_{i,k} \leftarrow s(x_i, x_k) - \max_{k' \neq k}\{a_{i,k'} + s(x_i, x_{k'})\}$$
    $$a_{i,k} \leftarrow \min\left(0,\; r_{k,k} + \sum_{i' \notin \{i,k\}} \max(0, r_{i',k})\right),\ i \neq k$$
    $$a_{k,k} \leftarrow \sum_{i' \neq k} \max(0, r_{i',k})$$

    The message-passing reads as messy but the idea is simple: points vote for exemplars, exemplars accumulate support, and the vote converges. Cost: it tends to over-produce clusters (see the metrics table below).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spectral clustering

    Recast clustering as a **graph-cut** problem. Build the **adjacency matrix** $A_{i,j} = -\|x_i - x_j\|^2$ — a fully connected graph where every point is a vertex and edge weights are similarities. Then split the graph so that points inside a piece are similar to each other and pieces are dissimilar to each other; formally that's the **normalized cuts** problem, solved via the eigenvectors of the graph Laplacian.

    Because it works on graph connectivity rather than distance-to-a-centre, spectral clustering finds **non-convex** clusters that K-means can't — rings, spirals, chains. It scores best-in-class alongside agglomerative on the digits table below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Agglomerative (hierarchical) clustering

    The easiest to explain of the no-fixed-$k$ methods — it's just repeated merging:

    1. Every point starts as its own cluster.
    2. Sort the pairwise distances between cluster centres.
    3. Merge the two closest clusters; recompute the centre.
    4. Repeat until one cluster remains.

    "Closest" needs defining, and the **linkage** choice changes the outcome:

    - **single linkage** — $d(C_i, C_j) = \min_{x_i \in C_i, x_j \in C_j} \|x_i - x_j\|$ (nearest pair; tends to chain clusters together)
    - **complete linkage** — $\max_{x_i \in C_i, x_j \in C_j} \|x_i - x_j\|$ (farthest pair; compact clusters)
    - **average linkage** — $\frac{1}{n_i n_j}\sum_{x_i \in C_i}\sum_{x_j \in C_j} \|x_i - x_j\|$
    - **centroid linkage** — $\|\mu_i - \mu_j\|$ (cheapest: no re-computing all pairwise distances on every merge)

    The payoff is the **dendrogram**: the whole merge history as a tree. You don't pick $k$ before running — you run once, look at where the merge heights jump, and cut the tree there. On our three blobs the three groups are unmistakable.
    """)
    return


@app.cell
def _(X_2, plt):
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    distance_mat = pdist(X_2)
    Z = hierarchy.linkage(distance_mat, 'single')  # upper triangle of pairwise distances
    plt.figure(figsize=(10, 5))  # the agglomerative merge history
    dn = hierarchy.dendrogram(Z, color_threshold=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluating a clustering

    This is the hard part of unsupervised learning. A metric can't lean on labels — and usually you don't have any, which is why you were clustering. Metrics split in two:

    - **External** — you *do* have true labels (e.g. benchmarking on a labelled dataset) and compare the split against them.
    - **Internal** — labels unknown; judge the split from the data geometry alone. These are what you use to pick $k$ in the wild.

    All of them live in `sklearn.metrics`.

    **Adjusted Rand Index (ARI)** *(external)* — count pairs of points the two splits agree on. With $a$ = pairs sharing a label *and* a cluster, $b$ = pairs differing in both:

    $$RI = \frac{2(a+b)}{n(n-1)}, \qquad ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$$

    The adjustment is what makes it useful — it subtracts the score a *random* clustering would get, so chance agreement doesn't inflate it. Range $[-1, 1]$: 1 = identical splits, ~0 = no better than random, negative = worse than random. Symmetric, and immune to label permutation (cluster "3" vs class "7" doesn't matter, only the grouping does).

    **Adjusted Mutual Information (AMI)** *(external)* — same spirit, entropy-based. Treat each split as a discrete distribution and measure their **mutual information**: how much knowing one split reduces uncertainty about the other. Raw MI creeps up as clusters multiply; the adjustment removes that. Range $[0, 1]$, 1 = match.

    **Homogeneity / completeness / V-measure** *(external)* — with $C$ = true split, $K$ = clustering:

    $$h = 1 - \frac{H(C \mid K)}{H(C)}, \qquad c = 1 - \frac{H(K \mid C)}{H(K)}, \qquad v = \frac{2hc}{h+c}$$

    **$h$** = is each cluster made of one true class? **$c$** = does each true class land in one cluster? They're not symmetric — you can max one by cheating on the other (put every point in its own cluster: perfect homogeneity, terrible completeness). **V-measure** is their harmonic mean and is symmetric. **Trap:** unlike ARI/AMI these are *unscaled*, so they drift with the number of clusters — with few observations and many clusters they flatter a random split. Prefer ARI there; above ~100 observations and under ~10 clusters, ignore the issue.

    **Silhouette** *(internal — the one you can actually use without labels)* — per point, with $a$ = mean distance to its own cluster-mates and $b$ = mean distance to the nearest *other* cluster:

    $$s = \frac{b - a}{\max(a, b)}$$

    Average it over the sample. Range $[-1, 1]$: near 1 = dense, well-separated clusters; near -1 = points are sitting in the wrong cluster. **Pick $k$ by maximizing silhouette** — this is the label-free alternative to eyeballing an elbow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### All four algorithms on digits

    Note the story in the numbers: Affinity Propagation posts near-perfect **homogeneity (0.96)** but dismal **completeness (0.49)** — the classic signature of *too many clusters*: every cluster is pure because each holds a fragment of one digit, but each digit is scattered across many clusters. Its ARI (0.17) exposes what homogeneity alone hides. Agglomerative and Spectral beat K-means across the board. And silhouette barely separates anyone (all ~0.12–0.18) — in 64 dimensions everything is far from everything, so the internal metric loses discriminating power.
    """)
    return


@app.cell
def _(KMeans, datasets, pd):
    from sklearn import metrics
    from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, SpectralClustering
    data = datasets.load_digits()
    X_3, y_2 = (data.data, data.target)
    algorithms = [KMeans(n_clusters=10, random_state=1, n_init='auto'), AffinityPropagation(), SpectralClustering(n_clusters=10, random_state=1, affinity='nearest_neighbors'), AgglomerativeClustering(n_clusters=10)]
    rows = []
    for algo in algorithms:
        algo.fit(X_3)
        rows.append({'ARI': metrics.adjusted_rand_score(y_2, algo.labels_), 'AMI': metrics.adjusted_mutual_info_score(y_2, algo.labels_), 'Homogeneity': metrics.homogeneity_score(y_2, algo.labels_), 'Completeness': metrics.completeness_score(y_2, algo.labels_), 'V-measure': metrics.v_measure_score(y_2, algo.labels_), 'Silhouette': metrics.silhouette_score(X_3, algo.labels_)})
    pd.DataFrame(rows, index=['K-means', 'Affinity', 'Spectral', 'Agglomerative'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Unsupervised = no labels, and therefore no scorecard.** Cheap data, expensive evaluation. Every metric is a proxy.
    - **PCA = eigenvectors of the covariance matrix, ranked by eigenvalue.** Top-$k$ directions of maximum variance, orthogonal by construction.
    - **Center (and usually scale) before PCA.** Otherwise the first component points at the mean, or at whatever feature has the biggest units.
    - **Pick $k$ components from cumulative `explained_variance_ratio_`** — the ~90% rule. Digits: 64 → 21.
    - **`components_` are readable.** Loadings tell you what each component is made of; PCA is not a black box.
    - **PCA is linear; t-SNE isn't.** PCA to preprocess and to reuse on new data, t-SNE only to look — it's slow and its distances lie.
    - **K-means is fast, simple, and picks fights with its own weaknesses:** you must supply $k$, initialization changes the answer (use `n_init`), and it only finds round blobs.
    - **Inertia can never choose $k$** — it hits zero at $k = n$. Use the elbow, or maximize silhouette.
    - **No-$k$-needed options exist:** agglomerative (cut the dendrogram after the fact), affinity propagation (exemplar voting), spectral (graph cuts, handles non-convex shapes).
    - **Read cluster metrics in pairs.** High homogeneity + low completeness = too many clusters. Silhouette is the only one here that works without labels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Unsupervised learning drops the labels, which buys unlimited data and costs you your evaluation metric. PCA handles dimensionality reduction by diagonalizing the covariance matrix: the principal components are its top eigenvectors, each explaining a known share of the variance, so you keep components until cumulative explained variance hits ~90% (64 → 21 on digits) and read the loadings to interpret them. It's linear, cheap, and reusable — t-SNE draws prettier pictures but is slow, one-off, and metrically untrustworthy. Clustering groups points without labels: K-means iterates assign-then-recenter and is the default despite needing $k$ up front, being init-sensitive, and only finding round clusters; agglomerative merges bottom-up into a dendrogram you cut later; affinity propagation elects exemplars; spectral clustering cuts a similarity graph and handles non-convex shapes. Choosing $k$ is the recurring trap — inertia is monotone and useless alone, so you take the elbow or maximize silhouette. Judging the result means ARI/AMI/V-measure when labels exist and silhouette when they don't, always reading homogeneity against completeness so a shredded clustering can't masquerade as a pure one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Glossary / Terms

    - **unsupervised learning** = learning structure from unlabeled data
    - **dimensionality reduction** = re-expressing data in fewer features while keeping most of the information
    - **covariance matrix** = symmetric matrix of feature variances (diagonal) and covariances (off-diagonal)
    - **eigenvector** = direction a matrix only stretches, never rotates
    - **eigenvalue** = that direction's stretch factor; ranks the components
    - **PCA** = project onto orthogonal axes ordered by variance explained
        - **principal component** = one such axis; an eigenvector of the covariance matrix
        - **centering** = subtracting column means, the first step of the fit
        - **explained variance ratio** = share of total variance a component captures
        - **loadings** = a component's weights over the original features; how you interpret it
    - **multicollinearity** = features that are strongly correlated with each other
    - **t-SNE** = nonlinear 2D embedding for visualization; slow, one-off, distances not meaningful
    - **clustering** = grouping similar observations with no labels
    - **centroid** = a cluster's mean point
    - **K-means** = repeatedly assign points to nearest centroid, then move centroids to member means
        - **inertia** = sum of squared distances from points to their centroid; $J(C)$
        - **n_init** = number of random restarts kept-best-of, the fix for init sensitivity
    - **MiniBatch K-means** = K-means with centroid updates from random batches; faster, slightly worse
    - **elbow method** = pick $k$ where the inertia curve stops dropping steeply
    - **agglomerative clustering** = bottom-up merging of nearest clusters
        - **linkage** = the rule defining distance between two clusters (single/complete/average/centroid)
        - **dendrogram** = the merge tree; cut it to choose the number of clusters
    - **affinity propagation** = points vote for exemplars via responsibility/availability messages; picks its own $k$
        - **exemplar** = the representative observation a cluster forms around
    - **spectral clustering** = normalized graph cut on the similarity graph; finds non-convex clusters
        - **adjacency matrix** = pairwise similarity matrix viewed as a weighted graph
    - **external metric** = scores a clustering against known true labels
    - **internal metric** = scores a clustering from the data alone, no labels
    - **ARI** = chance-adjusted agreement of point pairs between two splits; $[-1, 1]$
    - **AMI** = chance-adjusted mutual information between two splits; $[0, 1]$
    - **homogeneity** = each cluster contains only one true class
    - **completeness** = each true class lands in only one cluster
    - **V-measure** = harmonic mean of homogeneity and completeness
    - **silhouette** = $(b-a)/\max(a,b)$ per point; label-free cluster quality, maximize it to choose $k$
    """)
    return


if __name__ == "__main__":
    app.run()
