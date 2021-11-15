"""Clustering model."""
import hdbscan
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
import seaborn as sns
import umap

from gensim.models import Word2Vec

from peafowl.preprocessing.utils import lemmatizer_dataset


class Cluster:
    """Train the clustering model."""

    def __init__(self, data: pd.Series) -> None:
        """Init."""
        self.data = data
        self.lemmatized_data = lemmatizer_dataset(data)
        self.embed_model = Word2Vec(self.lemmatized_data, min_count=2, vector_size=300)
        self.reducer = umap.UMAP()
        self.vectors = self.embed_model.wv.vectors
        self.umap_vectors = self.reducer.fit_transform(self.vectors)
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=30).fit(self.umap_vectors)

    def viz(self):
        """Viz of the clustering."""
        # %matplotlib widget
        color_palette = sns.color_palette("deep", len(np.unique(self.clusterer.labels_)))
        cluster_colors = [
            color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in self.clusterer.labels_
        ]
        cluster_member_colors = [
            sns.desaturate(x, p) for x, p in zip(cluster_colors, self.clusterer.probabilities_)
        ]

        fig, ax = plt.subplots()
        sc = plt.scatter(
            *self.umap_vectors.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25
        )

        cursor = mplcursors.cursor(sc, hover=True)

        @cursor.connect("add")
        def on_add(sel: mplcursors._pick_info.Selection):
            sel.annotation.set(text=list(self.embed_model.wv.index_to_key)[sel.target.index])
