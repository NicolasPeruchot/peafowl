"""Clustering model."""
import warnings

import hdbscan
import pandas as pd
import umap

from gensim.models import Word2Vec

from peafowl.preprocessing.utils import lemmatizer_dataset


warnings.filterwarnings("ignore", category=DeprecationWarning)


class Cluster:
    """Train the clustering model."""

    def __init__(self, data: pd.Series) -> None:
        """Init."""
        lemmatized_data = lemmatizer_dataset(data)
        self.embed_model = Word2Vec(lemmatized_data, min_count=2, vector_size=300)
        self.reducer = umap.UMAP()
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=30)

    @property
    def vectors(self):
        """300D embeddings of the dataset, computed using Word2Vec."""
        return self.embed_model.wv.vectors

    @property
    def umap_vectors(self):
        """2D embeddings of the dataset, projected using UMAP."""
        return self.reducer.transform(self.vectors)

    def fit(self):
        """Cluster dataset using Word2Vec + UMAP + HDBSCAN."""
        self.reducer.fit(self.vectors)
        self.clusterer.fit(self.umap_vectors)

    def viz(self):
        """Viz with bokeh."""
        umap_mapper = self.reducer.fit(self.vectors)
        umap.plot.output_notebook()
        hover = pd.DataFrame({"word": self.embed_model.wv.index_to_key})
        labels = [str(x) for x in self.clusterer.labels_]
        p = umap.plot.interactive(umap_mapper, hover_data=hover, labels=labels, theme="viridis")
        umap.plot.show(p)
        return None
