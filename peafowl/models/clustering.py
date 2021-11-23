"""Clustering model."""
import warnings

from typing import List

import gensim
import hdbscan
import pandas as pd
import umap

from gensim.models import Doc2Vec, Word2Vec

from peafowl.preprocessing.utils import lemmatizer_dataset


warnings.filterwarnings("ignore", category=DeprecationWarning)


class Cluster:
    """Train the clustering model."""

    def __init__(self, data: pd.Series) -> None:
        """Init."""
        lemmatized_data: List[List[str]] = lemmatizer_dataset(data)
        # Skip-gram Negative Sampling algorithm
        self.embed_model = Word2Vec(
            sentences=lemmatized_data,
            vector_size=100,
            window=5,
            min_count=5,
            sg=1,
            hs=0,
            negative=5,
            ns_exponent=0.0,
            alpha=0.05,
            sample=0.0001,
            epochs=10,
        )
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


class Cluster_docs(Cluster):
    """Cluster docs."""

    def __init__(self, data: pd.Series) -> None:
        """Init."""
        super().__init__(data)
        lemmatized_data: List[List[str]] = lemmatizer_dataset(data)
        gensim_docs = [
            gensim.models.doc2vec.TaggedDocument(data[i], tags=[i])
            for i in range(len(lemmatized_data))
        ]
        self.embed_model = Doc2Vec(
            documents=gensim_docs,
            vector_size=100,
            window=5,
            min_count=5,
            hs=0,
            negative=5,
            ns_exponent=0.0,
            alpha=0.05,
            sample=0.0001,
            epochs=10,
        )

    def viz(self):
        """Viz with bokeh."""
        umap_mapper = self.reducer.fit(self.vectors)
        umap.plot.output_notebook()
        # hover = pd.DataFrame({"word": self.embed_model.wv.index_to_key})
        # labels = [str(x) for x in self.clusterer.labels_]
        # p = umap.plot.interactive(umap_mapper, hover_data=hover, labels=labels, theme="viridis")
        p = umap.plot.interactive(umap_mapper, theme="viridis")
        umap.plot.show(p)
        return None
