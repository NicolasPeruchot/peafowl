"""Clustering model."""
import warnings

import hdbscan
import pandas as pd
import umap

from gensim.models import Word2Vec

from peafowl.models.utils import viz_bokeh
from peafowl.preprocessing.utils import lemmatizer_dataset


warnings.filterwarnings("ignore", category=DeprecationWarning)


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

    def viz_topic(self, save: bool = False, name: str = "project_0"):
        """Viz with bokeh."""
        viz_bokeh(
            vectors=self.umap_vectors,
            words=list(self.embed_model.wv.index_to_key),
            label=[str(x) for x in self.clusterer.labels_],
            save=save,
            name=name,
        )
        return None
