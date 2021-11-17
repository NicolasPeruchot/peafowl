"""Clustering model."""
import hdbscan
import pandas as pd
import umap

from bokeh.io import output_file, show
from bokeh.models import CategoricalColorMapper, ColumnDataSource, HoverTool
from bokeh.palettes import plasma
from bokeh.plotting import figure
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

    def viz(self, save: bool = False, name: str = "project_0"):
        """Viz of the clustering."""
        list_x = self.umap_vectors[:, 0]
        list_y = self.umap_vectors[:, 1]
        desc = list(self.embed_model.wv.index_to_key)
        label = [str(x) for x in self.clusterer.labels_]

        source = ColumnDataSource(data=dict(x=list_x, y=list_y, desc=desc, label=label))
        hover = HoverTool(tooltips=[("Word", "@desc"),])
        mapper = CategoricalColorMapper(palette=plasma(len(set(label))), factors=list(set(label)))

        p = figure(plot_width=400, plot_height=400, tools=[hover], title="Clustering")
        p.circle("x", "y", size=10, source=source, color={"field": "label", "transform": mapper})
        if save:
            output_file(f"data/clustering_{name}.html")

        show(p)
        return None
