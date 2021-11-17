"""Viz with pyLDAvis."""
import warnings

from typing import List

import numpy as np
import pyLDAvis
import tomotopy

from bokeh.io import output_file, show
from bokeh.models import CategoricalColorMapper, ColumnDataSource, HoverTool
from bokeh.palettes import plasma
from bokeh.plotting import figure


def prepare_viz_LDA(model: tomotopy.LDAModel) -> pyLDAvis._prepare.PreparedData:
    """Prepare LDA Model for viz.

    Ref: https://github.com/bab2min/tomotopy/blob/main/examples/lda_visualization.py
    """
    topic_term_dists: np.ndarray = np.stack([model.get_topic_word_dist(k) for k in range(model.k)])
    doc_topic_dists: np.ndarray = np.stack([doc.get_topic_dist() for doc in model.docs])
    doc_lengths: np.ndarray = np.array([len(doc.words) for doc in model.docs])
    vocab: List[str] = list(model.used_vocabs)
    term_frequency: np.ndarray = model.used_vocab_freq

    warnings.filterwarnings("ignore")
    prepared_data = pyLDAvis.prepare(
        topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency
    )
    return prepared_data


def viz_bokeh(
    vectors: np.array,
    words: List[str],
    label: List[str],
    save: bool = False,
    name: str = "project_0",
):
    """Viz of the clustering."""
    list_x = vectors[:, 0]
    list_y = vectors[:, 1]
    desc = words
    source = ColumnDataSource(data=dict(x=list_x, y=list_y, desc=desc, label=label))
    hover = HoverTool(tooltips=[("Word", "@desc"),])
    mapper = CategoricalColorMapper(palette=plasma(len(set(label))), factors=list(set(label)))

    p = figure(plot_width=800, plot_height=800, tools=[hover], title="Clustering")
    p.circle("x", "y", size=10, source=source, color={"field": "label", "transform": mapper})
    if save:
        output_file(f"data/clustering_{name}.html")

    show(p)
    return None
