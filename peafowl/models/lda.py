"""Functions for the GuidedLDA interface."""

import warnings

from typing import List

import ipywidgets as widgets
import numpy as np
import pandas as pd
import pyLDAvis
import tomotopy as tp
import umap

from gensim.models import Word2Vec
from IPython.display import display

from peafowl.models.utils import prepare_viz_LDA, viz_bokeh
from peafowl.preprocessing.utils import lemmatizer_dataset


warnings.filterwarnings("ignore", category=DeprecationWarning)


class LDA:
    """LDA."""

    def __init__(self, k: int, is_guided: bool = True) -> None:
        """Init."""
        self.k = k
        self.topics: List[str] = []
        self.is_guided = is_guided
        self.model = tp.LDAModel(k=self.k)

        self._button_topic = widgets.Button(
            description="Add topic", button_style="", tooltip="Add topic", value="",
        )
        self._topic_name_widget = widgets.Text(
            value="", placeholder="Topic name", description="New topic:", disabled=False
        )
        self._button_seed = widgets.Button(
            description="Add seed", button_style="", tooltip="Add seed",
        )
        self._seed_widget = widgets.Text(
            value="", placeholder="Word", description="New seed:", disabled=False
        )
        self._topic_text = widgets.Textarea(
            value="", placeholder="", description="", disabled=False
        )
        self._seed_text = widgets.Textarea(value="", placeholder="", description="", disabled=False)

    def _on_button_seed(self, b: widgets.Button) -> None:
        """Button for seeds."""
        self.output_seed.clear_output()
        topic = self.dropdown_topics.value
        word = self._seed_widget.value
        if word not in self.seeds[topic] and word:
            self.seeds[topic].append(word)
        self._seed_text.value = "\n".join([k + ": " + ", ".join(v) for k, v in self.seeds.items()])
        with self.output_seed:
            display(self._seed_text)

    def explore(self, data: pd.Series, size: int = 10):
        """Show examples of data."""
        pd.set_option("display.max_colwidth", None)
        return data.sample(size)

    def _on_button_topic(self, b: widgets.Button) -> None:
        """Action on click for seed button."""
        self.output_topic.clear_output()
        topic = self._topic_name_widget.value
        if topic not in self.topics and topic:
            self.topics.append(topic)
        self._topic_text.value = "\n".join(self.topics)
        with self.output_topic:
            display(self._topic_text)

    def add_topics(self):
        """Add topics with widgets."""
        self.output_topic = widgets.Output()
        self._button_topic.on_click(self._on_button_topic)
        display(self._topic_name_widget)
        display(self._button_topic, self.output_topic)

    def add_seeds(self):
        """Add seeds with widgets."""
        self.output_seed = widgets.Output()
        self.dropdown_topics = widgets.Dropdown(
            options=self.topics, description="Topic:", disabled=False,
        )
        self.seeds = {topic: [] for topic in self.topics}
        self._button_seed.on_click(self._on_button_seed)
        display(self.dropdown_topics)
        display(self._seed_widget)
        display(self._button_seed, self.output_seed)

    def fit(self, data: pd.Series):
        """Train on data."""
        data = lemmatizer_dataset(data)
        for x in data:
            self.model.add_doc(x)
        if self.is_guided:
            n = len(self.seeds)
            for k, v in self.seeds.items():
                for word in v:
                    self.model.set_word_prior(word, [1.0 if k == i else 0 for i in range(n)])
        for _ in range(0, 100, 10):
            self.model.train(10)
        return None

    def viz(self):
        """Visualisation for a trained model."""
        prepared_data = prepare_viz_LDA(model=self.model)
        return pyLDAvis.display(prepared_data)

    def viz_2d(self, data: pd.Series, n: int = 1000):
        """Viz with bokeh."""
        lemmatized_data = lemmatizer_dataset(data)
        embed_model = Word2Vec(lemmatized_data, min_count=2, vector_size=300)
        reducer = umap.UMAP()
        vectors = embed_model.wv.vectors
        umap_vectors = reducer.fit_transform(vectors)
        mapping = dict(zip(embed_model.wv.index_to_key, umap_vectors))
        coordinates = []
        label = []
        words = []
        for i in range(2):
            for word in self.model.get_topic_words(topic_id=i, top_n=n):
                words.append(word[0])
                coordinates.append(mapping[word[0]])
                label.append(str(i))
        coordinates = np.array(coordinates)
        viz_bokeh(vectors=coordinates, words=words, label=label)
        return None
