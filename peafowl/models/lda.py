"""Functions for the GuidedLDA interface."""

import warnings

from typing import List

import ipywidgets as widgets
import pandas as pd
import pyLDAvis
import tomotopy as tp
import umap
import umap.plot

from gensim.models import Word2Vec
from IPython.display import display

from peafowl.models.utils import prepare_viz_LDA
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

    def report(self):
        """Get a feel for the model's performance (topic labelling + topic coherence).

        References:
            - https://bab2min.github.io/tomotopy/v0.12.1/en/label.html
            - https://bab2min.github.io/tomotopy/v0.12.1/en/coherence.html
        """
        # extract candidates for auto topic labeling
        extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
        cands = extractor.extract(self.model)

        # ranking the candidates of labels for a specific topic
        labeler = tp.label.FoRelevance(self.model, cands, min_df=5, smoothing=1e-2, mu=0.25)
        for k in range(self.model.k):
            print("== Topic #{} ==".format(k))
            print(
                "Labels:", ", ".join(label for label, score in labeler.get_topic_labels(k, top_n=5))
            )
            for word, prob in self.model.get_topic_words(k, top_n=10):
                print(word, prob, sep="\t")

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
        umap_mapper = reducer.fit(vectors)
        umap.plot.output_notebook()
        hover = pd.DataFrame({"word": embed_model.wv.index_to_key})
        p = umap.plot.interactive(umap_mapper, hover_data=hover)
        umap.plot.show(p)
        return None
