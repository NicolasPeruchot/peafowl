"""Functions for the GuidedLDA interface."""

import warnings

from typing import List

import ipywidgets as widgets
import pandas as pd
import pyLDAvis
import tomotopy as tp

from IPython.display import display

from peafowl.preprocessing.utils import lemmatizer_dataset
from peafowl.viz.utils import prepare_viz_LDA


warnings.filterwarnings("ignore", category=DeprecationWarning)


class GuidedLDAInterface:
    """Interface."""

    def __init__(self, k: int) -> None:
        """Init."""
        self.k = k
        self.topics: List[str] = []
        self.button_topic = widgets.Button(
            description="Add topic", button_style="", tooltip="Add topic", value="",
        )

        self.topic_name_widget = widgets.Text(
            value="", placeholder="Topic name", description="New topic:", disabled=False
        )

        self.button_seed = widgets.Button(
            description="Add seed", button_style="", tooltip="Add seed",
        )

        self.seed_widget = widgets.Text(
            value="", placeholder="Word", description="New seed:", disabled=False
        )

        self.model = tp.LDAModel(k=2)

        self.topic_text = widgets.Textarea(value="", placeholder="", description="", disabled=False)

        self.seed_text = widgets.Textarea(value="", placeholder="", description="", disabled=False)

    def on_button_seed(self, b: widgets.Button):
        """Button for seeds."""
        self.output_seed.clear_output()
        topic = self.dropdown_topics.value
        word = self.seed_widget.value
        if word not in self.seeds[topic] and word:
            self.seeds[topic].append(word)
        self.seed_text.value = "\n".join([k + ": " + ", ".join(v) for k, v in self.seeds.items()])
        with self.output_seed:
            display(self.seed_text)

    def on_button_topic(self, b: widgets.Button):
        """Action on click for seed button."""
        self.output_topic.clear_output()
        topic = self.topic_name_widget.value
        if topic not in self.topics and topic:
            self.topics.append(topic)
        self.topic_text.value = "\n".join(self.topics)
        with self.output_topic:
            display(self.topic_text)

    def add_topics(self):
        """Add topics with widgets."""
        self.output_topic = widgets.Output()
        self.button_topic.on_click(self.on_button_topic)
        display(self.topic_name_widget)
        display(self.button_topic, self.output_topic)

    def add_seeds(self):
        """Add seeds with widgets."""
        self.output_seed = widgets.Output()
        self.dropdown_topics = widgets.Dropdown(
            options=self.topics, description="Topic:", disabled=False,
        )
        self.seeds = {topic: [] for topic in self.topics}
        self.button_seed.on_click(self.on_button_seed)
        display(self.dropdown_topics)
        display(self.seed_widget)
        display(self.button_seed, self.output_seed)

    def train(self, data: pd.Series):
        """Train on data."""
        data = lemmatizer_dataset(data)
        for x in data:
            self.model.add_doc(x)
        n = len(self.seeds)
        for k, v in self.seeds.items():
            for word in v:
                self.model.set_word_prior(word, [1.0 if k == i else 0 for i in range(n)])

        for _ in range(0, 100, 10):
            self.model.train(10)

        prepared_data = prepare_viz_LDA(model=self.model)
        return pyLDAvis.display(prepared_data)
