"""Streamlit app."""
import pandas as pd
import pyLDAvis
import streamlit as st

from datasets import load_dataset

from peafowl.models.lda import GuidedLda


st.title("GuidedLDA interface")

st.sidebar.file_uploader("Choose a file")

if st.sidebar.button("Use example dataset"):
    if "data" not in st.session_state:
        data_tot = load_dataset("amazon_reviews_multi", "en")["train"]
        data = data_tot["review_body"]
        data = pd.Series([e for i, e in enumerate(data) if i < 1000 or i > 198999])
        st.session_state.data = data


if "seeds" not in st.session_state:
    st.session_state.seeds = {}

topic_input = st.sidebar.text_input("Write topic name")

if st.sidebar.button("Add topic"):
    st.session_state.seeds[topic_input] = []


option = st.sidebar.selectbox("Seeds", (k for k, v in st.session_state.seeds.items()))

seed_input = st.sidebar.text_input("Write seed")

if st.sidebar.button("Add seed"):
    st.session_state.seeds[option].append(seed_input)


if st.button("Show representation"):
    model = GuidedLda(seeds=st.session_state.seeds)
    model.fit(data=st.session_state.data)
    prepared_data = model.viz()
    st.components.v1.html(pyLDAvis.prepared_data_to_html(prepared_data), width=1500, height=800)
