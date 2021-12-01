"""Streamlit app."""
import pandas as pd
import streamlit as st

from datasets import load_dataset

from peafowl.models.lda import LDA


model = LDA(k=2, is_guided=False)

st.sidebar.file_uploader("Choose a file")

if st.sidebar.button("Use example dataset"):
    data_tot = load_dataset("amazon_reviews_multi", "en")["train"]
    data = data_tot["review_body"]
    data = pd.Series([e for i, e in enumerate(data) if i < 1000 or i > 198999])


if "seeds" not in st.session_state:
    st.session_state.seeds = []

seed_input = st.sidebar.text_input("Write seed")

if st.sidebar.button("Add seed"):
    st.session_state.seeds.append(seed_input)

if st.sidebar.button("Show seeds"):
    st.write(st.session_state.seeds)


with open("data/ldavis.html", "r", encoding="utf-8") as f:
    text = f.read()

# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?", ("Email", "Home phone", "Mobile phone")
# )


if st.button("Say hello"):
    st.write("aller la")
    st.components.v1.html(text, width=1500, height=800)
