import streamlit as st
from transformers import pipeline
import torch

st.title("AI Text Summarizer")

@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1

    summarizer = pipeline(
        task="summarization",
        model="sshleifer/distilbart-cnn-12-6",
        framework="pt",
        device=device
    )

    return summarizer

summarizer = load_summarizer()

text = st.text_area("Enter your text")

if st.button("Summarize"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        summary = summarizer(
            text,
            max_length=100,
            min_length=30,
            do_sample=False
        )

        st.subheader("Summary")
        st.write(summary[0]["summary_text"])
