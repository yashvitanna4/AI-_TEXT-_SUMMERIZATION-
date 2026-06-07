import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Text Summarizer")

st.title("📝 AI Text Summarizer")

@st.cache_resource
def load_model():

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    return summarizer

summarizer = load_model()

text = st.text_area("Enter your text")

if st.button("Summarize"):

    if text.strip() == "":
        st.warning("Please enter text")

    else:

        result = summarizer(
            text,
            max_length=100,
            min_length=20,
            do_sample=False
        )

        st.subheader("Summary")
        st.write(result[0]["summary_text"])
