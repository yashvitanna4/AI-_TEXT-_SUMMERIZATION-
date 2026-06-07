pip install transformers torch
import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="AI Text Summarizer", layout="wide")
def load_summarizer():
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=device
    )
def summarize_text(text, max_length=150, min_length=40):
    """Generate summary with error handling"""
    try:
        result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        return result[0]['summary_text']
    except Exception as e:
        return f"Summarization failed: {str(e)}"
st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h1 class="main-header">🤖 AI Text Summarizer</h1>
    </div>
""", unsafe_allow_html=True)
if 'summerizer' not in st.session_state:
    with st.spinner ("Loading AI model... (first time only)"):
        st.session_state.summarizer = load_summarizer()
summarizer = st.session_state.summarizer

col1, col2 = st.columns([2, 1])
with col1:
    input_text = st.text_area(
        "Enter text to summarize",
        height=300,
        placeholder="Paste your article, document, or any long text here..."
    )
with col2:
    st.info("💡 **Tips:**\n\n- Works best with 200-2000 words\n- News articles, research papers, emails\n- Model auto-handles truncation")
if st.button("✨ Generate Summary", type="primary"):
    if input_text:
        with st.spinner("Generating summary..."):
            summary = summarize_text(input_text, 500, 50)
            
            # Display results
            orig_words = len(input_text.split())
            sum_words = len(summary.split())
            ratio = (sum_words/orig_words)*100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Words", orig_words)
            with col2:
                st.metric("Summary Words", sum_words)
            with col3:
                st.metric("Compression", f"{ratio:.1f}%")
            
            st.markdown("### 📝 Generated Summary")
            st.markdown(f'<div class="summary-box"><p>{summary}</p></div>', unsafe_allow_html=True)
            
            # Copy button
            st.code(summary, language="text")
            
    else:
        st.warning("Please enter some text first!")

