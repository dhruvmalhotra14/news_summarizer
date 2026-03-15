import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# 1. LOAD THE AI MODEL (Explicit Class Loading)
@st.cache_resource
def load_summarizer():
    model_name = "google/flan-t5-small"
    # Explicitly loading model and tokenizer to avoid pipeline Registry errors
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_summarizer()

# 2. SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []

# 3. FRONTEND DESIGN
st.set_page_config(page_title="Professional News Summarizer", page_icon="📑", layout="wide")

with st.sidebar:
    st.title("📜 Summary History")
    if not st.session_state.history:
        st.write("No summaries yet.")
    else:
        for idx, h in enumerate(reversed(st.session_state.history)):
            st.write(f"{len(st.session_state.history) - idx}. {h}")
    
    st.divider()
    st.info("💡 **Tip:** Detailed summaries take 60-90s to generate.")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

st.title("📑 Professional News Summarizer")
st.write("Generating high-detail, long-form summaries.")

url = st.text_input("Enter Article URL here:", placeholder="https://www.ndtv.com/...")

if st.button("Generate Detailed Summary"):
    if url:
        start_time = time.time()
        status_text = st.empty()
        status_text.info("🔍 Initializing Stealth Browser...")