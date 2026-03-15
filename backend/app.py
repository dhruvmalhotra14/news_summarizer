import streamlit as st
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# 1. LOAD THE AI MODEL
@st.cache_resource
def load_summarizer():
    # Changed task to "summarization" to fix the KeyError
    return pipeline("summarization", model="google/flan-t5-small")

summarizer = load_summarizer()

# 2. SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []