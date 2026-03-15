import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from newspaper import Article # Universal extraction logic
import time

# 1. LOAD THE AI MODEL
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "google/flan-t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# 2. SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []

# 3. FRONTEND DESIGN
st.set_page_config(page_title="Universal AI News Summarizer", page_icon="🌍", layout="wide")

with st.sidebar:
    st.title("📜 Summary History")
    if not st.session_state.history:
        st.write("No summaries yet.")
    else:
        for idx, h in enumerate(reversed(st.session_state.history)):
            st.write(f"{len(st.session_state.history) - idx}. {h}")
    
    st.divider()
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

st.title("🌍 Universal AI News Summarizer")
st.write("Extract and summarize news from NDTV, BBC, CNN, Reuters, and more.")

url = st.text_input("Paste any news article URL here:", placeholder="https://www.bbc.com/news/...")

if st.button("Generate Summary"):
    if url:
        start_time = time.time()
        status_text = st.empty()
        status_text.info("🔍 Initializing Universal Stealth Browser...")
        
        try:
            # 4. UNIVERSAL SELENIUM BACKEND
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            chrome_options.add_argument(f"user-agent={user_agent}")

            driver = webdriver.Chrome(options=chrome_options)
            
            status_text.info("🛡️ Bypassing Security & Extracting Content...")
            driver.get(url)
            
            # Universal Wait & Scroll
            time.sleep(6) 
            driver.execute_script("window.scrollTo(0, 1000);")
            time.sleep(2)
            
            # Use Newspaper3k to parse the page source from Selenium
            page_source = driver.page_source
            article = Article(url)
            article.set_html(page_source)
            article.parse()
            
            raw_text = article.text
            article_title = article.title if article.title else "News Update"
            driver.quit()
            
            if len(raw_text) < 250:
                st.error("Extraction failed. This site might have strong bot protection or the content is behind a login.")
            else:
                status_text.info("🧠 AI Analysis in progress...")
                
                # 5. NATIVE AI LOGIC
                input_prompt = f"summarize this news in 10 clear sentences: {raw_text[:1200]}"
                inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=512)
                
                outputs = model.generate(
                    inputs["input_ids"], 
                    max_length=200, 
                    min_length=80, 
                    length_penalty=2.0, 
                    num_beams=4, 
                    early_stopping=True
                )
                
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                formatted_summary = summary.replace(". ", ".\n\n")
                
                # Add to History
                history_label = f"{article_title[:40]}..."