import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from newspaper import Article
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
st.set_page_config(page_title="Universal Deep AI Summarizer", page_icon="🌍", layout="wide")

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

st.title("🌍 Universal Deep AI Summarizer")
st.write("Extracting high-detail, 12-15 line summaries from any news site.")

url = st.text_input("Paste any news article URL here:", placeholder="https://www.thehindu.com/...")

if st.button("Generate Deep Summary"):
    if url:
        start_time = time.time()
        status_text = st.empty()
        status_text.info("🔍 Initializing Universal Stealth Browser...")
        
        try:
            # 4. STEALTH SELENIUM BACKEND (Stability Fixes for Cloud)
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--remote-debugging-port=9222")
            chrome_options.add_argument("--window-size=1920,1080")
            
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            chrome_options.add_argument(f"user-agent={user_agent}")

            driver = webdriver.Chrome(options=chrome_options)
            
            status_text.info("🛡️ Bypassing Security & Extracting Content...")
            driver.get(url)
            
            # Universal Wait & Scroll
            time.sleep(8) 
            driver.execute_script("window.scrollTo(0, 1000);")
            time.sleep(2)
            
            # Use Newspaper3k logic to find the main article body
            page_source = driver.page_source
            article = Article(url)
            article.set_html(page_source)
            article.parse()
            
            raw_text = article.text
            article_title = article.title if article.title else "News Update"
            driver.quit()
            
            if len(raw_text) < 250:
                st.error("Extraction failed. Site might be blocking bots or the content is too short.")
            else:
                status_text.info("🧠 AI Deep Analysis in progress (Generating 15 lines)...")
                
                # 5. DEEP AI LOGIC (No Pipeline - Fixes KeyError)
                input_prompt = f"Write a very long, detailed 15-line professional summary of this news: {raw_text[:1500]}"
                inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=512)
                
                # Parameters tuned to force length and detail
                outputs = model.generate(
                    inputs["input_ids"], 
                    max_length=512,        
                    min_length=300,        # FORCES a long response
                    length_penalty=2.5,    
                    num_beams=5, 
                    repetition_penalty=1.8, 
                    early_stopping=True
                )
                
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                formatted_summary = summary.replace(". ", ".\n\n")
                
                # Add to History
                history_label = f"{article_title[:40]}..."
                if history_label not in st.session_state.history:
                    st.session_state.history.append(history_label)

                status_text.empty()
                st.success(f"Analysis Complete! ({round(time.time() - start_time, 2)}s)")
                
                st.subheader(article_title)
                st.info(formatted_summary)
                
                st.download_button("💾 Download Detailed Summary", data=formatted_summary, file_name="summary.txt")

        except Exception as e:
            st.error(f"Error: {e}")
            if 'driver' in locals():
                driver.quit()
    else:
        st.warning("Please paste a URL first!")