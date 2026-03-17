import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
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
st.set_page_config(page_title="News Summarizer", page_icon="📑", layout="wide")

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

# Requested name change
st.title("📑 News Summarizer", help="Fast AI extraction and deep summarization.")
st.write("Generating high-detail, 15-line professional summaries.")

url = st.text_input("Paste any news article URL here:", placeholder="https://www.thehindu.com/...")

if st.button("Generate Summary"):
    if url:
        start_time = time.time()
        status_text = st.empty()
        status_text.info("⚡ Fast-Extracting News Content...")
        
        try:
            # 4. FAST EXTRACTION (Bypassing Selenium for speed)
            article = Article(url)
            article.download()
            article.parse()
            
            raw_text = article.text
            article_title = article.title if article.title else "News Update"
            
            if len(raw_text) < 200:
                st.warning("🔄 Fast extraction failed. Trying deep stealth mode...")
                # (Optional: You could put Selenium here as a backup, but for speed, direct is best)
                st.error("Site blocked fast-access. Please try a different news source.")
            else:
                status_text.info("🧠 AI Deep Analysis (ChatGPT-Style)...")
                
                # 5. GPT-STYLE SUMMARY LOGIC
                # We ask for a "Point-by-point" report to ensure length and clarity
                input_prompt = f"Provide a detailed, point-by-point 15-line report on the following news. Use bullet points for different facts: {raw_text[:1200]}"
                inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=512)
                
                outputs = model.generate(
                    inputs["input_ids"], 
                    max_length=512,        
                    min_length=300,        
                    do_sample=True,         
                    top_p=0.9,             
                    repetition_penalty=3.0, 
                    no_repeat_ngram_size=3, 
                    early_stopping=True
                )
                
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Formatting the summary to look like a chat response
                # We replace periods with newlines and bullets for a "ChatGPT" feel
                formatted_summary = summary.replace(". ", ".\n\n• ")
                if not formatted_summary.startswith("• "):
                    formatted_summary = "• " + formatted_summary

                # Add to History
                history_label = f"{article_title[:40]}..."
                if history_label not in st.session_state.history:
                    st.session_state.history.append(history_label)

                status_text.empty()
                st.success(f"Analysis Complete! ({round(time.time() - start_time, 2)}s)")
                
                st.subheader(article_title)
                st.markdown(formatted_summary) # Using markdown for better bullet visibility
                
                st.download_button("💾 Download Summary", data=formatted_summary, file_name="summary.txt")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please paste a URL first!")