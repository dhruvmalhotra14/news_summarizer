import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from newspaper import Article
import time

# 1. HYPER-FAST MODEL LOADING
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "google/flan-t5-small"
    # Using 'low_cpu_mem_usage' to speed up the initial boot
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# 2. SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []

# 3. CLEAN UI DESIGN
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

# Updated Title per your request
st.title("📑 News Summarizer", help="Optimized for high-speed AI analysis.")
st.write("Instant, high-detail 15-line reports.")

url = st.text_input("Paste any news article URL here:", placeholder="https://www.thehindu.com/...")

if st.button("Generate Summary"):
    if url:
        start_time = time.time()
        status_text = st.empty()
        status_text.info("⚡ Rapid Extraction Active...")
        
        try:
            # 4. LIGHTWEIGHT EXTRACTION (Turbo Mode)
            # We use a custom config to bypass heavy media files
            article = Article(url, keep_article_html=False, request_timeout=10)
            article.download()
            article.parse()
            
            raw_text = article.text
            article_title = article.title if article.title else "News Update"
            
            if len(raw_text) < 150:
                st.error("Extraction failed. The site may be blocking rapid access.")
            else:
                status_text.info("🧠 Performing Deep Semantic Analysis...")
                
                # 5. OPTIMIZED AI LOGIC (Point-by-Point / No Repeats)
                input_prompt = f"Provide a detailed, 15-line professional point-by-point report on this: {raw_text[:1200]}"
                
                # Tokenize on CPU but optimize for inference
                inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad(): # Speeds up inference by not calculating gradients
                    outputs = model.generate(
                        inputs["input_ids"], 
                        max_length=450,        
                        min_length=280,        
                        do_sample=True,         
                        top_p=0.92,             
                        repetition_penalty=3.5, # Increased to keep it moving fast and unique
                        no_repeat_ngram_size=3, 
                        early_stopping=True
                    )
                
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Format output with clean bullets for the modern look
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
                st.markdown(formatted_summary) 
                
                st.download_button("💾 Download Report", data=formatted_summary, file_name="report.txt")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please paste a URL first!")