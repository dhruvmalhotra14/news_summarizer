import streamlit as st
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

# 1. LOAD THE AI MODEL
@st.cache_resource
def load_summarizer():
    return pipeline("text-generation", model="google/flan-t5-small")

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
st.write("Generating high-detail, long-form summaries (10-15 lines).")

url = st.text_input("Enter Article URL here:", placeholder="https://www.ndtv.com/...")

if st.button("Generate Detailed Summary"):
    if url:
        start_time = time.time()
        status_text = st.empty()
        status_text.info("🔍 Initializing Stealth Browser...")
        
        try:
            # 4. STEALTH SELENIUM BACKEND (CLOUD OPTIMIZED)
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")  # Required for Cloud
            chrome_options.add_argument("--disable-dev-shm-usage")  # Required for Cloud
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--blink-settings=imagesEnabled=false")
            
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            chrome_options.add_argument(f"user-agent={user_agent}")

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            status_text.info("🛡️ Bypassing Security & Extracting...")
            driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {'headers': {'Referer': 'https://www.google.com/'}})
            
            driver.get(url)
            time.sleep(4) 
            
            paragraphs = driver.find_elements(By.TAG_NAME, "p")
            raw_text = " ".join([p.text for p in paragraphs if len(p.text) > 40])
            article_title = driver.title if driver.title else "News Update"
            driver.quit()
            
            if len(raw_text) < 150:
                st.error("Text not found. The site might be blocking extraction.")
            else:
                status_text.info("🧠 AI Deep Analysis in progress...")
                
                # 5. AI LOGIC (Reliable Long-Form Mode)
                input_text = f"Write a long, 15-line detailed summary for this news: {raw_text[:1000]}"
                
                result = summarizer(
                    input_text, 
                    max_new_tokens=512, 
                    min_new_tokens=220, 
                    do_sample=False,
                    repetition_penalty=2.5
                )
                
                summary = result[0]['generated_text'].split("news:")[-1].strip()

                # Cleanup & Professional Formatting
                summary = summary.replace("U.S.\n", "U.S. ").replace("pic.", "")
                
                if not summary.endswith((".", "!", "?")):
                    if "." in summary:
                        summary = summary.rsplit(".", 1)[0] + "."

                # Smart Category Logic
                category = "General"
                if any(x in summary.lower() for x in ["strike", "military", "forces", "targets"]): category = "🛡️ Defense"
                elif any(x in summary.lower() for x in ["oil", "economy", "trade", "global"]): category = "💰 Economy"
                elif any(x in summary.lower() for x in ["election", "minister", "government"]): category = "🏛️ Politics"

                formatted_summary = summary.replace(". ", ".\n\n")
                
                # Add to History
                history_label = f"[{category}] {article_title[:40]}..."
                if history_label not in st.session_state.history:
                    st.session_state.history.append(history_label)

                status_text.empty()
                st.success(f"Analysis Complete! ({round(time.time() - start_time, 2)}s)")
                
                st.subheader(f"{category} | {article_title}")
                st.markdown(f"**Stats:** {len(formatted_summary.split())} words | ~{len(formatted_summary.splitlines())} lines")
                st.info(formatted_summary)
                
                st.download_button("💾 Download Summary", data=formatted_summary, file_name="summary.txt")
                st.text_area("Copy summary:", value=formatted_summary, height=350)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please paste a URL first!")