import time
import streamlit as st

from extractor import extract_article
from ai import generate_summary


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide"
)


# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("📰 News Summarizer")
    st.markdown("---")
    st.subheader("Summary History")

    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**{i + 1}. {item['title'][:50]}**")
    else:
        st.info("No summaries generated yet.")

    st.markdown("---")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# =====================================================
# MAIN HEADER
# =====================================================

st.title("📰 News Summarizer")
st.markdown("Enter a news article URL or paste the raw article text to generate a concise summary.")


# =====================================================
# ARTICLE INPUT TABS
# =====================================================

st.subheader("📥 Article Input")

tab_url, tab_text = st.tabs(["🔗 Summarize via URL", "📝 Paste Article Text"])

with tab_url:
    with st.form("news_url_form"):
        url_input = st.text_input(
            "News Article URL",
            placeholder="https://indianexpress.com/article/... or https://reuters.com/...",
            label_visibility="collapsed"
        )
        submit_url = st.form_submit_button("🚀 Summarize from URL", use_container_width=True)

with tab_text:
    with st.form("news_raw_text_form"):
        raw_text_input = st.text_area(
            "Paste full article text here",
            height=200,
            placeholder="Paste article body here (especially for subscriber-only or paywalled articles)...",
            label_visibility="collapsed"
        )
        submit_text = st.form_submit_button("🚀 Summarize Pasted Text", use_container_width=True)


# =====================================================
# PROCESS ARTICLE
# =====================================================

article_text = None
source_title = ""

# --- Flow 1: URL Submission ---
if submit_url:
    cleaned_url = url_input.strip()
    if not cleaned_url:
        st.warning("⚠️ Please paste a valid news article URL.")
        st.stop()

    extraction_start = time.perf_counter()
    with st.spinner("🔎 Extracting article content..."):
        try:
            article_text = extract_article(cleaned_url)
        except Exception:
            article_text = None

    extraction_time = time.perf_counter() - extraction_start

    if not article_text:
        st.error("❌ Unable to extract the article from this website.")
        st.info(
            "The publisher is likely blocking scraping requests (403 Forbidden or Bot Wall). "
            "💡 **Workaround:** Copy the text from your browser and switch to the **'📝 Paste Article Text'** tab above."
        )
        st.stop()

    st.success(f"✓ Article extracted successfully in {extraction_time:.2f} seconds")
    source_title = cleaned_url

# --- Flow 2: Raw Text Submission ---
elif submit_text:
    cleaned_text = raw_text_input.strip()
    if not cleaned_text:
        st.warning("⚠️ Please paste some article text to summarize.")
        st.stop()

    article_text = cleaned_text
    source_title = "Pasted Text Article"


# =====================================================
# GENERATION & OUTPUT
# =====================================================

if article_text:
    # -------------------------------------------------
    # Check Minimum Length
    # -------------------------------------------------
    if len(article_text.strip()) < 300:
        st.error("❌ The article text is too short (less than 300 characters) to generate a reliable summary.")
        st.stop()

    # -------------------------------------------------
    # Generate Summary via Groq
    # -------------------------------------------------
    st.subheader("✨ Summary")
    summary_placeholder = st.empty()
    complete_summary = ""
    summary_start = time.perf_counter()

    try:
        with st.spinner("🤖 Generating summary with Groq..."):
            for chunk in generate_summary(article_text):
                complete_summary += chunk
                summary_placeholder.markdown(complete_summary)

        summary_time = time.perf_counter() - summary_start
        st.success(f"✓ Summary generated in {summary_time:.2f} seconds")

    except Exception as e:
        st.error("❌ Failed to generate the summary.")
        st.caption(f"Error: {e}")
        st.info("Please verify your `GROQ_API_KEY` in your Streamlit secrets settings.")
        st.stop()

    # -------------------------------------------------
    # Download Button & History
    # -------------------------------------------------
    if complete_summary.strip():
        st.download_button(
            label="⬇️ Download Summary",
            data=complete_summary,
            file_name="news_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.session_state.history.append({
            "title": complete_summary[:80],
            "url": source_title
        })