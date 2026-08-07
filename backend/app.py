import time
import streamlit as st

from extractor import extract_article
from ai import generate_summary

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide"
)

# ----------------------------
# Session State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:

    st.title("📰 News Summarizer")

    st.markdown("---")

    st.subheader("Summary History")

    if not st.session_state.history:
        st.info("No summaries generated yet.")
    else:
        for item in reversed(st.session_state.history):
            st.write("•", item)

    st.markdown("---")

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.rerun()

# ----------------------------
# Main UI
# ----------------------------
st.title("📰 News Summarizer")

with st.form("summary_form"):

    url = st.text_input(
        "Paste News Article URL",
        placeholder="https://..."
    )

    generate = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )

# ----------------------------
# Generate Summary
# ----------------------------
if generate:

    if not url.strip():
        st.warning("Please enter a news article URL.")
        st.stop()

    try:

        # ----------------------------
        # Extract Article
        # ----------------------------
        start = time.time()

        with st.spinner("Extracting article..."):
            title, article = extract_article(url)

        extract_time = round(time.time() - start, 2)

        st.success(f"✅ Article extracted in {extract_time} sec")

        if title not in st.session_state.history:
            st.session_state.history.append(title)

        st.subheader(title)

        # ----------------------------
        # Generate Summary
        # ----------------------------
        start = time.time()

        placeholder = st.empty()
        summary = ""

        with st.spinner("Generating Summary..."):

            for chunk in generate_summary(article):
                summary += chunk
                placeholder.markdown(summary + "▌")

        # Remove cursor when finished
        placeholder.markdown(summary)

        ai_time = round(time.time() - start, 2)

        st.success(f"🤖 Summary generated in {ai_time} sec")

        # ----------------------------
        # Download Button
        # ----------------------------
        st.download_button(
            "📥 Download Summary",
            data=summary,
            file_name="news_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ {e}")