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

if "latest_summary" not in st.session_state:
    st.session_state.latest_summary = ""

if "latest_time" not in st.session_state:
    st.session_state.latest_time = None


# =====================================================
# SIDEBAR (HISTORY)
# =====================================================

with st.sidebar:
    st.title("📰 News Summarizer")
    st.markdown("---")
    st.subheader("Summary History")

    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            # Clean headline fallback for history
            display_title = item["title"].replace("*", "").replace("-", "").strip()
            if not display_title:
                display_title = item["url"]
            st.markdown(f"**{i + 1}.** [{display_title[:45]}...]({item['url']})")
    else:
        st.info("No summaries generated yet.")

    st.markdown("---")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.latest_summary = ""
        st.session_state.latest_time = None
        st.rerun()


# =====================================================
# MAIN HEADER
# =====================================================

st.title("📰 News Summarizer")
st.markdown("Enter a news article URL and generate a concise summary.")


# =====================================================
# ARTICLE INPUT
# =====================================================

st.subheader("🔗 Article Input")

with st.form("news_summary_form"):
    url = st.text_input(
        "Paste News Article URL",
        placeholder="https://example.com/news/article",
        label_visibility="visible"
    )

    submitted = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )


# =====================================================
# PROCESS ARTICLE
# =====================================================

if submitted:
    cleaned_url = url.strip()

    if not cleaned_url:
        st.warning("⚠️ Please paste a news article URL.")
        st.stop()

    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = "https://" + cleaned_url

    # 1. Extraction
    extraction_start = time.perf_counter()
    with st.spinner("🔎 Extracting article..."):
        try:
            article_text = extract_article(cleaned_url)
        except Exception:
            article_text = None

    extraction_time = time.perf_counter() - extraction_start

    if not article_text:
        st.error("❌ Unable to extract the article from this website.")
        st.info("The publisher may be blocking automated scrapers. Please try another link.")
        st.stop()

    if len(article_text.strip()) < 300:
        st.error("❌ The extracted article text is too short to generate a reliable summary.")
        st.stop()

    st.success(f"✓ Article extracted successfully in {extraction_time:.2f} seconds")

    # 2. Summary Generation
    st.subheader("✨ Summary")
    summary_placeholder = st.empty()
    complete_summary = ""
    summary_start = time.perf_counter()

    try:
        with st.spinner("🤖 Generating summary..."):
            for chunk in generate_summary(article_text):
                complete_summary += chunk
                summary_placeholder.markdown(complete_summary)

        summary_time = time.perf_counter() - summary_start

        # Store in session state
        st.session_state.latest_summary = complete_summary
        st.session_state.latest_time = summary_time

        # Save to history immediately
        st.session_state.history.append({
            "title": complete_summary.strip().split("\n")[0][:60],
            "url": cleaned_url
        })

        # Instant rerun so the sidebar history refreshes immediately
        st.rerun()

    except Exception as e:
        st.error("❌ Failed to generate the summary.")
        st.caption(f"Error: {e}")
        st.info("Please verify your GROQ_API_KEY inside `.streamlit/secrets.toml`.")
        st.stop()


# =====================================================
# DISPLAY CURRENT / LAST GENERATED SUMMARY
# =====================================================

if st.session_state.latest_summary and not submitted:
    st.subheader("✨ Summary")
    st.markdown(st.session_state.latest_summary)

    if st.session_state.latest_time:
        st.success(f"✓ Summary generated in {st.session_state.latest_time:.2f} seconds")

    st.download_button(
        label="⬇️ Download Summary",
        data=st.session_state.latest_summary,
        file_name="news_summary.txt",
        mime="text/plain",
        use_container_width=True
    )