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

        for i, item in enumerate(
            reversed(st.session_state.history)
        ):

            st.markdown(
                f"**{i + 1}. {item['title'][:50]}**"
            )

    else:

        st.info("No summaries generated yet.")

    st.markdown("---")

    if st.button(
        "🗑️ Clear History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()


# =====================================================
# MAIN HEADER
# =====================================================

st.title("📰 News Summarizer")

st.markdown(
    "Paste a news article URL and generate a concise summary."
)


# =====================================================
# FORM
# =====================================================

with st.form("news_form"):

    url = st.text_input(
        "Paste News Article URL",
        placeholder="https://example.com/news/article"
    )

    generate_button = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )


# =====================================================
# PROCESS ARTICLE
# =====================================================

if generate_button:

    # -------------------------------------------------
    # CHECK URL
    # -------------------------------------------------

    if not url.strip():

        st.warning(
            "⚠️ Please paste a news article URL."
        )

        st.stop()


    # -------------------------------------------------
    # EXTRACT ARTICLE
    # -------------------------------------------------

    extraction_start = time.perf_counter()

    with st.spinner(
        "🔎 Extracting article..."
    ):

        article_text = extract_article(url)

    extraction_time = (
        time.perf_counter()
        - extraction_start
    )


    # -------------------------------------------------
    # EXTRACTION FAILED
    # -------------------------------------------------

    if not article_text:

        st.error(
            "❌ Unable to extract the article from this website."
        )

        st.info(
            "The publisher may be blocking automated access "
            "or the page may not contain accessible article text."
        )

        st.stop()


    # -------------------------------------------------
    # CHECK ARTICLE LENGTH
    # -------------------------------------------------

    if len(article_text.strip()) < 300:

        st.error(
            "❌ The extracted article text is too short "
            "to generate a reliable summary."
        )

        st.stop()


    # -------------------------------------------------
    # SHOW EXTRACTION TIME
    # -------------------------------------------------

    st.success(
        f"✓ Article extracted in {extraction_time:.2f} seconds"
    )


    # -------------------------------------------------
    # GENERATE SUMMARY
    # -------------------------------------------------

    st.subheader("✨ Summary")

    summary_start = time.perf_counter()

    try:

        with st.spinner(
            "🤖 Generating summary..."
        ):

            complete_summary = generate_summary(
                article_text
            )

        summary_time = (
            time.perf_counter()
            - summary_start
        )

        st.markdown(
            complete_summary
        )

        st.success(
            f"✓ Summary generated in {summary_time:.2f} seconds"
        )


    # -------------------------------------------------
    # ERROR
    # -------------------------------------------------

    except Exception as e:

        st.error(
            "❌ Failed to generate the summary."
        )

        st.caption(
            "Please check your Gemini API configuration."
        )

        st.stop()


    # -------------------------------------------------
    # DOWNLOAD SUMMARY
    # -------------------------------------------------

    st.download_button(
        label="⬇️ Download Summary",
        data=complete_summary,
        file_name="news_summary.txt",
        mime="text/plain",
        use_container_width=True
    )


    # -------------------------------------------------
    # SAVE HISTORY
    # -------------------------------------------------

    if complete_summary.strip():

        st.session_state.history.append(
            {
                "title": complete_summary[:80],
                "url": url
            }
        )