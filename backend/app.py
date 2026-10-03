import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import time

import streamlit as st

from backend.extractor import extract_article
from backend.ai import generate_summary
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "current_summary" not in st.session_state:
    st.session_state.current_summary = ""

if "current_url" not in st.session_state:
    st.session_state.current_url = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📰 News Summarizer")

    st.markdown("---")

    st.subheader("Summary History")

    if st.session_state.history:

        for i, item in enumerate(
            reversed(st.session_state.history)
        ):

            title = (
                item.get("title")
                or item.get("url", "Article")
            )

            st.markdown(
                f"**{i + 1}.** {title}"
            )

    else:

        st.info(
            "No summaries generated yet."
        )

    st.markdown("---")

    if st.button(
        "🗑️ Clear History",
        use_container_width=True
    ):

        st.session_state.history = []
        st.session_state.current_summary = ""
        st.session_state.current_url = ""

        st.rerun()


# ============================================================
# MAIN PAGE
# ============================================================

st.title("📰 News Summarizer")

st.markdown(
    "Enter a news article URL and generate a concise AI summary."
)


# ============================================================
# ARTICLE INPUT
# ============================================================

st.subheader("🔗 Article Input")


with st.form(
    "news_summary_form",
    clear_on_submit=False
):

    url_input = st.text_input(
        "Paste News Article URL",
        value=st.session_state.current_url,
        placeholder="https://example.com/news/article"
    )

    submitted = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )


# ============================================================
# GENERATE SUMMARY
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Clean URL
    # --------------------------------------------------------

    cleaned_url = url_input.strip()

    # --------------------------------------------------------
    # Validate URL
    # --------------------------------------------------------

    if not cleaned_url:

        st.warning(
            "⚠️ Please paste a news article URL."
        )

        st.stop()

    # --------------------------------------------------------
    # Add HTTPS if missing
    # --------------------------------------------------------

    if not cleaned_url.startswith(
        ("http://", "https://")
    ):

        cleaned_url = "https://" + cleaned_url

    # Save URL

    st.session_state.current_url = cleaned_url

    # --------------------------------------------------------
    # Extract Article
    # --------------------------------------------------------

    extraction_start = time.perf_counter()

    with st.spinner(
        "🔎 Extracting article..."
    ):

        try:

            article_text = extract_article(
                cleaned_url
            )

        except Exception as e:

            article_text = ""

            st.error(
                "❌ Extraction error."
            )

            st.exception(e)

    extraction_time = (
        time.perf_counter()
        - extraction_start
    )

    # --------------------------------------------------------
    # Debug: show extracted character count
    # --------------------------------------------------------

    extracted_length = len(
        article_text or ""
    )

    st.write(
        f"Extracted characters: {extracted_length}"
    )

    # --------------------------------------------------------
    # Extraction Failed
    # --------------------------------------------------------

    if not article_text:

        st.error(
            "❌ Unable to extract this article."
        )

        st.info(
            "The website returned no usable article text."
        )

        st.stop()

    # --------------------------------------------------------
    # Clean extracted text
    # --------------------------------------------------------

    article_text = article_text.strip()

    extracted_length = len(
        article_text
    )

    # --------------------------------------------------------
    # Article Too Short
    # --------------------------------------------------------

    if extracted_length < 300:

        st.error(
            "❌ The extracted article text is too short "
            "to generate a reliable summary."
        )

        st.write(
            f"Extracted characters: {extracted_length}"
        )

        st.stop()

    # --------------------------------------------------------
    # Extraction Successful
    # --------------------------------------------------------

    st.success(
        f"✓ Article extracted successfully "
        f"in {extraction_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Generate AI Summary
    # --------------------------------------------------------

    st.subheader("✨ Summary")

    summary_placeholder = st.empty()

    complete_summary = ""

    summary_start = time.perf_counter()

    try:

        with st.spinner(
            "🤖 Generating summary..."
        ):

            for chunk in generate_summary(
                article_text
            ):

                complete_summary += chunk

                summary_placeholder.markdown(
                    complete_summary
                )

        summary_time = (
            time.perf_counter()
            - summary_start
        )

        # ----------------------------------------------------
        # Check Summary
        # ----------------------------------------------------

        if not complete_summary.strip():

            st.error(
                "❌ The AI model returned an empty summary."
            )

            st.stop()

        # ----------------------------------------------------
        # Summary Generated
        # ----------------------------------------------------

        st.success(
            f"✓ Summary generated in "
            f"{summary_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Create History Title
        # ----------------------------------------------------

        first_line = (
            complete_summary
            .strip()
            .split("\n")[0]
            .replace("*", "")
            .replace("#", "")
            .strip()
        )

        headline = (
            first_line
            if first_line
            else cleaned_url
        )

        # ----------------------------------------------------
        # Save Summary to History
        # ----------------------------------------------------

        st.session_state.history.append(
            {
                "title": headline,
                "url": cleaned_url,
                "summary": complete_summary
            }
        )

        # ----------------------------------------------------
        # Save Current Summary
        # ----------------------------------------------------

        st.session_state.current_summary = (
            complete_summary
        )

        # ----------------------------------------------------
        # Refresh Page
        # ----------------------------------------------------

        st.rerun()

    except Exception as e:

        st.error(
            "❌ Failed to generate the summary."
        )

        st.exception(e)

        st.info(
            "Please verify your GROQ_API_KEY "
            "inside `.streamlit/secrets.toml`."
        )

        st.stop()


# ============================================================
# DISPLAY EXISTING SUMMARY
# ============================================================

if (
    st.session_state.current_summary
    and not submitted
):

    st.subheader("✨ Summary")

    st.markdown(
        st.session_state.current_summary
    )

    # --------------------------------------------------------
    # Download Summary
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download Summary",
        data=st.session_state.current_summary,
        file_name="news_summary.txt",
        mime="text/plain",
        use_container_width=True
    )