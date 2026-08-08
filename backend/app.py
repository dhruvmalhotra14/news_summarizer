import time
import streamlit as st

from extractor import extract_article
from ai import generate_summary


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# Session State
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("📰 News Summarizer")

    st.markdown("---")

    st.subheader("Summary History")

    if not st.session_state.history:

        st.info(
            "No summaries generated yet."
        )

    else:

        for item in reversed(
            st.session_state.history
        ):

            st.write(
                "•",
                item
            )

    st.markdown("---")

    if st.button(
        "🗑 Clear History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# Main Page
# ============================================================

st.title("📰 News Summarizer")

st.caption(
    "Generate concise summaries from news articles using Gemini AI."
)


# ============================================================
# URL Input
# ============================================================

with st.form("summary_form"):

    st.subheader("Article Input")

    url = st.text_input(
        "🔗 Paste News Article URL",
        placeholder="https://example.com/news/article"
    )

    generate = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )


# ============================================================
# Generate Summary
# ============================================================

if generate:

    # --------------------------------------------------------
    # Validate URL
    # --------------------------------------------------------

    if not url.strip():

        st.warning(
            "Please enter a news article URL."
        )

        st.stop()


    # --------------------------------------------------------
    # Extract Article
    # --------------------------------------------------------

    extraction_start = time.time()

    try:

        with st.spinner(
            "Extracting article..."
        ):

            title, article_text = extract_article(
                url
            )

    except Exception as e:

        st.error(
            f"❌ {str(e)}"
        )

        st.stop()


    extraction_time = round(
        time.time() - extraction_start,
        2
    )


    # --------------------------------------------------------
    # Validate extracted article
    # --------------------------------------------------------

    if not article_text:

        st.error(
            "❌ No article content could be extracted."
        )

        st.stop()


    if len(article_text.strip()) < 100:

        st.error(
            "❌ The extracted article is too short."
        )

        st.stop()


    # --------------------------------------------------------
    # Add to History
    # --------------------------------------------------------

    if title not in st.session_state.history:

        st.session_state.history.append(
            title
        )


    # --------------------------------------------------------
    # Article Information
    # --------------------------------------------------------

    st.subheader(title)

    st.caption(
        f"Article extracted in "
        f"{extraction_time} seconds"
    )


    # --------------------------------------------------------
    # Generate Gemini Summary
    # --------------------------------------------------------

    summary_start = time.time()

    summary_placeholder = st.empty()

    summary = ""


    try:

        with st.spinner(
            "Generating summary..."
        ):

            for chunk in generate_summary(
                article_text
            ):

                summary += chunk

                # Streaming output
                summary_placeholder.markdown(
                    summary + "▌"
                )


        # Remove streaming cursor
        summary_placeholder.markdown(
            summary
        )


    except Exception as e:

        summary_placeholder.empty()

        st.error(
            f"❌ Error generating summary: {str(e)}"
        )

        st.stop()


    # --------------------------------------------------------
    # Generation Time
    # --------------------------------------------------------

    summary_time = round(
        time.time() - summary_start,
        2
    )


    st.success(
        f"✅ Summary generated in "
        f"{summary_time} seconds"
    )


    # --------------------------------------------------------
    # Powered by Gemini
    # --------------------------------------------------------

    st.caption(
        "✨ Powered by Gemini"
    )


    # --------------------------------------------------------
    # Download Summary
    # --------------------------------------------------------

    st.download_button(
        label="📥 Download Summary",
        data=summary,
        file_name="news_summary.txt",
        mime="text/plain",
        use_container_width=True
    )