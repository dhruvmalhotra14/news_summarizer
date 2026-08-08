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
    "Enter a news article URL and generate an AI-powered summary."
)


# =====================================================
# URL INPUT
# =====================================================

st.subheader("🔗 Article Input")

url = st.text_input(
    "Paste News Article URL",
    placeholder="https://example.com/news/article",
    label_visibility="visible"
)


# =====================================================
# GENERATE BUTTON
# =====================================================

generate_button = st.button(
    "🚀 Generate Summary",
    use_container_width=True
)


# =====================================================
# PROCESS ARTICLE
# =====================================================

if generate_button:

    if not url.strip():

        st.warning(
            "⚠️ Please paste a news article URL."
        )

        st.stop()

    # -------------------------------------------------
    # Extract article
    # -------------------------------------------------

    with st.spinner(
        "🔎 Extracting article..."
    ):

        article_text = extract_article(url)

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
    # Check article length
    # -------------------------------------------------

    if len(article_text.strip()) < 300:

        st.error(
            "❌ The extracted article text is too short "
            "to generate a reliable summary."
        )

        st.stop()

    # -------------------------------------------------
    # Generate summary
    # -------------------------------------------------

    st.subheader("✨ AI Summary")

    summary_placeholder = st.empty()

    complete_summary = ""

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

    except Exception as e:

        st.error(
            "❌ Failed to generate the summary."
        )

        st.caption(
            "Please check your Gemini API configuration."
        )

        st.stop()

    # -------------------------------------------------
    # Save history
    # -------------------------------------------------

    if complete_summary.strip():

        st.session_state.history.append(
            {
                "title": complete_summary[:80],
                "url": url
            }
        )

    # -------------------------------------------------
    # Powered by Gemini
    # -------------------------------------------------

    st.caption("✨ Powered by Gemini")