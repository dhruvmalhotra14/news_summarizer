import time
import streamlit as st

from extractor import extract_article
from ai import generate_summary


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide"
)


# -------------------------------------------------
# Session State
# -------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

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


# -------------------------------------------------
# Main UI
# -------------------------------------------------

st.title("📰 News Summarizer")

st.caption(
    "Summarize news articles using AI"
)


# -------------------------------------------------
# Input Form
# -------------------------------------------------

with st.form("summary_form"):

    st.subheader("Article Input")

    url = st.text_input(
        "🔗 Paste News Article URL",
        placeholder="https://example.com/news/article"
    )

    st.markdown(
        "**If the website blocks extraction, paste the article text below:**"
    )

    article_text = st.text_area(
        "📄 Article Text (Optional)",
        placeholder=(
            "Paste the article text here if URL extraction "
            "is blocked..."
        ),
        height=180
    )

    generate = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )


# -------------------------------------------------
# Generate Summary
# -------------------------------------------------

if generate:

    # ---------------------------------------------
    # Check Input
    # ---------------------------------------------

    if not url.strip() and not article_text.strip():

        st.warning(
            "Please enter a news article URL or paste article text."
        )

        st.stop()


    try:

        # -----------------------------------------
        # Extract Article
        # -----------------------------------------

        if url.strip():

            start = time.time()

            with st.spinner("Extracting article..."):

                try:

                    title, article = extract_article(url)

                except Exception as extraction_error:

                    # If URL extraction fails but user
                    # provided article text, use it.

                    if article_text.strip():

                        title = "Pasted Article"

                        article = article_text.strip()

                        st.info(
                            "URL extraction was blocked. "
                            "Using the pasted article text."
                        )

                    else:

                        raise extraction_error

            extract_time = round(
                time.time() - start,
                2
            )

        else:

            title = "Pasted Article"

            article = article_text.strip()

            extract_time = 0


        # -----------------------------------------
        # Validate Article
        # -----------------------------------------

        if len(article.strip()) < 100:

            st.warning(
                "The article text is too short to generate "
                "a meaningful summary."
            )

            st.stop()


        # -----------------------------------------
        # History
        # -----------------------------------------

        if title not in st.session_state.history:

            st.session_state.history.append(title)


        # -----------------------------------------
        # Display Title
        # -----------------------------------------

        st.subheader(title)


        if extract_time > 0:

            st.caption(
                f"Article extracted in {extract_time} seconds"
            )


        # -----------------------------------------
        # Generate AI Summary
        # -----------------------------------------

        start = time.time()

        placeholder = st.empty()

        summary = ""


        with st.spinner("Generating summary..."):

            for chunk in generate_summary(article):

                summary += chunk

                # ChatGPT-style streaming
                placeholder.markdown(
                    summary + "▌"
                )


        # Remove cursor
        placeholder.markdown(summary)


        ai_time = round(
            time.time() - start,
            2
        )


        st.success(
            f"✅ Summary generated in {ai_time} seconds"
        )


        # -----------------------------------------
        # Download
        # -----------------------------------------

        st.download_button(
            "📥 Download Summary",
            summary,
            file_name="news_summary.txt",
            mime="text/plain"
        )


    except Exception as e:

        st.error(
            f"❌ {str(e)}"
        )