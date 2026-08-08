import streamlit as st

from extractor import extract_article
from ai import generate_summary


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📰 News Summarizer")

    st.divider()

    st.subheader("Summary History")

    if st.session_state.history:

        for i, item in enumerate(
            st.session_state.history,
            start=1
        ):

            st.markdown(
                f"**{i}.** {item['title']}"
            )

    else:

        st.info("No summaries generated yet.")

    st.divider()

    if st.button(
        "🗑️ Clear History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()


# =========================================================
# MAIN PAGE
# =========================================================

st.title("📰 News Summarizer")

st.write(
    "Paste a news article URL below and generate a concise summary."
)


# =========================================================
# URL INPUT FORM
# =========================================================

with st.form("news_form"):

    url = st.text_input(
        "🔗 Paste News Article URL",
        placeholder="https://..."
    )

    generate_button = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )


# =========================================================
# GENERATE SUMMARY
# =========================================================

if generate_button:

    # -----------------------------------------------------
    # CHECK URL
    # -----------------------------------------------------

    if not url.strip():

        st.warning(
            "⚠️ Please enter a news article URL."
        )

        st.stop()


    # -----------------------------------------------------
    # CHECK URL FORMAT
    # -----------------------------------------------------

    if not (
        url.startswith("http://")
        or url.startswith("https://")
    ):

        st.error(
            "❌ Please enter a valid article URL."
        )

        st.stop()


    # -----------------------------------------------------
    # EXTRACT ARTICLE
    # -----------------------------------------------------

    with st.spinner("🔎 Extracting article..."):

        article_text = extract_article(url)


    # -----------------------------------------------------
    # EXTRACTION FAILED
    # -----------------------------------------------------

    if not article_text:

        st.error(
            "❌ Unable to extract the article from this website. "
            "The publisher may be blocking automated access."
        )

        st.stop()


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    st.subheader("✨ Summary")

    summary_placeholder = st.empty()

    full_summary = ""


    try:

        with st.spinner("Generating summary..."):

            for chunk in generate_summary(article_text):

                full_summary += chunk

                summary_placeholder.markdown(
                    full_summary
                )


    except Exception as e:

        st.error(
            f"❌ Error generating summary: {e}"
        )

        st.stop()


    # -----------------------------------------------------
    # DOWNLOAD SUMMARY
    # -----------------------------------------------------

    st.download_button(
        label="📥 Download Summary",
        data=full_summary,
        file_name="news_summary.txt",
        mime="text/plain",
        use_container_width=True
    )


    # -----------------------------------------------------
    # SAVE HISTORY
    # -----------------------------------------------------

    st.session_state.history.append(
        {
            "title": "News Article",
            "summary": full_summary,
            "url": url
        }
    )