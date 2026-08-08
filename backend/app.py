import streamlit as st
from extractor import extract_article
from ai import generate_summary


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide"
)


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📰 News Summarizer")

    st.divider()

    st.subheader("Summary History")

    if st.session_state.history:

        for i, item in enumerate(st.session_state.history, 1):

            title = item.get("title", "News Article")

            st.markdown(
                f"**{i}.** {title}"
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


# ==========================================
# MAIN TITLE
# ==========================================

st.title("🔗 Article Input")


# ==========================================
# ARTICLE INPUT FORM
# ==========================================

with st.form("article_form"):

    url = st.text_input(
        "🔗 Paste News Article URL",
        placeholder="https://..."
    )

    generate = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )


# ==========================================
# GENERATE SUMMARY
# ==========================================

if generate:

    # --------------------------------------
    # Check URL
    # --------------------------------------

    if not url.strip():

        st.warning(
            "⚠️ Please enter a news article URL."
        )

        st.stop()


    # --------------------------------------
    # Extract Article
    # --------------------------------------

    with st.spinner("🔎 Extracting article..."):

        try:

            article_text, title = extract_article(url)

        except Exception:

            st.error(
                "❌ Unable to extract the article from this website. "
                "The publisher may be blocking automated access."
            )

            st.stop()


    # --------------------------------------
    # Validate Article
    # --------------------------------------

    if not article_text or len(article_text.strip()) < 100:

        st.error(
            "❌ Unable to extract enough article content "
            "from this website."
        )

        st.stop()


    # ======================================
    # SUMMARY
    # ======================================

    st.subheader("✨ Summary")

    summary_placeholder = st.empty()

    full_summary = ""


    # --------------------------------------
    # Generate Summary
    # --------------------------------------

    with st.spinner("Generating summary..."):

        try:

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


    # ======================================
    # DOWNLOAD SUMMARY
    # ======================================

    st.download_button(
        label="📥 Download Summary",
        data=full_summary,
        file_name="news_summary.txt",
        mime="text/plain",
        use_container_width=True
    )


    # ======================================
    # SAVE TO HISTORY
    # ======================================

    st.session_state.history.append(
        {
            "title": title if title else "News Article",
            "summary": full_summary,
            "url": url
        }
    )