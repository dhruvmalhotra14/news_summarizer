import streamlit as st
from extractor import extract_article
from ai import generate_summary


# ============================================================
# PAGE CONFIG
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


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 46px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #555;
        margin-bottom: 25px;
    }

    .summary-title {
        font-size: 30px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .history-item {
        padding: 10px 5px;
        margin-bottom: 8px;
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 📰 News Summarizer"
    )

    st.divider()

    st.markdown(
        "## Summary History"
    )

    if st.session_state.history:

        for i, item in enumerate(
            st.session_state.history,
            start=1
        ):

            preview = item["summary"]

            # Make sidebar preview shorter
            if len(preview) > 180:
                preview = preview[:180] + "..."

            st.markdown(
                f"""
                <div class="history-item">
                <b>{i}.</b> {preview}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "No summaries generated yet."
        )

    st.divider()

    if st.button(
        "🗑️ Clear History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📰 News Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Paste a news article URL below and generate a concise summary.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# URL FORM
# ============================================================

with st.form(
    "news_form",
    clear_on_submit=False
):

    url = st.text_input(
        "🔗 Paste News Article URL",
        placeholder="https://example.com/news-article",
        label_visibility="visible"
    )

    submitted = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )


# ============================================================
# GENERATE SUMMARY
# ============================================================

if submitted:

    if not url.strip():

        st.warning(
            "⚠️ Please paste a news article URL."
        )

    else:

        url = url.strip()

        # ----------------------------------------------------
        # EXTRACTION
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Extracting article..."
        ):

            article_text = extract_article(url)

        # ----------------------------------------------------
        # EXTRACTION FAILED
        # ----------------------------------------------------

        if not article_text:

            st.error(
                "❌ Unable to extract the article from this website. "
                "The publisher may be blocking automated access."
            )

        else:

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.markdown(
                '<div class="summary-title">✨ Summary</div>',
                unsafe_allow_html=True
            )

            summary_placeholder = st.empty()

            full_summary = ""

            try:

                for chunk in generate_summary(
                    article_text
                ):

                    full_summary += chunk

                    summary_placeholder.markdown(
                        full_summary
                    )

                # --------------------------------------------
                # SAVE HISTORY
                # --------------------------------------------

                if full_summary.strip():

                    st.session_state.history.insert(
                        0,
                        {
                            "url": url,
                            "summary": full_summary
                        }
                    )

                    # Keep latest 10
                    st.session_state.history = (
                        st.session_state.history[:10]
                    )

                    # ----------------------------------------
                    # DOWNLOAD BUTTON
                    # ----------------------------------------

                    st.download_button(
                        label="⬇️ Download Summary",
                        data=full_summary,
                        file_name="news_summary.txt",
                        mime="text/plain",
                        use_container_width=False
                    )

            except Exception as e:

                st.error(
                    "❌ Something went wrong while generating "
                    "the summary."
                )

                st.exception(e)