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
# SESSION STATE INITIALIZATION
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []  # Stores dicts: {"title": ..., "url": ..., "summary": ...}

if "current_summary" not in st.session_state:
    st.session_state.current_summary = ""

if "selected_url" not in st.session_state:
    st.session_state.selected_url = ""


# =====================================================
# SIDEBAR: RECENT 5 SEARCHES
# =====================================================

with st.sidebar:
    st.title("📰 News Summarizer")
    st.markdown("---")
    st.subheader("🕒 Recent 5 Articles")

    if st.session_state.history:
        # Show last 5 articles in reverse chronological order
        recent_items = list(reversed(st.session_state.history))[:5]

        for i, item in enumerate(recent_items):
            # Show clickable button for each past article
            btn_label = f"📌 {item['title'][:40]}..."
            if st.button(btn_label, key=f"hist_btn_{i}", use_container_width=True):
                st.session_state.selected_url = item["url"]
                st.session_state.current_summary = item["summary"]
                st.rerun()

    else:
        st.info("No recent articles yet.")

    st.markdown("---")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_summary = ""
        st.session_state.selected_url = ""
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
    url_input = st.text_input(
        "Paste News Article URL",
        value=st.session_state.selected_url,
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
    cleaned_url = url_input.strip()

    if not cleaned_url:
        st.warning("⚠️ Please paste a news article URL.")
        st.stop()

    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = "https://" + cleaned_url

    # Check if this exact URL was already summarized before
    cached_entry = next((item for item in st.session_state.history if item["url"] == cleaned_url), None)

    if cached_entry:
        st.session_state.current_summary = cached_entry["summary"]
        st.session_state.selected_url = cleaned_url
        st.info("⚡ Loaded summary from recent history.")
    else:
        # Extract
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

        # Generate summary
        st.subheader("✨ Summary")
        summary_placeholder = st.empty()
        complete_summary = ""
        summary_start = time.perf_counter()

        try:
            with st.spinner("🤖 Generating summary with Groq..."):
                for chunk in generate_summary(article_text):
                    complete_summary += chunk
                    summary_placeholder.markdown(complete_summary)

            summary_time = time.perf_counter() - summary_start
            st.success(f"✓ Summary generated in {summary_time:.2f} seconds")

            # Extract clean title line
            first_line = complete_summary.strip().split("\n")[0].replace("*", "").replace("#", "").strip()
            headline = first_line[:60] if first_line else cleaned_url

            # Store in session state history
            st.session_state.history.append({
                "title": headline,
                "url": cleaned_url,
                "summary": complete_summary
            })

            st.session_state.current_summary = complete_summary
            st.session_state.selected_url = cleaned_url

            st.rerun()

        except Exception as e:
            st.error("❌ Failed to generate the summary.")
            st.caption(f"Error: {e}")
            st.info("Please verify your GROQ_API_KEY inside `.streamlit/secrets.toml`.")
            st.stop()


# =====================================================
# DISPLAY ACTIVE SUMMARY
# =====================================================

if st.session_state.current_summary and not submitted:
    st.subheader("✨ Summary")
    st.markdown(st.session_state.current_summary)

    st.download_button(
        label="⬇️ Download Summary",
        data=st.session_state.current_summary,
        file_name="news_summary.txt",
        mime="text/plain",
        use_container_width=True
    )