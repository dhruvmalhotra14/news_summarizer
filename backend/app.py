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
    st.session_state.history = []

if "current_summary" not in st.session_state:
    st.session_state.current_summary = ""

if "selected_url" not in st.session_state:
    st.session_state.selected_url = ""


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("📰 News Summarizer")
    st.markdown("---")
    st.subheader("Summary History")

    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            title = item.get("title") or item.get("url", "Article")
            st.markdown(f"**{i + 1}.** {title[:40]}...")
    else:
        st.info("No summaries generated yet.")

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
# ARTICLE INPUT & RECENT URL PICKER
# =====================================================

st.subheader("🔗 Article Input")

# Collect past unique URLs for quick autofill
past_urls = [item.get("url") for item in reversed(st.session_state.history) if item.get("url")]

# If user has past searches, show a quick autofill picker right above/at the input
if past_urls:
    selected_from_dropdown = st.selectbox(
        "🕒 Recently Searched Links (click to auto-fill):",
        options=["-- Type or paste a new URL below --"] + past_urls,
        index=0
    )
    if selected_from_dropdown != "-- Type or paste a new URL below --":
        st.session_state.selected_url = selected_from_dropdown

# Main URL input box
url_input = st.text_input(
    "Paste News Article URL",
    value=st.session_state.selected_url,
    placeholder="https://example.com/news/article",
    autocomplete="url"  # Triggers browser autofill popup
)

submitted = st.button("🚀 Generate Summary", use_container_width=True)


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

    # Check cache
    cached_entry = next(
        (item for item in st.session_state.history if item.get("url") == cleaned_url),
        None
    )

    if cached_entry and cached_entry.get("summary"):
        st.session_state.current_summary = cached_entry.get("summary", "")
        st.session_state.selected_url = cleaned_url
        st.info("⚡ Loaded summary from recent history.")
        st.rerun()
    else:
        # Extract article text
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

        # Generate summary via Groq
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