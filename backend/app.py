import time
import streamlit as st
import streamlit.components.v1 as components

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

# Hide the black datalist arrow icon from the input box
st.markdown(
    """
    <style>
    /* Removes the small dropdown/calendar-picker arrow from input fields */
    input::-webkit-calendar-picker-indicator {
        display: none !important;
        opacity: 0 !important;
        -webkit-appearance: none !important;
        width: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "current_summary" not in st.session_state:
    st.session_state.current_summary = ""


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
            st.markdown(f"**{i + 1}.** {title}")
    else:
        st.info("No summaries generated yet.")

    st.markdown("---")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_summary = ""
        st.rerun()


# =====================================================
# MAIN HEADER
# =====================================================

st.title("📰 News Summarizer")
st.markdown("Enter a news article URL and generate a concise summary.")


# =====================================================
# ARTICLE INPUT WITH POPUP URL HISTORY
# =====================================================

st.subheader("🔗 Article Input")

with st.form("news_summary_form"):
    url_input = st.text_input(
        "Paste News Article URL",
        placeholder="https://example.com/news/article",
        key="news_article_url_input"
    )

    submitted = st.form_submit_button(
        "🚀 Generate Summary",
        use_container_width=True
    )

# Inject browser datalist so clicking the input box shows past URLs
past_unique_urls = list(dict.fromkeys([
    item.get("url") for item in reversed(st.session_state.history) if item.get("url")
]))

if past_unique_urls:
    options_html = "".join([f"<option value='{u}'>" for u in past_unique_urls])
    components.html(
        f"""
        <script>
        const input = window.parent.document.querySelector('input[aria-label="Paste News Article URL"]');
        if (input) {{
            let datalist = window.parent.document.getElementById('recent_urls_list');
            if (!datalist) {{
                datalist = window.parent.document.createElement('datalist');
                datalist.id = 'recent_urls_list';
                window.parent.document.body.appendChild(datalist);
            }}
            datalist.innerHTML = "{options_html}";
            input.setAttribute('list', 'recent_urls_list');
            input.setAttribute('autocomplete', 'on');
        }}
        </script>
        """,
        height=0,
        width=0,
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

    # Check cache
    cached_entry = next(
        (item for item in st.session_state.history if item.get("url") == cleaned_url),
        None
    )

    if cached_entry and cached_entry.get("summary"):
        st.session_state.current_summary = cached_entry.get("summary", "")
        st.info("⚡ Loaded summary from recent history.")
    else:
        # Extract article text
        extraction_start = time.perf_counter()
        with st.spinner("🔎 Extracting article"):
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
            with st.spinner("🤖 Generating summary"):
                for chunk in generate_summary(article_text):
                    complete_summary += chunk
                    summary_placeholder.markdown(complete_summary)

            summary_time = time.perf_counter() - summary_start
            st.success(f"✓ Summary generated in {summary_time:.2f} seconds")

            first_line = complete_summary.strip().split("\n")[0].replace("*", "").replace("#", "").strip()
            headline = first_line if first_line else cleaned_url

            # Store in session state history
            st.session_state.history.append({
                "title": headline,
                "url": cleaned_url,
                "summary": complete_summary
            })

            st.session_state.current_summary = complete_summary
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