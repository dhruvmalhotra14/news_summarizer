# 📰 Professional News Summarizer

An AI-powered web application that extracts news articles from URLs and generates concise, structured summaries using the Groq API.

---

### 🚀 Key Features

* **🤖 Fast AI Summarization**: Streams concise, high-impact 5-bullet summaries powered by Groq's high-speed inference engine.
* **🛡️ Multi-Layer Article Extraction**: Bypasses bot protections and Cloudflare firewalls using `curl_cffi` TLS fingerprinting and JSON-LD structured data.
* **🌐 Jina Reader Fallback**: Automatically routes through Jina Reader proxy if direct article scraping encounters edge blocks.
* **🕒 Instant URL Autofill**: Native browser-level autocomplete memory (`<datalist>`) allows instant re-selection of previously analyzed links upon click.
* **⚡ Real-Time Latency Metrics**: Displays extraction speed and streaming generation times directly in the UI.
* **📋 Session History**: Maintains a sidebar log of generated summaries during the active session.
* **⬇️ One-Click Export**: Allows immediate export of generated summaries to a `.txt` file.
* **💻 Clean Streamlit Interface**: Intuitive, single-input user experience with zero manual copy-paste requirements.

---

### 🛠️ Technologies Used

* **Python 3.10+**
* **Streamlit**: Web dashboard and state management
* **Groq SDK**: Ultra-fast LLM inference
* **curl_cffi**: Browser TLS fingerprint impersonation (Cloudflare bypass)
* **Trafilatura**: High-precision main-body web scraper
* **BeautifulSoup4**: HTML parsing and JSON-LD metadata extraction
* **Jina Reader**: Fallback proxy content extraction

---

### 🧠 AI Engine

This project uses the **Groq API** for low-latency text summarization. Once the scraper isolates the core article text and strips away ads, navigation, and site boilerplates, the LLM digests the text to output five distinct, informative takeaways.

---

### 🔄 Architecture Flow

```text
User enters news URL (or clicks autofill history)
                    ↓
        Protocol validation & cleanup
                    ↓
         Multi-Layer Scraper Pipeline
   ┌────────────────────────────────────────┐
   │ 1. TLS-Fingerprinted Fetch (curl_cffi) │
   │ 2. JSON-LD Schema Extraction           │
   │ 3. Trafilatura Precision Parser        │
   │ 4. Jina Reader Engine Proxy            │
   └────────────────────────────────────────┘
                    ↓
         Clean Article Body (>300 chars)
                    ↓
         Groq Inference Engine
                    ↓
        Streamed Concise Summary
                    ↓
      Display, Sidebar Log & Download

👨‍💻 Author

**Dhruv Malhotra**  
GitHub: [@dhruvmalhotra14](https://github.com/dhruvmalhotra14)



⭐ **Found this helpful?** If you like this project, please consider giving it a star on GitHub—it helps support future updates!
