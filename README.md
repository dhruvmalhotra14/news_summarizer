# 📰 Professional News Summarizer

An AI-powered web application that extracts news articles from URLs and generates concise, informative summaries using Google Gemini AI.

## 🚀 Key Features

* 🤖 **AI-Powered Summarization:** Uses Google Gemini AI to generate concise summaries in 5 key bullet points.
* 🔎 **Article Extraction:** Extracts article content using Trafilatura and BeautifulSoup.
* 🌐 **Jina Reader Fallback:** Uses Jina Reader when direct article extraction is unsuccessful.
* ⚡ **Fast Processing:** Displays article extraction and summary generation time.
* 📋 **Summary History:** Keeps track of previously generated summaries during the current session.
* ⬇️ **Download Summary:** Allows users to download the generated summary as a text file.
* 💻 **Interactive Web Interface:** Built using Streamlit for an easy-to-use interface.

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Google Gemini AI**
* **Google GenAI SDK**
* **Trafilatura**
* **BeautifulSoup**
* **Jina Reader**
* **Requests**

## 🧠 AI Technology

This project uses **Google Gemini AI** through the **Gemini API** and the `google-genai` Python SDK.

The AI reads the extracted article content and generates a concise summary containing five important points.

## 🔄 How It Works

```text
User enters article URL
        ↓
URL validation
        ↓
Article extraction
        ↓
Trafilatura
        ↓
BeautifulSoup fallback
        ↓
Jina Reader fallback
        ↓
Extracted article text
        ↓
Google Gemini AI
        ↓
5-point summary
        ↓
Display summary
        ↓
Download summary
```

## 👨‍💻 Author

**Dhruv Malhotra**

GitHub: [@dhruvmalhotra14](https://github.com/dhruvmalhotra14)

---

⭐ If you find this project useful, consider giving it a star!
