import streamlit as st
from groq import Groq
from prompt import summary_prompt


def generate_summary(article_text: str):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in Streamlit secrets.")

    client = Groq(api_key=api_key)

    # Trim to 4,000 characters to keep prompt compact and fast
    trimmed_article = article_text[:4000]
    prompt = summary_prompt(trimmed_article)

    # Streaming completion via Groq (llama-3.3-70b-versatile or llama-3.1-8b-instant)
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta