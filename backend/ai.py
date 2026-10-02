import streamlit as st
from groq import Groq
from prompt import summary_prompt


def generate_summary(article_text: str):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in Streamlit secrets.")

    client = Groq(api_key=api_key)

    # Trim input context (8000 chars = lambe articles ke liye better)
    prompt = summary_prompt(article_text[:8000])

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=True,
        )

        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:
        raise RuntimeError(f"Groq API Error: {str(e)}")