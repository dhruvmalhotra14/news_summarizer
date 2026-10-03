import streamlit as st
from groq import Groq

from prompt import summary_prompt


def generate_summary(article_text: str):
    # Read API key from Streamlit secrets
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from Streamlit secrets.")

    # Create Groq client
    client = Groq(api_key=api_key)

    # Create prompt
    prompt = summary_prompt(article_text[:4000])

    # Generate streaming response
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        stream=True
    )

    for chunk in completion:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if content:
            yield content