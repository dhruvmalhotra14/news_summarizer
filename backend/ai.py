import streamlit as st
from groq import Groq

from prompt import summary_prompt


def generate_summary(article_text: str):
    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(
        api_key=api_key
    )

    prompt = summary_prompt(
        article_text[:4000]
    )

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
        if chunk.choices:
            delta = chunk.choices[0].delta.content

            if delta:
                yield delta