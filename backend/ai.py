from google import genai
from google.genai import types
import streamlit as st

from prompt import summary_prompt


def generate_summary(article_text):

    api_key = st.secrets["GEMINI_API_KEY"]

    client = genai.Client(
        api_key=api_key
    )

    # Keep enough article content for a good summary
    article_text = article_text[:3000]

    prompt = summary_prompt(article_text)

    response = client.models.generate_content_stream(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="minimal"
            )
        )
    )

    for chunk in response:

        if chunk.text:
            yield chunk.text