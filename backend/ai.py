from google import genai
import streamlit as st
from prompt import summary_prompt

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

def generate_summary(article_text):
    article_text = article_text[:3000]

    prompt = summary_prompt(article_text)

    response = client.models.generate_content_stream(
        model="gemini-3.6-flash",
        contents=prompt
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text