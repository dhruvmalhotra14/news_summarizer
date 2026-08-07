from google import genai
import streamlit as st
from prompt import summary_prompt

# Create Gemini client
client = genai.Client(
    api_key=st.secrets[""]
)

def generate_summary(article_text):
    # Limit article length for faster responses
    article_text = article_text[:3000]

    prompt = summary_prompt(article_text)

    # Stream response like ChatGPT
    response = client.models.generate_content_stream(
        model="gemini-3.6-flash",
        contents=prompt
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text