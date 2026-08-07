from google import genai
import streamlit as st
from prompt import summary_prompt

# Debug: Show available secret keys
st.write("Available Secrets:", list(st.secrets.keys()))

# Check if API key exists
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ GEMINI_API_KEY not found in Streamlit Secrets.")
    st.stop()

# Create Gemini client
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

def generate_summary(article_text):
    article_text = article_text[:3000]

    prompt = summary_prompt(article_text)

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text