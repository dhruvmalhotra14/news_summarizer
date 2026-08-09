from google import genai
import streamlit as st
from prompt import summary_prompt


# =====================================================
# GEMINI CLIENT
# =====================================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# =====================================================
# GENERATE SUMMARY
# =====================================================

def generate_summary(article_text):

    # Limit input size
    article_text = article_text[:2000]

    prompt = summary_prompt(article_text)

    response = client.models.generate_content_stream(
        model="gemini-3.6-flash",
        contents=prompt
    )

    for chunk in response:

        if chunk.text:

            yield chunk.text