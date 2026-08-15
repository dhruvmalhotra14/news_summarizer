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

    # Limit article length for faster processing
    # while keeping enough context for a useful summary.
    article_text = article_text[:2500]

    # Create prompt
    prompt = summary_prompt(article_text)

    # Generate complete response
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    # Return generated summary
    if response.text:
        return response.text.strip()

    return "Unable to generate summary."