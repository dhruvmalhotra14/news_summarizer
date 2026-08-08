from google import genai
import streamlit as st

from prompt import summary_prompt


def generate_summary(article_text):

    # Get API key only when summary generation is requested
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not available in Streamlit Secrets."
        )

    # Create Gemini client
    client = genai.Client(
        api_key=api_key
    )

    # Limit article length
    article_text = article_text[:6000]

    # Create prompt
    prompt = summary_prompt(article_text)

    # Generate streaming response
    response = client.models.generate_content_stream(
        model="gemini-3.6-flash",
        contents=prompt
    )

    for chunk in response:

        if chunk.text:
            yield chunk.text