import time

from google import genai
import streamlit as st

from prompt import summary_prompt


client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


def generate_summary(article_text):

    article_text = article_text[:2500]

    prompt_start = time.perf_counter()

    prompt = summary_prompt(article_text)

    prompt_time = time.perf_counter() - prompt_start

    print(f"PROMPT CREATION: {prompt_time:.2f}s")

    api_start = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    api_time = time.perf_counter() - api_start

    print(f"GEMINI API TIME: {api_time:.2f}s")

    if response.text:
        return response.text.strip()

    return "Unable to generate summary."