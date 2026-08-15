def summary_prompt(article_text):

    return f"""
Summarize the following news article in exactly 5 concise bullet points.

Include:
- Main topic
- Key facts
- Important people or organizations
- Conclusion

Rules:
- Keep the total summary under 120 words.
- Use simple and clear language.
- Do not add information that is not present in the article.
- Do not use headings.
- Return only 5 bullet points.

Article:
{article_text}
"""