def summary_prompt(article_text):

    return f"""
You are an AI news summarizer.

Read the following news article and return a concise summary
in exactly 5 bullet points.

Include:
- Main topic
- Key facts
- Important people or organizations
- Conclusion

Keep the summary under 200 words.

Do not use markdown headings or long explanations.

Article:
{article_text}
"""