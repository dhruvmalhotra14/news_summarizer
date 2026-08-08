import json
import requests
import trafilatura

from bs4 import BeautifulSoup
from newspaper import Article, Config


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def clean_text(text):
    if not text:
        return ""

    lines = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if len(line) > 30:
            lines.append(line)

    return "\n\n".join(lines)


# ============================================================
# 1. Trafilatura
# ============================================================

def extract_with_trafilatura(url):

    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return None, None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_precision=True,
        )

        if text:
            text = clean_text(text)

            if len(text) >= 300:
                return "News Article", text

    except Exception:
        pass

    return None, None


# ============================================================
# 2. JSON-LD
# ============================================================

def extract_from_json_ld(soup):

    try:

        scripts = soup.find_all(
            "script",
            type="application/ld+json"
        )

        for script in scripts:

            if not script.string:
                continue

            try:
                data = json.loads(script.string)
            except Exception:
                continue

            objects = data if isinstance(data, list) else [data]

            for item in objects:

                if not isinstance(item, dict):
                    continue

                article_body = item.get("articleBody")

                if article_body:

                    text = clean_text(article_body)

                    if len(text) >= 300:

                        title = (
                            item.get("headline")
                            or item.get("name")
                            or "News Article"
                        )

                        return title, text

    except Exception:
        pass

    return None, None


# ============================================================
# 3. HTML Article
# ============================================================

def extract_from_html(soup):

    try:

        for element in soup.find_all([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "iframe"
        ]):
            element.decompose()

        selectors = [
            "article",
            "[itemprop='articleBody']",
            ".article-body",
            ".article-content",
            ".articleBody",
            ".story-content",
            ".story-body",
            ".story__content",
            ".article__content",
            ".story-detail",
            ".entry-content",
            ".post-content",
        ]

        for selector in selectors:

            container = soup.select_one(selector)

            if not container:
                continue

            paragraphs = container.find_all("p")

            text = "\n".join(
                p.get_text(" ", strip=True)
                for p in paragraphs
            )

            text = clean_text(text)

            if len(text) >= 300:

                h1 = soup.find("h1")

                title = (
                    h1.get_text(" ", strip=True)
                    if h1
                    else "News Article"
                )

                return title, text

    except Exception:
        pass

    return None, None


# ============================================================
# 4. Newspaper3k
# ============================================================

def extract_with_newspaper(url):

    try:

        config = Config()

        config.browser_user_agent = HEADERS["User-Agent"]
        config.request_timeout = 20

        article = Article(
            url,
            config=config
        )

        article.download()
        article.parse()

        text = clean_text(article.text)

        if len(text) >= 300:

            title = (
                article.title.strip()
                if article.title
                else "News Article"
            )

            return title, text

    except Exception:
        pass

    return None, None


# ============================================================
# 5. Jina Reader API
# ============================================================

def extract_with_jina(url):

    try:

        reader_url = "https://r.jina.ai/" + url

        response = requests.get(
            reader_url,
            timeout=30,
            headers={
                "User-Agent": HEADERS["User-Agent"]
            }
        )

        if response.status_code != 200:
            return None, None

        content = response.text.strip()

        if not content:
            return None, None

        # Jina Reader returns markdown/plain text.
        # Remove obvious metadata if present.
        lines = content.splitlines()

        title = "News Article"

        cleaned_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            if line.startswith("Title:"):
                title = line.replace(
                    "Title:",
                    "",
                    1
                ).strip()

                continue

            if line.startswith("URL Source:"):
                continue

            if line.startswith("Published Time:"):
                continue

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        text = clean_text(text)

        if len(text) >= 300:

            # Try to get a better title from markdown heading
            for line in cleaned_lines[:10]:

                if line.startswith("# "):

                    title = line[2:].strip()

                    break

            return title, text

    except Exception:
        pass

    return None, None


# ============================================================
# MAIN FUNCTION
# ============================================================

def extract_article(url):

    url = url.strip()

    if not url:

        raise Exception(
            "Please enter a news article URL."
        )

    if not (
        url.startswith("http://")
        or url.startswith("https://")
    ):

        raise Exception(
            "Please enter a valid URL starting with "
            "http:// or https://"
        )


    # --------------------------------------------------------
    # Try normal extraction first
    # --------------------------------------------------------

    title, text = extract_with_trafilatura(url)

    if text:
        return title, text


    # --------------------------------------------------------
    # Download original page
    # --------------------------------------------------------

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
            allow_redirects=True
        )

        if response.status_code == 200:

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            # JSON-LD
            title, text = extract_from_json_ld(soup)

            if text:
                return title, text

            # HTML article
            title, text = extract_from_html(soup)

            if text:
                return title, text

    except Exception:
        pass


    # --------------------------------------------------------
    # Newspaper3k
    # --------------------------------------------------------

    title, text = extract_with_newspaper(url)

    if text:
        return title, text


    # --------------------------------------------------------
    # Jina Reader fallback
    # --------------------------------------------------------

    title, text = extract_with_jina(url)

    if text:
        return title, text


    # --------------------------------------------------------
    # Failed
    # --------------------------------------------------------

    raise Exception(
        "Unable to extract this article. "
        "The publisher or its access controls may be "
        "preventing automated retrieval."
    )