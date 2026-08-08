import requests
import trafilatura
from newspaper import Article
from bs4 import BeautifulSoup


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


def extract_article(url):
    """
    Extract article title and text using multiple methods.

    Returns:
        title, article_text
    """

    url = url.strip()

    if not url:
        raise Exception("Please enter a valid article URL.")

    # -------------------------------------------------
    # Method 1: Trafilatura
    # -------------------------------------------------
    try:
        downloaded = trafilatura.fetch_url(url)

        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )

            if text and len(text.strip()) > 300:
                metadata = trafilatura.extract(
                    downloaded,
                    output_format="json",
                    with_metadata=True,
                    include_comments=False,
                    include_tables=False,
                )

                title = "News Article"

                if metadata:
                    try:
                        import json

                        data = json.loads(metadata)

                        title = (
                            data.get("title")
                            or "News Article"
                        )
                    except Exception:
                        pass

                return title, text.strip()

    except Exception:
        pass

    # -------------------------------------------------
    # Method 2: newspaper3k
    # -------------------------------------------------
    try:
        article = Article(url)

        article.download()
        article.parse()

        if article.text and len(article.text.strip()) > 300:

            title = article.title.strip() if article.title else "News Article"

            return title, article.text.strip()

    except Exception:
        pass

    # -------------------------------------------------
    # Method 3: Requests + BeautifulSoup
    # -------------------------------------------------
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Try to get title
        title = "News Article"

        if soup.title:
            title = soup.title.get_text(
                " ",
                strip=True
            )

        # Remove unnecessary elements
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "nav",
                "footer",
                "header",
                "aside",
            ]
        ):
            element.decompose()

        paragraphs = soup.find_all("p")

        text = "\n".join(
            p.get_text(" ", strip=True)
            for p in paragraphs
            if len(p.get_text(" ", strip=True)) > 30
        )

        if len(text.strip()) > 300:
            return title, text.strip()

    except Exception:
        pass

    # -------------------------------------------------
    # All methods failed
    # -------------------------------------------------
    raise Exception(
        "This news website is blocking automated article extraction. "
        "Please copy the article text and use the 'Paste Article Text' "
        "option below."
    )