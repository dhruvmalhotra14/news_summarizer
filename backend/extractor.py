import requests
import trafilatura
from newspaper import Article
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0 Safari/537.36"
    )
}


def extract_article(url):

    # ---------------------------------
    # Method 1: Trafilatura (Improved)
    # ---------------------------------
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        response.raise_for_status()

        downloaded = response.text

        if downloaded:

            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False
            )

            if text and len(text) > 300:
                return "News Article", text

    except Exception:
        pass


    # ---------------------------------
    # Method 2: Newspaper3k
    # ---------------------------------
    try:
        article = Article(url)

        article.download()
        article.parse()

        if article.text and len(article.text) > 300:
            return article.title, article.text

    except Exception:
        pass


    # ---------------------------------
    # Method 3: BeautifulSoup Fallback
    # ---------------------------------
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )


        # Remove unwanted sections
        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside"
            ]
        ):
            tag.decompose()


        title = (
            soup.title.string.strip()
            if soup.title
            else "News Article"
        )


        paragraphs = soup.find_all("p")


        text = "\n".join(
            p.get_text(
                " ",
                strip=True
            )
            for p in paragraphs
        )


        if len(text) > 300:
            return title, text


    except Exception:
        pass


    # ---------------------------------
    # If all methods fail
    # ---------------------------------
    raise Exception(
        "Unable to extract article. "
        "This website may block automated access."
    )