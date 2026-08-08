import requests
import trafilatura
from bs4 import BeautifulSoup


def extract_article(url):

    url = url.strip()

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=30,
            allow_redirects=True
        )

        print("STATUS:", response.status_code)
        print("FINAL URL:", response.url)
        print("CONTENT TYPE:", response.headers.get("content-type"))
        print("CONTENT LENGTH:", len(response.text))

        # ------------------------------------------------
        # If request itself failed
        # ------------------------------------------------

        if response.status_code != 200:
            print("REQUEST FAILED")
            return None

        html = response.text

        # ------------------------------------------------
        # Trafilatura
        # ------------------------------------------------

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_recall=True,
            favor_precision=False
        )

        if text:

            text = text.strip()

            print("TRAFILATURA LENGTH:", len(text))

            if len(text) >= 300:
                return text

        # ------------------------------------------------
        # BeautifulSoup
        # ------------------------------------------------

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # Remove unnecessary elements
        for element in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            element.decompose()

        # ------------------------------------------------
        # Article tag
        # ------------------------------------------------

        article = soup.find("article")

        if article:

            text = article.get_text(
                separator="\n",
                strip=True
            )

            print(
                "ARTICLE TAG LENGTH:",
                len(text)
            )

            if len(text) >= 300:
                return text

        # ------------------------------------------------
        # Paragraphs
        # ------------------------------------------------

        paragraphs = soup.find_all("p")

        text = "\n".join(
            p.get_text(
                " ",
                strip=True
            )
            for p in paragraphs
        )

        print(
            "PARAGRAPH LENGTH:",
            len(text)
        )

        if len(text) >= 300:
            return text

        print("NO ARTICLE FOUND")

        return None

    except Exception as e:

        print(
            "EXTRACTOR ERROR:",
            repr(e)
        )

        return None