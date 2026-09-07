import requests
import trafilatura

from bs4 import BeautifulSoup
from urllib.parse import urlparse


# =====================================================
# REQUEST HEADERS
# =====================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    if not text:
        return ""

    lines = []

    for line in text.splitlines():

        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n\n".join(lines).strip()


# =====================================================
# DIRECT EXTRACTION
# =====================================================

def extract_direct(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
            allow_redirects=True
        )

        print("STATUS:", response.status_code)

        if response.status_code != 200:
            return ""

        html = response.text

        if not html:
            return ""

        # -------------------------------------------------
        # 1. Trafilatura
        # -------------------------------------------------

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_recall=True,
            favor_precision=False
        )

        text = clean_text(text)

        if len(text) >= 300:

            print("EXTRACTED USING: Trafilatura")

            return text


        # -------------------------------------------------
        # 2. BeautifulSoup
        # -------------------------------------------------

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # Remove unnecessary content

        for element in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "svg"
        ]):

            element.decompose()


        # -------------------------------------------------
        # 3. Article tag
        # -------------------------------------------------

        article = soup.find("article")

        if article:

            text = article.get_text(
                separator="\n",
                strip=True
            )

            text = clean_text(text)

            if len(text) >= 300:

                print("EXTRACTED USING: Article tag")

                return text


        # -------------------------------------------------
        # 4. Paragraphs
        # -------------------------------------------------

        paragraphs = soup.find_all("p")

        text = "\n\n".join(
            p.get_text(
                " ",
                strip=True
            )
            for p in paragraphs
            if len(
                p.get_text(
                    " ",
                    strip=True
                )
            ) > 40
        )

        text = clean_text(text)

        if len(text) >= 300:

            print("EXTRACTED USING: Paragraphs")

            return text

    except Exception as e:

        print(
            "DIRECT EXTRACTION ERROR:",
            e
        )

    return ""


# =====================================================
# JINA READER FALLBACK
# =====================================================

def extract_jina(url):

    try:

        response = requests.get(
            "https://r.jina.ai/" + url,
            headers={
                "User-Agent": "NewsSummarizer/1.0"
            },
            timeout=40
        )

        print(
            "JINA STATUS:",
            response.status_code
        )

        if response.status_code != 200:
            return ""

        text = clean_text(
            response.text
        )

        if len(text) >= 300:

            print(
                "EXTRACTED USING: Jina Reader"
            )

            return text

    except Exception as e:

        print(
            "JINA ERROR:",
            e
        )

    return ""


# =====================================================
# MAIN EXTRACTION FUNCTION
# =====================================================

def extract_article(url):

    url = url.strip()

    if not url:
        return None

    # -------------------------------------------------
    # Validate URL
    # -------------------------------------------------

    try:

        parsed = urlparse(url)

        if parsed.scheme not in (
            "http",
            "https"
        ):
            return None

        if not parsed.netloc:
            return None

    except Exception:

        return None


    # -------------------------------------------------
    # Try direct extraction
    # -------------------------------------------------

    article = extract_direct(url)

    if article:

        return article


    # -------------------------------------------------
    # Try Jina Reader
    # -------------------------------------------------

    article = extract_jina(url)

    if article:

        return article


    # -------------------------------------------------
    # Nothing found
    # -------------------------------------------------

    return None