import json
import re
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup
import trafilatura
from curl_cffi import requests as curl_requests


REQUEST_TIMEOUT = 20
MIN_ARTICLE_LENGTH = 300


# ============================================================
# URL
# ============================================================

def clean_url(url: str) -> str:

    url = url.strip()

    parts = urlsplit(url)

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            parts.query,
            ""
        )
    )


def get_domain(url: str) -> str:

    return (
        urlsplit(url)
        .netloc
        .lower()
        .replace("www.", "")
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\xa0", " ")

    text = re.sub(
        r"\r\n?",
        "\n",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# REQUEST
# ============================================================

def fetch_page(url: str):

    try:

        response = curl_requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            impersonate="chrome",
            allow_redirects=True,
            headers={
                "Accept": (
                    "text/html,"
                    "application/xhtml+xml,"
                    "application/xml;q=0.9,"
                    "image/avif,"
                    "image/webp,"
                    "*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            }
        )

        return response

    except Exception as e:

        print(
            f"Request error: {e}"
        )

        return None


# ============================================================
# JSON-LD
# ============================================================

def extract_json_ld(soup: BeautifulSoup) -> str:

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    print(
        f"JSON-LD scripts found: {len(scripts)}"
    )

    for script in scripts:

        raw = script.string

        if not raw:

            raw = script.get_text()

        if not raw:

            continue

        try:

            data = json.loads(raw)

        except Exception:

            continue

        candidates = []

        if isinstance(data, dict):

            candidates.append(data)

            graph = data.get("@graph")

            if isinstance(graph, list):

                candidates.extend(graph)

        elif isinstance(data, list):

            candidates.extend(data)

        for item in candidates:

            if not isinstance(item, dict):

                continue

            article_body = item.get(
                "articleBody"
            )

            if not isinstance(
                article_body,
                str
            ):

                continue

            article_body = clean_text(
                article_body
            )

            print(
                f"JSON-LD articleBody length: "
                f"{len(article_body)}"
            )

            if len(article_body) >= MIN_ARTICLE_LENGTH:

                return article_body

    return ""


# ============================================================
# ARTICLE TAG
# ============================================================

def extract_article_tag(
    soup: BeautifulSoup
) -> str:

    article = soup.find("article")

    if not article:

        return ""

    for tag in article.find_all(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]
    ):

        tag.decompose()

    paragraphs = []

    for p in article.find_all(
        ["p", "h1", "h2", "h3"]
    ):

        text = p.get_text(
            " ",
            strip=True
        )

        if len(text) >= 30:

            paragraphs.append(text)

    return clean_text(
        "\n\n".join(paragraphs)
    )


# ============================================================
# MAIN
# ============================================================

def extract_main(
    soup: BeautifulSoup
) -> str:

    main = soup.find("main")

    if not main:

        return ""

    for tag in main.find_all(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]
    ):

        tag.decompose()

    paragraphs = []

    for p in main.find_all("p"):

        text = p.get_text(
            " ",
            strip=True
        )

        if len(text) >= 30:

            paragraphs.append(text)

    return clean_text(
        "\n\n".join(paragraphs)
    )


# ============================================================
# GENERIC PARAGRAPHS
# ============================================================

def extract_paragraphs(
    soup: BeautifulSoup
) -> str:

    for tag in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]
    ):

        tag.decompose()

    paragraphs = []

    for p in soup.find_all("p"):

        text = p.get_text(
            " ",
            strip=True
        )

        if len(text) < 40:

            continue

        paragraphs.append(text)

    return clean_text(
        "\n\n".join(paragraphs)
    )


# ============================================================
# TRAFILATURA
# ============================================================

def extract_trafilatura(
    html: str
) -> str:

    if not html:

        return ""

    try:

        result = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            include_images=False,
            favor_precision=True
        )

        if result:

            return clean_text(result)

    except Exception as e:

        print(
            f"Trafilatura error: {e}"
        )

    return ""


# ============================================================
# INDIAN EXPRESS
# ============================================================

def extract_indian_express(
    url: str
) -> str:

    print(
        "Indian Express detected."
    )

    print(
        "Trying Indian Express direct extraction..."
    )

    response = fetch_page(url)

    if response is None:

        print(
            "Indian Express request failed."
        )

        return ""

    print(
        f"Indian Express page status: "
        f"{response.status_code}"
    )

    if response.status_code != 200:

        return ""

    html = response.text

    print(
        f"Indian Express HTML length: "
        f"{len(html)}"
    )

    if not html:

        return ""

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # --------------------------------------------------------
    # 1. JSON-LD
    # --------------------------------------------------------

    result = extract_json_ld(
        soup
    )

    print(
        f"JSON-LD result length: "
        f"{len(result)}"
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express JSON-LD extraction successful."
        )

        return result

    # --------------------------------------------------------
    # 2. Article
    # --------------------------------------------------------

    result = extract_article_tag(
        soup
    )

    print(
        f"Article tag result length: "
        f"{len(result)}"
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express article extraction successful."
        )

        return result

    # --------------------------------------------------------
    # 3. Main
    # --------------------------------------------------------

    result = extract_main(
        soup
    )

    print(
        f"Main result length: "
        f"{len(result)}"
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express main extraction successful."
        )

        return result

    # --------------------------------------------------------
    # 4. Trafilatura
    # --------------------------------------------------------

    result = extract_trafilatura(
        html
    )

    print(
        f"Trafilatura result length: "
        f"{len(result)}"
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express Trafilatura extraction successful."
        )

        return result

    # --------------------------------------------------------
    # 5. Paragraphs
    # --------------------------------------------------------

    result = extract_paragraphs(
        soup
    )

    print(
        f"Paragraph result length: "
        f"{len(result)}"
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express paragraph extraction successful."
        )

        return result

    print(
        "Indian Express extraction completely failed."
    )

    return ""


# ============================================================
# GENERIC WEBSITE
# ============================================================

def extract_generic(
    url: str
) -> str:

    print(
        "Trying generic article extraction..."
    )

    response = fetch_page(url)

    if response is None:

        return ""

    print(
        f"Website status: "
        f"{response.status_code}"
    )

    if response.status_code != 200:

        return ""

    html = response.text

    if not html:

        return ""

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    result = extract_json_ld(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    result = extract_article_tag(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    result = extract_main(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    result = extract_trafilatura(
        html
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    result = extract_paragraphs(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    return ""


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def extract_article(
    url: str
) -> str:

    print(
        "========================================"
    )

    print(
        "extract_article() called"
    )

    print(
        f"URL: {url}"
    )

    print(
        "========================================"
    )

    if not url:

        print(
            "Empty URL."
        )

        return ""

    url = clean_url(url)

    if not url.startswith(
        ("http://", "https://")
    ):

        print(
            "Invalid URL."
        )

        return ""

    domain = get_domain(url)

    # --------------------------------------------------------
    # Indian Express
    # --------------------------------------------------------

    if (
        domain == "indianexpress.com"
        or domain.endswith(
            ".indianexpress.com"
        )
    ):

        result = extract_indian_express(
            url
        )

    else:

        result = extract_generic(
            url
        )

    print(
        "========================================"
    )

    print(
        f"FINAL EXTRACTED TEXT LENGTH: "
        f"{len(result)}"
    )

    print(
        "========================================"
    )

    return result