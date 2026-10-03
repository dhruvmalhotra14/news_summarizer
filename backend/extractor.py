import json
import re
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup
import trafilatura

from curl_cffi import requests as curl_requests


# ============================================================
# CONSTANTS
# ============================================================

REQUEST_TIMEOUT = 20

MIN_ARTICLE_LENGTH = 300


# ============================================================
# URL HELPERS
# ============================================================

def clean_url(url: str) -> str:

    url = url.strip()

    parts = urlsplit(url)

    cleaned = urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            parts.query,
            ""
        )
    )

    return cleaned


def get_domain(url: str) -> str:

    return (
        urlsplit(url)
        .netloc
        .lower()
        .replace("www.", "")
    )


# ============================================================
# HTML CLEANING
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
# JSON-LD EXTRACTION
# ============================================================

def extract_json_ld(
    soup: BeautifulSoup
) -> str:

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    for script in scripts:

        raw = script.string

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

            if (
                isinstance(article_body, str)
                and len(article_body.strip()) >= MIN_ARTICLE_LENGTH
            ):

                return clean_text(
                    article_body
                )

    return ""


# ============================================================
# ARTICLE CONTAINER EXTRACTION
# ============================================================

def extract_from_article_tag(
    soup: BeautifulSoup
) -> str:

    article = soup.find("article")

    if not article:

        return ""

    # Remove unwanted elements

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
# MAIN CONTENT EXTRACTION
# ============================================================

def extract_from_main(
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
# KNOWN CONTENT CONTAINERS
# ============================================================

def extract_from_known_containers(
    soup: BeautifulSoup
) -> str:

    selectors = [
        "[itemprop='articleBody']",
        ".article-body",
        ".article-content",
        ".article__content",
        ".story-content",
        ".story__content",
        ".post-content",
        ".entry-content",
        ".article-detail",
        ".article-details",
        ".content-area",
        ".articleBody",
        "#article-body"
    ]

    for selector in selectors:

        container = soup.select_one(
            selector
        )

        if not container:
            continue

        for tag in container.find_all(
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

        for p in container.find_all(
            "p"
        ):

            text = p.get_text(
                " ",
                strip=True
            )

            if len(text) >= 30:

                paragraphs.append(text)

        result = clean_text(
            "\n\n".join(paragraphs)
        )

        if len(result) >= MIN_ARTICLE_LENGTH:

            return result

    return ""


# ============================================================
# GENERIC PARAGRAPH EXTRACTION
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

        # Ignore obvious navigation/social text

        lowered = text.lower()

        ignored_phrases = [
            "subscribe",
            "follow us",
            "advertisement",
            "sign up",
            "newsletter",
            "read more",
            "share this article",
            "cookie policy",
            "privacy policy"
        ]

        if any(
            phrase in lowered
            for phrase in ignored_phrases
        ):

            continue

        paragraphs.append(text)

    return clean_text(
        "\n\n".join(paragraphs)
    )


# ============================================================
# TRafilatura EXTRACTION
# ============================================================

def extract_with_trafilatura(
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

    except Exception:

        pass

    return ""


# ============================================================
# HTTP REQUEST
# ============================================================

def fetch_page(
    url: str
):

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
                "Accept-Language": (
                    "en-US,en;q=0.9"
                ),
                "Cache-Control": "no-cache"
            }
        )

        return response

    except Exception as e:

        print(
            f"Request error: {e}"
        )

        return None


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

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # Do not attempt to bypass a 403 response.
    #
    # Indian Express may intentionally block automated
    # requests from cloud/server IP addresses.
    # --------------------------------------------------------

    if response.status_code == 403:

        print(
            "Indian Express returned HTTP 403."
        )

        print(
            "Automated extraction is blocked."
        )

        return ""

    if response.status_code != 200:

        print(
            "Indian Express request failed."
        )

        return ""

    html = response.text

    if not html:

        return ""

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # --------------------------------------------------------
    # JSON-LD
    # --------------------------------------------------------

    result = extract_json_ld(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express JSON-LD extraction successful."
        )

        return result

    # --------------------------------------------------------
    # Article tag
    # --------------------------------------------------------

    result = extract_from_article_tag(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express article extraction successful."
        )

        return result

    # --------------------------------------------------------
    # Known containers
    # --------------------------------------------------------

    result = extract_from_known_containers(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express container extraction successful."
        )

        return result

    # --------------------------------------------------------
    # Main
    # --------------------------------------------------------

    result = extract_from_main(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express main extraction successful."
        )

        return result

    # --------------------------------------------------------
    # Generic paragraphs
    # --------------------------------------------------------

    result = extract_paragraphs(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express paragraph extraction successful."
        )

        return result

    # --------------------------------------------------------
    # Trafilatura
    # --------------------------------------------------------

    result = extract_with_trafilatura(
        html
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        print(
            "Indian Express Trafilatura extraction successful."
        )

        return result

    print(
        "Indian Express extraction failed."
    )

    return ""


# ============================================================
# GENERIC ARTICLE EXTRACTION
# ============================================================

def extract_generic_article(
    url: str
) -> str:

    print(
        "Trying generic article extraction..."
    )

    response = fetch_page(url)

    if response is None:

        print(
            "Website request failed."
        )

        return ""

    print(
        f"Website status: "
        f"{response.status_code}"
    )

    if response.status_code == 403:

        print(
            "Website returned HTTP 403."
        )

        print(
            "Automated extraction is blocked."
        )

        return ""

    if response.status_code != 200:

        return ""

    html = response.text

    if not html:

        return ""

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # --------------------------------------------------------
    # JSON-LD
    # --------------------------------------------------------

    result = extract_json_ld(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    # --------------------------------------------------------
    # Article tag
    # --------------------------------------------------------

    result = extract_from_article_tag(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    # --------------------------------------------------------
    # Known containers
    # --------------------------------------------------------

    result = extract_from_known_containers(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    # --------------------------------------------------------
    # Main
    # --------------------------------------------------------

    result = extract_from_main(
        soup
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    # --------------------------------------------------------
    # Trafilatura
    # --------------------------------------------------------

    result = extract_with_trafilatura(
        html
    )

    if len(result) >= MIN_ARTICLE_LENGTH:

        return result

    # --------------------------------------------------------
    # Generic paragraphs
    # --------------------------------------------------------

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

    """
    Extract article text from a news URL.

    Returns:
        str: Extracted article text.
        Empty string if extraction fails.
    """

    if not url:

        return ""

    url = clean_url(url)

    if not url.startswith(
        ("http://", "https://")
    ):

        return ""

    domain = get_domain(url)

    print(
        f"URL: {url}"
    )

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

        if result:

            return result

        print(
            "Indian Express extraction completely failed."
        )

        return ""

    # --------------------------------------------------------
    # Generic websites
    # --------------------------------------------------------

    result = extract_generic_article(
        url
    )

    if result:

        return result

    print(
        "Generic article extraction failed."
    )

    return ""