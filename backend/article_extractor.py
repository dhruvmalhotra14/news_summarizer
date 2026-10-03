import json
import re
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests


REQUEST_TIMEOUT = 20
MIN_ARTICLE_LENGTH = 300


# ============================================================
# CLEAN URL
# ============================================================

def clean_url(url):
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


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


# ============================================================
# FETCH PAGE
# ============================================================

def fetch_page(url):

    headers = {
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,"
            "image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    response = curl_requests.get(
        url,
        headers=headers,
        impersonate="chrome",
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )

    print("HTTP STATUS:", response.status_code)
    print("FINAL URL:", response.url)
    print("HTML LENGTH:", len(response.text))

    return response


# ============================================================
# EXTRACT JSON-LD
# ============================================================

def extract_json_ld(html):

    soup = BeautifulSoup(html, "html.parser")

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    print("JSON-LD scripts found:", len(scripts))

    for script in scripts:

        raw = script.string or script.get_text()

        if not raw:
            continue

        raw = raw.strip()

        try:
            data = json.loads(raw)

        except Exception:
            continue

        objects = []

        if isinstance(data, dict):

            if "@graph" in data:
                graph = data["@graph"]

                if isinstance(graph, list):
                    objects.extend(graph)

            objects.append(data)

        elif isinstance(data, list):

            objects.extend(data)

        for obj in objects:

            if not isinstance(obj, dict):
                continue

            article_body = obj.get("articleBody")

            if not article_body:
                continue

            if isinstance(article_body, list):
                article_body = " ".join(
                    str(x) for x in article_body
                )

            article_body = clean_text(
                str(article_body)
            )

            print(
                "JSON-LD articleBody length:",
                len(article_body)
            )

            if len(article_body) >= MIN_ARTICLE_LENGTH:

                print(
                    "JSON-LD EXTRACTION RETURNING:",
                    len(article_body)
                )

                return article_body

    print("JSON-LD articleBody not found.")

    return ""


# ============================================================
# EXTRACT <ARTICLE>
# ============================================================

def extract_article_tag(html):

    soup = BeautifulSoup(html, "html.parser")

    article = soup.find("article")

    if not article:
        return ""

    for tag in article.find_all(
        ["script", "style", "noscript"]
    ):
        tag.decompose()

    paragraphs = []

    for p in article.find_all("p"):

        text = p.get_text(" ", strip=True)

        if text:
            paragraphs.append(text)

    result = clean_text(
        "\n".join(paragraphs)
    )

    print(
        "<article> extraction length:",
        len(result)
    )

    return result


# ============================================================
# EXTRACT <MAIN>
# ============================================================

def extract_main(html):

    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main")

    if not main:
        return ""

    for tag in main.find_all(
        ["script", "style", "noscript", "nav", "footer"]
    ):
        tag.decompose()

    paragraphs = []

    for p in main.find_all("p"):

        text = p.get_text(" ", strip=True)

        if len(text) > 20:
            paragraphs.append(text)

    result = clean_text(
        "\n".join(paragraphs)
    )

    print(
        "<main> extraction length:",
        len(result)
    )

    return result


# ============================================================
# INDIAN EXPRESS
# ============================================================

def extract_indian_express(url):

    print()
    print("========================================")
    print("INDIAN EXPRESS EXTRACTION")
    print("URL:", url)
    print("========================================")

    response = fetch_page(url)

    if response.status_code != 200:

        print(
            "Indian Express returned status:",
            response.status_code
        )

        return ""

    html = response.text

    print(
        "Indian Express page status:",
        response.status_code
    )

    print(
        "Indian Express HTML length:",
        len(html)
    )

    # --------------------------------------------------------
    # METHOD 1: JSON-LD
    # --------------------------------------------------------

    print()
    print("Trying JSON-LD...")

    text = extract_json_ld(html)

    print(
        "JSON-LD RESULT LENGTH:",
        len(text)
    )

    if len(text) >= MIN_ARTICLE_LENGTH:

        print(
            "SUCCESS: Returning JSON-LD article text"
        )

        print(
            "FINAL RETURN LENGTH:",
            len(text)
        )

        return text

    # --------------------------------------------------------
    # METHOD 2: ARTICLE TAG
    # --------------------------------------------------------

    print()
    print("Trying <article>...")

    text = extract_article_tag(html)

    print(
        "<article> RESULT LENGTH:",
        len(text)
    )

    if len(text) >= MIN_ARTICLE_LENGTH:

        print(
            "SUCCESS: Returning <article> text"
        )

        return text

    # --------------------------------------------------------
    # METHOD 3: MAIN
    # --------------------------------------------------------

    print()
    print("Trying <main>...")

    text = extract_main(html)

    print(
        "<main> RESULT LENGTH:",
        len(text)
    )

    if len(text) >= MIN_ARTICLE_LENGTH:

        print(
            "SUCCESS: Returning <main> text"
        )

        return text

    print()
    print(
        "Indian Express extraction FAILED."
    )

    return ""


# ============================================================
# GENERIC EXTRACTION
# ============================================================

def extract_generic(url):

    print()
    print("========================================")
    print("GENERIC EXTRACTION")
    print("========================================")

    response = fetch_page(url)

    if response.status_code != 200:

        print(
            "Generic page returned:",
            response.status_code
        )

        return ""

    html = response.text

    # JSON-LD first
    text = extract_json_ld(html)

    if len(text) >= MIN_ARTICLE_LENGTH:
        return text

    # <article>
    text = extract_article_tag(html)

    if len(text) >= MIN_ARTICLE_LENGTH:
        return text

    # <main>
    text = extract_main(html)

    if len(text) >= MIN_ARTICLE_LENGTH:
        return text

    print(
        "Generic extraction failed."
    )

    return ""


# ============================================================
# MAIN FUNCTION
# ============================================================

def extract_article(url):

    print()
    print("========================================")
    print("extract_article() CALLED")
    print("URL:", url)
    print("========================================")

    url = clean_url(url)

    print("Clean URL:", url)

    domain = urlsplit(url).netloc.lower()

    print("DOMAIN:", domain)

    # --------------------------------------------------------
    # INDIAN EXPRESS
    # --------------------------------------------------------

    if "indianexpress.com" in domain:

        print("Indian Express detected.")

        result = extract_indian_express(url)

    # --------------------------------------------------------
    # OTHER WEBSITES
    # --------------------------------------------------------

    else:

        print("Generic news website detected.")

        result = extract_generic(url)

    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    result = clean_text(result)

    print()
    print("========================================")
    print("FINAL EXTRACTED TEXT LENGTH:", len(result))
    print("========================================")

    if len(result) < MIN_ARTICLE_LENGTH:

        print(
            "WARNING: Final result is too short."
        )

        return ""

    return result