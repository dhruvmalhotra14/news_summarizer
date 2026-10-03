import json
import re

from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup
import trafilatura
from curl_cffi import requests as curl_requests


def _clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def _is_error_payload(text: str) -> bool:
    if not text or len(text.strip()) < 200:
        return True

    lower = text.lower()

    error_markers = [
        "403 forbidden",
        "404 not found",
        "cloudfront",
        "request could not be satisfied",
        "access denied",
        "attention required",
        "robot or human",
        "enable javascript",
        "just a moment...",
    ]

    return any(marker in lower for marker in error_markers)


def _extract_from_json_ld(soup: BeautifulSoup) -> str:
    for script in soup.find_all(
        "script",
        type="application/ld+json"
    ):
        if not script.string:
            continue

        try:
            data = json.loads(script.string.strip())

            items = data if isinstance(data, list) else [data]

            for item in items:
                if not isinstance(item, dict):
                    continue

                if item.get("articleBody"):
                    return item["articleBody"].strip()

                for graph_item in item.get("@graph", []):
                    if (
                        isinstance(graph_item, dict)
                        and graph_item.get("articleBody")
                    ):
                        return graph_item["articleBody"].strip()

        except Exception:
            continue

    return ""


def _extract_soup_paragraphs(soup: BeautifulSoup) -> str:

    for tag in soup([
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
    ]):
        tag.decompose()

    container = soup.find(
        "div",
        class_=re.compile(
            r"(story-details|full-details|story_details|ie-content|"
            r"art-content|wp-block-post-content|article-body|storycontent)",
            re.IGNORECASE,
        ),
    )

    if not container:
        container = soup.find("article")

    if not container:
        container = soup.find("main")

    target = container if container else soup

    paragraphs = []

    for p in target.find_all("p"):
        text = p.get_text(" ", strip=True)

        if len(text) <= 35:
            continue

        if text.startswith(
            (
                "Also Read",
                "Click here",
                "Subscribe",
                "Explained |",
                "Follow us on",
                "Follow us",
                "Read More",
                "Advertisement",
            )
        ):
            continue

        paragraphs.append(text)

    if len(paragraphs) >= 2:
        return "\n\n".join(paragraphs)

    return ""


def _indian_express_lite_url(url: str) -> str:
    """
    Convert:

    https://indianexpress.com/article/india/example/

    into:

    https://indianexpress.com/article/india/example/lite/
    """

    parts = urlsplit(url)

    path = parts.path.rstrip("/")

    if not path.endswith("/lite"):
        path += "/lite"

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            path + "/",
            parts.query,
            parts.fragment,
        )
    )


def _extract_indian_express_page(
    url: str,
    label: str = "Indian Express"
) -> str:

    print(f"Trying {label} extraction...")
    print("URL:", url)

    try:
        response = curl_requests.get(
            url,
            impersonate="chrome120",
            timeout=15,
            headers={
                "Referer": "https://www.google.com/",
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,image/avif,"
                    "image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        )

        print(f"{label} page status:", response.status_code)

        if response.status_code != 200:
            return ""

        html = response.text

        if not html:
            return ""

        if _is_error_payload(html):
            print(f"{label} returned an error page.")
            return ""

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # 1. JSON-LD

        json_ld = _extract_from_json_ld(soup)

        if json_ld and not _is_error_payload(json_ld):
            print(
                f"{label} JSON-LD extraction successful."
            )

            return _clean_text(json_ld)

        # 2. Known article containers

        containers = soup.find_all(
            "div",
            class_=re.compile(
                r"(story-details|full-details|ie-content|"
                r"article-body|storycontent|story_details|"
                r"art-content)",
                re.IGNORECASE,
            ),
        )

        for container in containers:

            paragraphs = []

            for p in container.find_all("p"):

                text = p.get_text(
                    " ",
                    strip=True
                )

                if len(text) <= 35:
                    continue

                if text.startswith(
                    (
                        "Also Read",
                        "Click here",
                        "Subscribe",
                        "Follow us",
                        "Read More",
                        "Advertisement",
                    )
                ):
                    continue

                paragraphs.append(text)

            if paragraphs:

                article_text = "\n\n".join(
                    paragraphs
                )

                article_text = _clean_text(
                    article_text
                )

                if not _is_error_payload(
                    article_text
                ):
                    print(
                        f"{label} HTML container "
                        "extraction successful."
                    )

                    return article_text

        # 3. <article> or <main>

        container = soup.find("article")

        if not container:
            container = soup.find("main")

        if container:

            paragraphs = []

            for p in container.find_all("p"):

                text = p.get_text(
                    " ",
                    strip=True
                )

                if len(text) <= 35:
                    continue

                if text.startswith(
                    (
                        "Also Read",
                        "Click here",
                        "Subscribe",
                        "Follow us",
                        "Read More",
                        "Advertisement",
                    )
                ):
                    continue

                paragraphs.append(text)

            if paragraphs:

                article_text = "\n\n".join(
                    paragraphs
                )

                article_text = _clean_text(
                    article_text
                )

                if not _is_error_payload(
                    article_text
                ):
                    print(
                        f"{label} article/main "
                        "extraction successful."
                    )

                    return article_text

        # 4. General paragraph extraction

        dom = _extract_soup_paragraphs(soup)

        if dom and not _is_error_payload(dom):

            print(
                f"{label} general paragraph "
                "extraction successful."
            )

            return _clean_text(dom)

    except Exception as e:

        print(
            f"{label} extraction error:",
            e
        )

    return ""


def _fetch_indian_express_direct(url: str) -> str:

    print("Indian Express detected.")

    # Attempt 1: Normal article

    article_text = _extract_indian_express_page(
        url,
        "Indian Express direct"
    )

    if article_text:
        return article_text

    print(
        "Indian Express direct extraction failed."
    )

    # Attempt 2: /lite/ version

    lite_url = _indian_express_lite_url(url)

    if lite_url != url:

        print(
            "Trying Indian Express /lite/ fallback..."
        )

        article_text = _extract_indian_express_page(
            lite_url,
            "Indian Express /lite/"
        )

        if article_text:

            print(
                "Indian Express /lite/ "
                "extraction successful."
            )

            return article_text

    print(
        "Indian Express /lite/ extraction failed."
    )

    return ""


def extract_article(url: str) -> str:

    clean_url = url.strip()

    # INDIAN EXPRESS

    if "indianexpress.com" in clean_url.lower():

        ie_text = _fetch_indian_express_direct(
            clean_url
        )

        if ie_text:
            return ie_text

        print(
            "Indian Express extraction "
            "completely failed."
        )

    # STANDARD WEBSITE EXTRACTION

    try:

        response = curl_requests.get(
            clean_url,
            impersonate="chrome120",
            timeout=10,
            headers={
                "Referer": "https://www.google.com/",
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        )

        print(
            "Website status:",
            response.status_code
        )

        if response.status_code == 200:

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            # JSON-LD

            json_ld = _extract_from_json_ld(
                soup
            )

            if (
                json_ld
                and not _is_error_payload(json_ld)
            ):

                print(
                    "JSON-LD extraction successful."
                )

                return _clean_text(
                    json_ld
                )

            # Trafilatura

            traf = trafilatura.extract(
                response.text,
                no_fallback=False
            )

            if (
                traf
                and not _is_error_payload(traf)
            ):

                print(
                    "Trafilatura extraction successful."
                )

                return _clean_text(traf)

            # BeautifulSoup

            dom = _extract_soup_paragraphs(
                soup
            )

            if (
                dom
                and not _is_error_payload(dom)
            ):

                print(
                    "BeautifulSoup extraction successful."
                )

                return _clean_text(dom)

    except Exception as e:

        print(
            "Standard extraction error:",
            e
        )

    # TRAFILATURA FALLBACK

    try:

        print(
            "Trying Trafilatura fetch fallback..."
        )

        html = trafilatura.fetch_url(
            clean_url
        )

        if html:

            traf = trafilatura.extract(
                html,
                no_fallback=False
            )

            if (
                traf
                and not _is_error_payload(traf)
            ):

                print(
                    "Trafilatura fallback "
                    "successful."
                )

                return _clean_text(traf)

    except Exception as e:

        print(
            "Trafilatura fallback error:",
            e
        )

    return ""