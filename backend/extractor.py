import json
import re

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
            r"(story-details|full-details|story_details|"
            r"ie-content|art-content|wp-block-post-content|"
            r"article-body|storycontent)",
            re.IGNORECASE,
        ),
    )

    if not container:
        container = soup.find(["article", "main"])

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
            )
        ):
            continue

        paragraphs.append(text)

    if len(paragraphs) >= 2:
        return "\n\n".join(paragraphs)

    return ""


def _fetch_indian_express_direct(url: str) -> str:
    """
    Try extracting Indian Express articles using
    the WordPress REST API.
    """

    clean_target = url.split("?")[0].rstrip("/")

    match = re.search(r"-(\d+)$", clean_target)

    if not match:
        print("Indian Express: Article ID not found.")
        return ""

    post_id = match.group(1)

    api_url = (
        f"https://indianexpress.com/wp-json/wp/v2/posts/{post_id}"
    )

    print("Indian Express API URL:", api_url)

    try:
        response = curl_requests.get(
            api_url,
            impersonate="chrome120",
            timeout=15,
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        )

        print(
            "Indian Express API status:",
            response.status_code
        )

        if response.status_code != 200:
            return ""

        data = response.json()

        rendered_html = (
            data
            .get("content", {})
            .get("rendered", "")
        )

        if not rendered_html:
            return ""

        soup = BeautifulSoup(
            rendered_html,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "iframe",
            "figure",
            "figcaption",
        ]):
            tag.decompose()

        paragraphs = []

        for p in soup.find_all("p"):
            text = p.get_text(" ", strip=True)

            if len(text) <= 35:
                continue

            if text.startswith(
                (
                    "Also Read",
                    "Click here",
                    "Subscribe",
                    "Follow us",
                )
            ):
                continue

            paragraphs.append(text)

        if paragraphs:
            article_text = "\n\n".join(paragraphs)
            article_text = _clean_text(article_text)

            if not _is_error_payload(article_text):
                print(
                    "Indian Express article extracted successfully."
                )

                return article_text

    except Exception as e:
        print(
            "Indian Express API error:",
            e
        )

    return ""


def extract_article(url: str) -> str:
    clean_url = url.strip()

    # Indian Express special extraction
    if "indianexpress.com" in clean_url.lower():

        print("Indian Express detected.")

        ie_text = _fetch_indian_express_direct(
            clean_url
        )

        if ie_text:
            return ie_text

        print(
            "Indian Express API failed."
        )

    # Normal extraction for other websites
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

        if response.status_code == 200:

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            # JSON-LD extraction
            json_ld = _extract_from_json_ld(soup)

            if (
                json_ld
                and not _is_error_payload(json_ld)
            ):
                return _clean_text(json_ld)

            # Trafilatura extraction
            traf = trafilatura.extract(
                response.text,
                no_fallback=False
            )

            if (
                traf
                and not _is_error_payload(traf)
            ):
                return _clean_text(traf)

            # BeautifulSoup extraction
            dom = _extract_soup_paragraphs(
                soup
            )

            if (
                dom
                and not _is_error_payload(dom)
            ):
                return _clean_text(dom)

    except Exception as e:
        print(
            "Standard extraction error:",
            e
        )

    # Trafilatura fallback
    try:
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
                return _clean_text(traf)

    except Exception as e:
        print(
            "Trafilatura fallback error:",
            e
        )

    return ""