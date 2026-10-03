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
    """
    Try extracting articleBody from JSON-LD metadata.
    """

    for script in soup.find_all(
        "script",
        type="application/ld+json"
    ):
        if not script.string:
            continue

        try:
            data = json.loads(
                script.string.strip()
            )

            items = (
                data
                if isinstance(data, list)
                else [data]
            )

            for item in items:

                if not isinstance(item, dict):
                    continue

                # Direct articleBody
                if item.get("articleBody"):
                    return item["articleBody"].strip()

                # @graph articleBody
                for graph_item in item.get(
                    "@graph",
                    []
                ):

                    if (
                        isinstance(
                            graph_item,
                            dict
                        )
                        and graph_item.get(
                            "articleBody"
                        )
                    ):
                        return graph_item[
                            "articleBody"
                        ].strip()

        except Exception:
            continue

    return ""


def _extract_soup_paragraphs(
    soup: BeautifulSoup
) -> str:
    """
    Extract article paragraphs using
    common HTML containers.
    """

    # Remove unwanted elements
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

    # Try known article containers
    container = soup.find(
        "div",
        class_=re.compile(
            r"(story-details|full-details|"
            r"story_details|ie-content|"
            r"art-content|wp-block-post-content|"
            r"article-body|storycontent)",
            re.IGNORECASE,
        ),
    )

    # Try article tag
    if not container:
        container = soup.find(
            "article"
        )

    # Try main tag
    if not container:
        container = soup.find(
            "main"
        )

    target = (
        container
        if container
        else soup
    )

    paragraphs = []

    for p in target.find_all("p"):

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

        return "\n\n".join(
            paragraphs
        )

    return ""


def _fetch_indian_express_direct(
    url: str
) -> str:
    """
    Extract Indian Express article directly
    from the webpage.

    This does NOT assume that the number
    at the end of the URL is a WordPress
    post ID.
    """

    print(
        "Trying Indian Express direct extraction..."
    )

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
                "Accept-Language": (
                    "en-US,en;q=0.9"
                ),
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        )

        print(
            "Indian Express page status:",
            response.status_code
        )

        if response.status_code != 200:
            return ""

        html = response.text

        if not html:
            return ""

        # Check if we received an error page
        if _is_error_payload(html):
            print(
                "Indian Express returned an error page."
            )
            return ""

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # --------------------------------------------------
        # METHOD 1: JSON-LD
        # --------------------------------------------------

        json_ld = _extract_from_json_ld(
            soup
        )

        if (
            json_ld
            and not _is_error_payload(
                json_ld
            )
        ):

            print(
                "Indian Express JSON-LD "
                "extraction successful."
            )

            return _clean_text(
                json_ld
            )

        # --------------------------------------------------
        # METHOD 2: Known Indian Express containers
        # --------------------------------------------------

        containers = soup.find_all(
            "div",
            class_=re.compile(
                r"(story-details|full-details|"
                r"ie-content|article-body|"
                r"storycontent|story_details|"
                r"art-content)",
                re.IGNORECASE,
            ),
        )

        for container in containers:

            paragraphs = []

            for p in container.find_all(
                "p"
            ):

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

                paragraphs.append(
                    text
                )

            if paragraphs:

                article_text = (
                    "\n\n".join(
                        paragraphs
                    )
                )

                article_text = _clean_text(
                    article_text
                )

                if not _is_error_payload(
                    article_text
                ):

                    print(
                        "Indian Express HTML "
                        "container extraction successful."
                    )

                    return article_text

        # --------------------------------------------------
        # METHOD 3: <article> / <main>
        # --------------------------------------------------

        container = soup.find(
            "article"
        )

        if not container:
            container = soup.find(
                "main"
            )

        if container:

            paragraphs = []

            for p in container.find_all(
                "p"
            ):

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

                paragraphs.append(
                    text
                )

            if paragraphs:

                article_text = (
                    "\n\n".join(
                        paragraphs
                    )
                )

                article_text = _clean_text(
                    article_text
                )

                if not _is_error_payload(
                    article_text
                ):

                    print(
                        "Indian Express article/main "
                        "extraction successful."
                    )

                    return article_text

        # --------------------------------------------------
        # METHOD 4: General paragraph extraction
        # --------------------------------------------------

        dom = _extract_soup_paragraphs(
            soup
        )

        if (
            dom
            and not _is_error_payload(dom)
        ):

            print(
                "Indian Express general "
                "paragraph extraction successful."
            )

            return _clean_text(
                dom
            )

    except Exception as e:

        print(
            "Indian Express direct "
            "extraction error:",
            e
        )

    return ""


def extract_article(url: str) -> str:
    """
    Main article extraction function.

    Uses a special extractor for Indian Express
    and normal extraction methods for other sites.
    """

    clean_url = url.strip()

    # ==================================================
    # INDIAN EXPRESS SPECIAL EXTRACTION
    # ==================================================

    if (
        "indianexpress.com"
        in clean_url.lower()
    ):

        print(
            "Indian Express detected."
        )

        ie_text = (
            _fetch_indian_express_direct(
                clean_url
            )
        )

        if ie_text:
            return ie_text

        print(
            "Indian Express direct "
            "extraction failed."
        )

    # ==================================================
    # NORMAL EXTRACTION
    # ==================================================

    try:

        response = curl_requests.get(
            clean_url,
            impersonate="chrome120",
            timeout=10,
            headers={
                "Referer": (
                    "https://www.google.com/"
                ),
                "Accept-Language": (
                    "en-US,en;q=0.9"
                ),
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

            # ------------------------------------------
            # JSON-LD extraction
            # ------------------------------------------

            json_ld = (
                _extract_from_json_ld(
                    soup
                )
            )

            if (
                json_ld
                and not _is_error_payload(
                    json_ld
                )
            ):

                return _clean_text(
                    json_ld
                )

            # ------------------------------------------
            # Trafilatura extraction
            # ------------------------------------------

            traf = trafilatura.extract(
                response.text,
                no_fallback=False
            )

            if (
                traf
                and not _is_error_payload(
                    traf
                )
            ):

                return _clean_text(
                    traf
                )

            # ------------------------------------------
            # BeautifulSoup extraction
            # ------------------------------------------

            dom = (
                _extract_soup_paragraphs(
                    soup
                )
            )

            if (
                dom
                and not _is_error_payload(
                    dom
                )
            ):

                return _clean_text(
                    dom
                )

    except Exception as e:

        print(
            "Standard extraction error:",
            e
        )

    # ==================================================
    # TRAFILATURA FALLBACK
    # ==================================================

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
                and not _is_error_payload(
                    traf
                )
            ):

                return _clean_text(
                    traf
                )

    except Exception as e:

        print(
            "Trafilatura fallback error:",
            e
        )

    return ""