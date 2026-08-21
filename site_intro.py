"""
Builds a short, plain-language introduction to the page that was analysed, so
the person gets an immediate sense of "what is this site?" before reading the
model verdicts.

DESIGN NOTE - why this is framed as "what the page says about itself":

Everything here is self-reported by the page: its <title>, its meta
description, its Open Graph tags. A phishing page impersonating a bank will
happily set <title>PayPal - Log In</title>. So this panel is deliberately NOT
presented as verification or as evidence the site is genuine - it is presented
as the page's own claim about itself, which is a different and honest thing.
That framing matters: a summary panel that reads like an endorsement would
actively help a convincing phishing page, which is the opposite of what this
app is for.

All extracted text is rendered through st.text()/st.code() rather than
st.markdown(). Page content is attacker-controlled, and html.escape() alone is
NOT sufficient here: it neutralises < > &, but Streamlit renders markdown, so a
<title> of "[Click here](http://evil.tld)" would still become a live link
inside this app's own interface. Using the non-markdown text primitives avoids
that class of injection entirely.

Usage:
    from site_intro import extract_site_intro, render_site_intro
    intro = extract_site_intro(response.text, url)
    render_site_intro(intro)
"""

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

# Meta names/properties worth reading, in priority order.
_DESCRIPTION_KEYS = [
    ("meta", {"name": "description"}),
    ("meta", {"property": "og:description"}),
    ("meta", {"name": "twitter:description"}),
]
_TITLE_KEYS = [
    ("meta", {"property": "og:title"}),
    ("meta", {"name": "twitter:title"}),
]
_SITE_NAME_KEYS = [
    ("meta", {"property": "og:site_name"}),
    ("meta", {"name": "application-name"}),
]

MAX_TITLE_CHARS = 120
MAX_DESCRIPTION_CHARS = 300


def _clean(text):
    """Collapse whitespace and trim. Returns None for empty/absent text."""
    if not text:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text or None


def _truncate(text, limit):
    if text is None:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _first_meta(soup, keys):
    for tag_name, attrs in keys:
        tag = soup.find(tag_name, attrs=attrs)
        if tag:
            value = _clean(tag.get("content"))
            if value:
                return value
    return None


def extract_site_intro(html_text: str, url: str) -> dict:
    """
    Pull the page's self-description metadata into a small dict. Pure function - no
    Streamlit, no network - so it can be unit-tested directly.

    Returns keys: domain, title, site_name, description, language,
    heading, has_any (bool - whether anything useful was found at all).
    """
    soup = BeautifulSoup(html_text, "html.parser")

    title = _first_meta(soup, _TITLE_KEYS)
    if not title and soup.title:
        title = _clean(soup.title.get_text())

    description = _first_meta(soup, _DESCRIPTION_KEYS)
    site_name = _first_meta(soup, _SITE_NAME_KEYS)

    h1 = soup.find("h1")
    heading = _clean(h1.get_text()) if h1 else None

    html_tag = soup.find("html")
    language = _clean(html_tag.get("lang")) if html_tag else None

    domain = _clean(urlparse(url).netloc) or "unknown"

    intro = {
        "domain": domain,
        "title": _truncate(title, MAX_TITLE_CHARS),
        "site_name": _truncate(site_name, MAX_TITLE_CHARS),
        "description": _truncate(description, MAX_DESCRIPTION_CHARS),
        "language": language,
        "heading": _truncate(heading, MAX_TITLE_CHARS),
    }
    intro["has_any"] = any(
        intro[k] for k in ("title", "site_name", "description", "heading")
    )
    return intro


def build_intro_sentence(intro: dict) -> str:
    """
    One-line plain-text summary, e.g.
    'youtube.com presents itself as "YouTube".'
    Useful for logging or a compact display; returns a fallback if the page
    supplied no usable metadata.
    """
    name = intro.get("site_name") or intro.get("title") or intro.get("heading")
    if not name:
        return f"{intro['domain']} did not provide a title or description."
    return f'{intro["domain"]} presents itself as "{name}".'


def render_site_intro(intro: dict):
    """Render the introduction panel in the current Streamlit app.

    Uses st.text() for all page-derived strings: these are attacker-controlled
    and must not be passed through st.markdown(), which would render any
    markdown they contain (bold, headings, and - most importantly - clickable
    links) inside this app's own UI.
    """
    import streamlit as st

    st.subheader("About this site")
    st.caption(
        "Taken from the page's own title and description tags \u2014 this is what "
        "the page claims about itself, not an independent check of who runs it."
    )

    if not intro["has_any"]:
        st.text(f"{intro['domain']} did not provide a title or description.")
        st.caption(
            "That is common for parked domains, redirect pages, and pages that "
            "build their content with JavaScript."
        )
        return

    headline = intro.get("site_name") or intro.get("title") or intro.get("heading")
    st.text(headline)

    if intro.get("description"):
        st.text(intro["description"])

    st.markdown("**Details**")
    st.text(f"Domain:            {intro['domain']}")
    if intro.get("title") and intro["title"] != headline:
        st.text(f"Page title:        {intro['title']}")
    if intro.get("heading") and intro["heading"] != headline:
        st.text(f"Main heading:      {intro['heading']}")
    if intro.get("language"):
        st.text(f"Declared language: {intro['language']}")
