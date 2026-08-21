"""
Content-based feature extraction for the phishing detection model.

Fetches a URL's HTML and extracts the same 43 structural features used to
train content_rf_model.pkl (column order confirmed via model.feature_names_in_).

SECURITY: unlike the original notebook's bare `requests.get(url, verify=False)`,
this app fetches arbitrary user-supplied URLs from a server, which is a classic
SSRF (server-side request forgery) risk - a malicious input like
"http://169.254.169.254/latest/meta-data/" (a cloud metadata endpoint) or
"http://localhost:6379/" (an internal service) could otherwise make the
server attack itself or its internal network. safe_fetch() resolves the
hostname and rejects private/loopback/link-local/reserved IP ranges before
connecting, and re-validates on every redirect hop.
"""

import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

MAX_REDIRECTS = 5
FETCH_TIMEOUT = 6
MAX_CONTENT_BYTES = 3_000_000  # 3 MB cap - phishing pages are typically small


class UnsafeURLError(Exception):
    """Raised when a URL resolves to a disallowed (internal/private) address."""
    pass


def _is_public_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _assert_safe_host(hostname: str):
    """Resolve hostname and raise UnsafeURLError if it points anywhere internal."""
    if not hostname:
        raise UnsafeURLError("No hostname in URL.")
    try:
        addrs = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise UnsafeURLError(f"Could not resolve host: {hostname}")
    for family, _, _, _, sockaddr in addrs:
        ip_str = sockaddr[0]
        if not _is_public_ip(ip_str):
            raise UnsafeURLError(f"URL resolves to a non-public address ({ip_str}) - refusing to fetch.")


def safe_fetch(url: str):
    """
    Fetch a URL's HTML, validating that the URL (and every redirect hop)
    resolves to a public IP address. Returns the requests.Response, or None
    if the fetch failed for a non-security reason (timeout, connection error,
    non-HTML content, etc).

    Raises UnsafeURLError if the URL or any redirect target is internal.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError("Only http/https URLs are allowed.")

    current_url = url
    session = requests.Session()
    session.headers.update({"User-Agent": "phishing-detector-app/1.0"})

    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(current_url)
        _assert_safe_host(parsed.hostname)

        try:
            resp = session.get(
                current_url,
                timeout=FETCH_TIMEOUT,
                verify=True,
                allow_redirects=False,
                stream=True,
            )
        except requests.exceptions.RequestException:
            return None

        if resp.is_redirect or resp.is_permanent_redirect:
            next_url = resp.headers.get("Location")
            if not next_url:
                return None
            # Resolve relative redirects against the current URL
            from urllib.parse import urljoin
            current_url = urljoin(current_url, next_url)
            continue

        # Got a real (non-redirect) response - read up to the size cap
        content = b""
        for chunk in resp.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_CONTENT_BYTES:
                break
        resp._content = content
        return resp

    return None  # too many redirects


# --------------------------------------------------------------------------
# Feature extraction (mirrors Data_Collector_Note_Book_2.ipynb's create_vector)
# --------------------------------------------------------------------------

def has_title(soup): return 1 if soup.title and len(soup.title.text) > 0 else 0
def has_input(soup): return 1 if soup.find_all("input") else 0
def has_button(soup): return 1 if soup.find_all("button") else 0
def has_image(soup): return 1 if soup.find_all("image") else 0

def has_submit(soup):
    for i in soup.find_all("input"):
        if i.get("type") == "submit":
            return 1
    return 0

def has_link(soup): return 1 if soup.find_all("link") else 0

def has_password(soup):
    for i in soup.find_all("input"):
        if (i.get("type") or i.get("name") or i.get("id")) == "password":
            return 1
    return 0

def has_email_input(soup):
    for i in soup.find_all("input"):
        if (i.get("type") or i.get("id") or i.get("name")) == "email":
            return 1
    return 0

def has_hidden_element(soup):
    for i in soup.find_all("input"):
        if i.get("type") == "hidden":
            return 1
    return 0

def has_audio(soup): return 1 if soup.find_all("audio") else 0
def has_video(soup): return 1 if soup.find_all("video") else 0
def number_of_inputs(soup): return len(soup.find_all("input"))
def number_of_buttons(soup): return len(soup.find_all("button"))

def number_of_images(soup):
    meta_image_count = sum(1 for m in soup.find_all("meta") if m.get("type") or m.get("name") == "image")
    return len(soup.find_all("image")) + meta_image_count

def number_of_option(soup): return len(soup.find_all("option"))
def number_of_list(soup): return len(soup.find_all("li"))
def number_of_TH(soup): return len(soup.find_all("th"))
def number_of_TR(soup): return len(soup.find_all("tr"))
def number_of_href(soup): return sum(1 for l in soup.find_all("link") if l.get("href"))
def number_of_paragraph(soup): return len(soup.find_all("p"))
def number_of_script(soup): return len(soup.find_all("script"))
def length_of_title(soup): return len(soup.title.text) if soup.title else 0
def has_h1(soup): return 1 if soup.find_all("h1") else 0
def has_h2(soup): return 1 if soup.find_all("h2") else 0
def has_h3(soup): return 1 if soup.find_all("h3") else 0
def length_of_text(soup): return len(soup.get_text())
def number_of_clickable_button(soup): return sum(1 for b in soup.find_all("button") if b.get("type") == "button")
def number_of_a(soup): return len(soup.find_all("a"))
def number_of_img(soup): return len(soup.find_all("img"))
def number_of_div(soup): return len(soup.find_all("div"))
def number_of_figure(soup): return len(soup.find_all("figure"))
def has_footer(soup): return 1 if soup.find_all("footer") else 0
def has_form(soup): return 1 if soup.find_all("form") else 0
def has_text_area(soup): return 1 if soup.find_all("textarea") else 0
def has_iframe(soup): return 1 if soup.find_all("iframe") else 0

def has_text_input(soup):
    for i in soup.find_all("input"):
        if i.get("type") == "text":
            return 1
    return 0

def number_of_meta(soup): return len(soup.find_all("meta"))
def has_nav(soup): return 1 if soup.find_all("nav") else 0
def has_object(soup): return 1 if soup.find_all("object") else 0
def has_picture(soup): return 1 if soup.find_all("picture") else 0
def number_of_sources(soup): return len(soup.find_all("source"))
def number_of_span(soup): return len(soup.find_all("span"))
def number_of_table(soup): return len(soup.find_all("table"))


# Order matches content_rf_model.pkl's model.feature_names_in_ exactly.
FEATURE_COLUMNS = [
    "has_title", "has_input", "has_button", "has_image", "has_submit",
    "has_link", "has_password", "has_email_input", "has_hidden_element",
    "has_audio", "has_video", "number_of_inputs", "number_of_buttons",
    "number_of_images", "number_of_option", "number_of_list", "number_of_th",
    "number_of_tr", "number_of_href", "number_of_paragraph", "number_of_script",
    "length_of_title", "has_h1", "has_h2", "has_h3", "length_of_text",
    "number_of_clickable_button", "number_of_a", "number_of_img", "number_of_div",
    "number_of_figure", "has_footer", "has_form", "has_text_area", "has_iframe",
    "has_text_input", "number_of_meta", "has_nav", "has_object", "has_picture",
    "number_of_sources", "number_of_span", "number_of_table",
]

# Groups for the "what we found on the page" UI panel. Only the first two
# groups get red/orange risk styling - that framing is backed by general
# web-security convention (credential fields + hidden fields + iframes are
# well-established generic phishing tells), independent of this particular
# model's calibration. Everything else is shown as neutral, unjudged fact.
SIGNAL_GROUPS = [
    ("Asking for information", [
        ("has_password", "Password field", "ti-lock"),
        ("has_email_input", "Email field", "ti-mail"),
        ("has_hidden_element", "Hidden form field", "ti-eye-off"),
        ("has_form", "Form present", "ti-forms"),
    ]),
    ("Embedded content", [
        ("has_iframe", "Iframe embed", "ti-frame"),
        ("has_object", "Embedded object", "ti-box"),
    ]),
    ("Signs of a real, established site", [
        ("has_nav", "Navigation menu", "ti-menu-2"),
        ("has_footer", "Footer", "ti-layout-bottombar-expand"),
        ("has_h1", "Main heading (h1)", "ti-heading"),
        ("length_of_text", "Page text length", "ti-align-left"),
        ("number_of_meta", "Meta tags", "ti-tag"),
    ]),
    ("Page structure", [
        ("number_of_a", "Links", "ti-link"),
        ("number_of_img", "Images", "ti-photo"),
        ("number_of_script", "Scripts", "ti-code"),
        ("number_of_div", "Div elements", "ti-layout"),
    ]),
]


def extract_content_features(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    values = [
        has_title(soup), has_input(soup), has_button(soup), has_image(soup), has_submit(soup),
        has_link(soup), has_password(soup), has_email_input(soup), has_hidden_element(soup),
        has_audio(soup), has_video(soup), number_of_inputs(soup), number_of_buttons(soup),
        number_of_images(soup), number_of_option(soup), number_of_list(soup), number_of_TH(soup),
        number_of_TR(soup), number_of_href(soup), number_of_paragraph(soup), number_of_script(soup),
        length_of_title(soup), has_h1(soup), has_h2(soup), has_h3(soup), length_of_text(soup),
        number_of_clickable_button(soup), number_of_a(soup), number_of_img(soup), number_of_div(soup),
        number_of_figure(soup), has_footer(soup), has_form(soup), has_text_area(soup), has_iframe(soup),
        has_text_input(soup), number_of_meta(soup), has_nav(soup), has_object(soup), has_picture(soup),
        number_of_sources(soup), number_of_span(soup), number_of_table(soup),
    ]
    return dict(zip(FEATURE_COLUMNS, values))
