"""
Phishing URL & Content Detector - Streamlit app.

Combines two Random Forest models:
  - url_random_forest_model.pkl: 38 URL-structure features, no network call
    needed (see url_feature_extractor.py).
  - content_rf_model.pkl: 43 page-content features, requires fetching the
    page's HTML with SSRF protections (see content_feature_extractor.py).

Design principle: separate deterministic facts from probabilistic verdicts.
The "what we found on the page" panel (signals_panel.py) shows real,
checkable facts about the page regardless of how confident either model's
prediction is - so the person always has something trustworthy to look at,
even on borderline or low-confidence calls.

Run with:
    streamlit run app.py
"""

import pickle
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import streamlit as st

from url_feature_extractor import extract_url_features, FEATURE_COLUMNS as URL_FEATURE_COLUMNS
from content_feature_extractor import (
    safe_fetch, UnsafeURLError, extract_content_features,
    FEATURE_COLUMNS as CONTENT_FEATURE_COLUMNS,
)
from signals_panel import render_signals_panel
from site_intro import extract_site_intro, render_site_intro

URL_MODEL_PATH = Path(__file__).parent / "url_random_forest_model.pkl"
CONTENT_MODEL_PATH = Path(__file__).parent / "content_rf_model.pkl"

st.set_page_config(page_title="Phishing Detector", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_models():
    with open(URL_MODEL_PATH, "rb") as f:
        url_model = pickle.load(f)
    with open(CONTENT_MODEL_PATH, "rb") as f:
        content_model = pickle.load(f)
    return url_model, content_model


@st.cache_data
def get_url_importances(_model):
    return pd.Series(_model.feature_importances_, index=URL_FEATURE_COLUMNS).sort_values(ascending=False)


# Strip a leading "www." from the hostname before analysis, so that
# https://www.example.com/ and https://example.com/ are treated as the same
# address. See STRIP_WWW note in the README/report: this changes 7 of the 40
# URL features (URL_Length, Dot_Count, Domain_Length, Letter_Count,
# Subdomain_Count, URL_Entropy, Domain_Entropy), so it is kept as a single
# switch that can be turned off if the model is ever retrained on www-inclusive
# hostnames.
STRIP_WWW = True


def strip_www(url: str) -> str:
    """Remove a single leading 'www.' from the hostname, leaving the rest of
    the URL (scheme, port, path, query, fragment) untouched. Hostnames that
    merely start with the letters 'www' - e.g. 'wwwsomething.com' - are not
    affected, and a bare 'www.com' is left alone since removing it would
    destroy the hostname."""
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if not host.lower().startswith("www."):
        return url

    remainder = host[4:]
    if "." not in remainder:          # e.g. "www.com" -> "com": not a real host
        return url

    netloc = remainder
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username:
        credentials = parsed.username
        if parsed.password:
            credentials += f":{parsed.password}"
        netloc = f"{credentials}@{netloc}"

    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def normalize_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if not raw_url:
        return raw_url
    if "://" not in raw_url:
        raw_url = "http://" + raw_url
    if STRIP_WWW:
        raw_url = strip_www(raw_url)
    return raw_url


def phishing_probability(model, row_df):
    proba = model.predict_proba(row_df)[0]
    classes = list(model.classes_)
    return proba[classes.index(1)] if 1 in classes else None


def verdict_card(label, prob):
    if prob is None:
        st.info(f"**{label}**\n\nNot available")
        return
    if prob >= 0.66:
        st.error(f"**{label}**\n\n⚠️ Suspicious — {prob:.0%}")
    elif prob >= 0.4:
        st.warning(f"**{label}**\n\n❓ Uncertain — {prob:.0%}")
    else:
        st.success(f"**{label}**\n\n✅ Looks fine — {prob:.0%}")


st.title("🛡️ Phishing URL & Content Detector")
st.write(
    "Checks both the **web address structure** and the **page's actual content** "
    "for signs of phishing, and shows exactly what was found — not just a score."
)

with st.expander("How this works"):
    st.markdown(
        """
Two independent Random Forest models run on every check:

- **Web address** — 38 features computed from the URL text alone (length,
  entropy, subdomains, path structure, suspicious keywords, HTTPS usage).
  No network call needed.
- **Page content** — 43 features extracted from the page's actual HTML
  (forms, password fields, navigation structure, embedded content, etc).
  This requires fetching the page, with safety checks in place to prevent
  the request from being redirected to internal/private network addresses.

Both scores are shown separately rather than blended into one number —
if the web address looks fine but the content looks suspicious (or vice
versa), that disagreement is itself useful information. The **"what we
found on the page"** panel below shows the actual, checkable facts
extracted from the page itself, which stay meaningful regardless of how
confident either model's score is.
        """
    )

url_input = st.text_input("URL to check", placeholder="e.g. https://www.example.com")
check_clicked = st.button("Check URL", type="primary", use_container_width=True)

if check_clicked:
    if not url_input.strip():
        st.warning("Please enter a URL first.")
        st.stop()

    url = normalize_url(url_input)

    try:
        url_model, content_model = load_models()
    except FileNotFoundError as e:
        st.error(
            f"Model file not found: {e}. Make sure `url_random_forest_model.pkl` "
            "and `content_rf_model.pkl` are both in the same folder as this app."
        )
        st.stop()

    typed = url_input.strip()
    if url != typed and url != f"http://{typed}":
        st.caption(f"`{url}`  \u00b7  normalised from `{typed}`")
    else:
        st.caption(f"`{url}`")

    # --- URL-structure model (no network) ---
    url_feats = extract_url_features(url)
    url_row = pd.DataFrame([[url_feats[c] for c in URL_FEATURE_COLUMNS]], columns=URL_FEATURE_COLUMNS)
    url_phishing_prob = phishing_probability(url_model, url_row)

    # --- Content model (fetches the page, SSRF-protected) ---
    content_feats = None
    content_phishing_prob = None
    fetch_error = None
    site_intro = None

    with st.spinner("Fetching page content…"):
        try:
            response = safe_fetch(url)
        except UnsafeURLError as e:
            fetch_error = f"Refused to fetch this URL: {e}"
            response = None

        if response is not None:
            try:
                site_intro = extract_site_intro(response.text, url)
                content_feats = extract_content_features(response.text)
                content_row = pd.DataFrame(
                    [[content_feats[c] for c in CONTENT_FEATURE_COLUMNS]], columns=CONTENT_FEATURE_COLUMNS
                )
                content_phishing_prob = phishing_probability(content_model, content_row)
            except Exception as e:
                fetch_error = f"Could not analyze page content: {e}"
        elif fetch_error is None:
            fetch_error = "Could not fetch this page (it may be offline, blocking automated requests, or timed out)."

    # --- Verdict cards ---
    col1, col2 = st.columns(2)
    with col1:
        verdict_card("Web address", url_phishing_prob)
    with col2:
        verdict_card("Page content", content_phishing_prob)
        if content_phishing_prob is None and fetch_error:
            st.caption(fetch_error)

    # --- Site introduction: the page's own self-description ---
    if site_intro is not None:
        st.divider()
        render_site_intro(site_intro)

    # --- Signals panel: deterministic facts, shown whenever content was fetched ---
    if content_feats is not None:
        st.divider()
        render_signals_panel(content_feats)

    # --- URL model feature detail ---
    st.divider()
    with st.expander("URL structure features used for this prediction"):
        importances = get_url_importances(url_model)
        url_feat_df = pd.DataFrame(
            {
                "Feature": importances.index,
                "Model importance": [f"{v:.1%}" for v in importances.values],
                "This URL's value": [url_feats[c] for c in importances.index],
            }
        )
        st.dataframe(url_feat_df, use_container_width=True, hide_index=True)

    st.caption(
        "These are machine-learning predictions, not guarantees. Always exercise caution "
        "with unfamiliar links, especially ones asking for credentials or payment info."
    )

st.divider()
st.caption("Two Random Forest models: URL structure (38 features) + page content (43 features)")
