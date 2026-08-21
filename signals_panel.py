"""
Identifies the human-meaningful signals from the content model's 43 extracted
features, and renders them as a "what we found on the page" panel.

Only "Asking for information" and "Embedded content" get red/orange warning
styling when a signal is present - that's backed by general web-security
convention (credential fields, hidden inputs, and iframes are well-known
phishing tells on their own), not by trusting the model's calibration.
Everything else is shown as neutral, unjudged fact.

Usage in your app:
    from signals_panel import render_signals_panel
    render_signals_panel(content_features)   # content_features = dict from extract_content_features()
"""

import streamlit as st

# (feature_key, display_name, icon) - icon is a Tabler icon name, used if you
# want to add icons later; safe to ignore if not using Tabler.
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

# Groups whose "found" signals get warning styling - see module docstring.
RISKY_GROUPS = {"Asking for information", "Embedded content"}

# Boolean features (0/1) vs count features get displayed differently.
BOOLEAN_FEATURES = {
    "has_password", "has_email_input", "has_hidden_element", "has_form",
    "has_iframe", "has_object", "has_nav", "has_footer", "has_h1",
}


def get_signal_summary(content_features: dict) -> dict:
    """
    Pure-Python (no Streamlit) summary of which risky signals were found.
    Useful for testing, logging, or building the verdict text without
    needing to render anything.

    Returns:
        {
            "risky_found": ["Password field", "Hidden form field", ...],
            "risky_count": 2,
            "trust_markers_missing": ["Navigation menu", "Footer", ...],
        }
    """
    risky_found = []
    trust_markers_missing = []

    for group_name, items in SIGNAL_GROUPS:
        for feat_key, display_name, _icon in items:
            value = content_features.get(feat_key)
            if feat_key in BOOLEAN_FEATURES:
                found = value == 1
                if group_name in RISKY_GROUPS and found:
                    risky_found.append(display_name)
                if group_name == "Signs of a real, established site" and not found:
                    trust_markers_missing.append(display_name)

    return {
        "risky_found": risky_found,
        "risky_count": len(risky_found),
        "trust_markers_missing": trust_markers_missing,
    }


def render_signals_panel(content_features: dict):
    """
    Render the full "what we found on the page" panel in the current
    Streamlit app. Call this after you have a content_features dict from
    extract_content_features().
    """
    summary = get_signal_summary(content_features)

    st.subheader("What we found on the page")

    if summary["risky_count"] > 0:
        st.markdown(
            f"⚠️ **{summary['risky_count']} signal{'s' if summary['risky_count'] != 1 else ''} "
            f"commonly seen in phishing pages:** {', '.join(summary['risky_found'])}"
        )
    else:
        st.markdown("✅ No common credential-harvesting or embedding red flags found.")

    for group_name, items in SIGNAL_GROUPS:
        st.markdown(f"**{group_name}**")
        risky_group = group_name in RISKY_GROUPS

        for feat_key, display_name, _icon in items:
            value = content_features.get(feat_key)

            if feat_key in BOOLEAN_FEATURES:
                found = value == 1
                if found and risky_group:
                    st.markdown(f"- ⚠️ {display_name}: **found**")
                elif found:
                    st.markdown(f"- ✅ {display_name}: found")
                else:
                    st.markdown(f"- {display_name}: not found")
            else:
                # Count-style feature - show the raw number, no judgment.
                st.markdown(f"- {display_name}: **{value}**")

    with st.expander("View all raw content features"):
        import pandas as pd
        df = pd.DataFrame(
            {"Feature": list(content_features.keys()), "Value": list(content_features.values())}
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
