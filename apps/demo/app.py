"""Philosophical Presupposition Finder: a web demo of the presupp classifier.

Run from the repository root, in the demo environment (see apps/demo/README.md):

    python apps/demo/app.py

The model is read from PRESUPP_MODEL_DIR (default models/presupp-roberta-v3). On the Hugging Face
Space it's downloaded from the private repo in PRESUPP_MODEL_REPO, using the HF_TOKEN secret.
"""

import base64
import os
import warnings
from html import escape
from pathlib import Path

import gradio as gr
from display import (
    CSS,
    FLAMES_JS,
    FRAME_CSS,
    how_it_reads_markdown,
    reliability_markdown,
    results_html,
)
from examples import EXAMPLES

from presupp.inference import RELEASE_FILE, Classifier

DEFAULT_MODEL_DIR = Path("models") / "presupp-roberta-v3"
warnings.filterwarnings("ignore", r"The '(theme|css|head)' parameter in the Blocks constructor",
                        DeprecationWarning)
warnings.filterwarnings("ignore", r"Setting 'api_name=False'", DeprecationWarning)

FONTS = ("https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600"
         "&family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=Anton&display=swap")
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link rel="stylesheet" href="{FONTS}"><style>{FRAME_CSS}</style>'
        f"<script>{FLAMES_JS}</script>")

TITLE = ('<header class="pf-top"><div class="pf-blaze">'
         '<canvas class="pf-flames" aria-hidden="true"></canvas>'
         '<h1><span class="pf-pilcrow" aria-hidden="true">¶</span>'
         "Philosophical Presupposition Finder</h1></div></header>")
INTRO = """
<p class="pf-lede">Hello! Welcome to my Philosophical Presupposition Classifier Demo!
Paste an article, a speech or a policy paper to see what presuppositions my model can infer.
My model was trained in English only and on a very limited amount of data.
The labels it assigns will be at best, an interpretation of the author's beliefs
or positions, and at worst just completely wrong.
It reads and applies labels at the sentence level and marks where it believes an argument
can be associated with a philosophical tradition.</p>
<p class="pf-caveat">It learned from about a thousand passages labelled all by me, and it
finds some positions far more reliably than others, so <u>be speculative of every label</u>
(seriously).</p>
"""
GITHUB = "https://github.com/rafaelrdrgz/presupp/tree/main"
LINKEDIN = "https://www.linkedin.com/in/rafael-morris/"
GITHUB_ICON = (
    "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49"
    "-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58"
    " 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31"
    "-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09"
    " 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87"
    " 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0"
    " 0016 8c0-4.42-3.58-8-8-8z")
LINKEDIN_ICON = (
    "M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136"
    " 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37"
    " 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063"
    " 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782"
    " 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792"
    " 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z")


BLOG = "https://substack.com/home/post/p-222763702"
BLOG_ICON = base64.b64encode((Path(__file__).parent / "blog_icon.png").read_bytes()).decode()


def svg(path: str, size: int) -> str:
    return f'<svg viewBox="0 0 {size} {size}" aria-hidden="true"><path d="{path}"/></svg>'


def link(href: str, label: str, icon: str) -> str:
    return f'<a href="{href}" target="_blank" rel="noopener">{icon}{label}</a>'


LINKS = ('<p class="pf-links">' + link(GITHUB, "GitHub", svg(GITHUB_ICON, 16))
         + link(LINKEDIN, "LinkedIn", svg(LINKEDIN_ICON, 24))
         + link(BLOG, "Blog Post", f'<img src="data:image/png;base64,{BLOG_ICON}" alt="">')
         + "</p>")
PRIVACY = '<p class="pf-privacy">Text you paste isn’t stored or used beyond this demo.</p>'

OGL = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
FOOTER = ('<p class="pf-footer">The examples are from GOV.UK: '
          + ", ".join(f'<a href="{escape(e["url"])}">{escape(e["source"])}</a>' for e in EXAMPLES)
          + f'. Public sector information licensed under the <a href="{OGL}">Open '
          "Government Licence v3.0</a>.</p>")

# after a run, scroll the results into view if they're off screen
SCROLL = """() => {
  const results = document.getElementById("pf-results");
  const top = results ? results.getBoundingClientRect().top : 0;
  if (results && (top < 0 || top > 0.6 * innerHeight)) results.scrollIntoView({block: "start",
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
}"""

INK, RAISED, RULE = "#0f1730", "#16203f", "#27335a"
VELLUM, FADED, LINK = "#ece6d9", "#a4adc2", "#8fc3ea"
RUBRIC, RUBRIC_DEEP = "#c12440", "#a11d37"


def theme() -> gr.themes.Base:
    """Always dark: each colour is set for gradio's light and dark modes."""
    base = gr.themes.Base(font=["Schibsted Grotesk", "system-ui", "sans-serif"],
                          font_mono=["ui-monospace", "monospace"])
    values = {
        "body_background_fill": INK, "body_text_color": VELLUM, "body_text_color_subdued": FADED,
        "background_fill_primary": INK, "background_fill_secondary": RAISED,
        "border_color_primary": RULE, "border_color_accent": LINK, "color_accent": RUBRIC,
        "color_accent_soft": RAISED, "link_text_color": LINK, "link_text_color_hover": VELLUM,
        "link_text_color_visited": LINK, "link_text_color_active": VELLUM,
        "block_background_fill": "transparent", "block_border_width": "0px",
        "block_border_color": RULE, "block_shadow": "none", "block_label_text_color": FADED,
        "block_title_text_color": VELLUM, "block_info_text_color": FADED,
        "input_background_fill": RAISED, "input_background_fill_focus": RAISED,
        "input_background_fill_hover": RAISED, "input_border_color": RULE,
        "input_border_color_hover": RULE, "input_border_color_focus": LINK,
        "input_border_width": "1px", "input_placeholder_color": "#8a93ab",
        "input_shadow": "none", "input_shadow_focus": "none", "input_radius": "6px",
        "button_primary_background_fill": RUBRIC_DEEP,
        "button_primary_background_fill_hover": RUBRIC,
        "button_primary_border_color": RUBRIC_DEEP, "button_primary_border_color_hover": RUBRIC,
        "button_primary_text_color": "#fbf2ee", "button_primary_text_color_hover": "#fbf2ee",
        "button_primary_shadow": "none", "button_primary_shadow_hover": "none",
        "button_secondary_background_fill": "transparent",
        "button_secondary_background_fill_hover": RAISED,
        "button_secondary_border_color": RULE, "button_secondary_border_color_hover": FADED,
        "button_secondary_text_color": VELLUM, "button_secondary_text_color_hover": VELLUM,
        "button_secondary_shadow": "none", "button_secondary_shadow_hover": "none",
        "loader_color": RUBRIC, "panel_background_fill": INK, "panel_border_color": RULE,
        "accordion_text_color": VELLUM, "table_border_color": RULE,
        "table_even_background_fill": INK, "table_odd_background_fill": RAISED,
        "table_text_color": VELLUM, "code_background_fill": RAISED, "shadow_spread": "0px",
    }
    both = {}
    for name, value in values.items():
        both[name] = value
        if hasattr(base, name + "_dark"):
            both[name + "_dark"] = value
    return base.set(**both)


def load_classifier() -> Classifier:
    repo = os.environ.get("PRESUPP_MODEL_REPO")
    if repo:
        from huggingface_hub import snapshot_download

        return Classifier.load(snapshot_download(repo, token=os.environ.get("HF_TOKEN")))
    directory = Path(os.environ.get("PRESUPP_MODEL_DIR", DEFAULT_MODEL_DIR))
    if not (directory / RELEASE_FILE).exists():
        raise SystemExit(f"No released model in {directory}. Run python -m presupp.release, or "
                         "set PRESUPP_MODEL_DIR to a release folder.")
    return Classifier.load(directory)


def build(classifier: Classifier) -> gr.Blocks:
    def analyse(text: str | None, progress=gr.Progress()) -> str:  # noqa: B008 (gradio's way)
        def report(done: int, total: int, stage: str) -> None:
            progress((done, total), desc=f"Reading {stage}", unit=stage)

        reading, results = classifier.classify(text or "", report)
        return results_html(reading, results)

    # layout is in CSS: two columns on wide screens, stacked in this order on narrow ones
    with gr.Blocks(title="Philosophical Presupposition Finder", theme=theme(), css=CSS,
                   head=HEAD, analytics_enabled=False, fill_width=False) as demo:
        gr.HTML(TITLE, padding=False)
        with gr.Row(elem_id="pf-layout", equal_height=False):
            with gr.Column(elem_id="pf-intro"):
                gr.HTML(INTRO, padding=False)
                gr.HTML(LINKS, padding=False)
            with gr.Column(elem_id="pf-work"):
                text = gr.Textbox(label="Text to read", show_label=False, lines=10, max_lines=24,
                                  container=False, elem_id="pf-input",
                                  placeholder="Paste text here... :)")
                gr.HTML(PRIVACY, padding=False, elem_id="pf-privacy")
                with gr.Row(elem_id="pf-actions"):
                    run = gr.Button("Find presuppositions", variant="primary")
                    clear = gr.Button("Clear")
                with gr.Row(elem_id="pf-examples"):
                    gr.Markdown("Or try a GOV.UK text:", elem_classes="pf-examples-label",
                                padding=False)
                    examples = [gr.Button(e["label"], size="sm") for e in EXAMPLES]
                output = gr.HTML(padding=False)
            with gr.Column(elem_id="pf-about"):
                with gr.Accordion("How it reads your text", open=False):
                    gr.Markdown(how_it_reads_markdown(classifier.release))
                with gr.Accordion("How reliable it is", open=False):
                    gr.Markdown(reliability_markdown(classifier.release))
                gr.HTML(FOOTER, padding=False)

        # only /analyse is exposed in the API
        run.click(analyse, text, output, api_name="analyse").then(None, js=SCROLL)
        for button, example in zip(examples, EXAMPLES, strict=True):
            button.click(lambda e=example: e["text"], None, text, api_name=False).then(
                analyse, text, output, api_name=False).then(None, js=SCROLL)
        clear.click(lambda: ("", ""), None, [text, output], api_name=False)
    return demo


def asgi_app():
    """The demo as an ASGI app, for hosts that serve one (modal_app.py)."""
    from fastapi import FastAPI

    return gr.mount_gradio_app(FastAPI(), build(load_classifier()).queue(max_size=16), path="/")


if __name__ == "__main__":
    build(load_classifier()).queue(max_size=16).launch()
