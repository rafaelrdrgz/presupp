"""HTML for the demo's results. Kept out of app.py so tests don't need gradio."""

from collections import Counter
from html import escape

from presupp import reading
from presupp.inference import PassageResult

# cool colours for theoretical philosophy, warm for practical
DOMAINS = {
    "Metaphysical": ("Metaphysical", "metaphysical"),
    "Epistemological": ("Epistemological", "epistemological"),
    "Ethical": ("Ethical", "ethical"),
    "Political-Philosophical": ("Political", "political"),
}

SKIPPED = {
    reading.FRAGMENT: "too short to be prose",
    reading.BOILERPLATE: "site text",
    reading.REPEAT: "repeated",
    reading.NOT_PROSE: "not sentences",
    reading.NOT_ENGLISH: "not English prose",
}
SKIPPED_LISTED = 40  # lines listed in a folded run; the rest are counted
LINKED_REFERENCES = 30  # hover CSS is generated for this many refs per passage

PROBLEMS = {
    reading.EMPTY: "Paste some text first.",
    reading.LINK: "That’s a link. Paste the article’s text instead: the app doesn’t open web "
                  "pages.",
    reading.ONLY_NOT_ENGLISH: "This doesn’t look like English. The model only learned from "
                              "English writing, so it hasn’t read it.",
    reading.TOO_SHORT: "There isn’t enough prose here to read. Paste at least a full paragraph.",
}

# Doom-style fire behind the title
FLAMES_JS = """
(() => {
  const CELL = 5, SHADES = 37;
  const STOPS = [[0, 0, 0, 0], [60, 8, 8, 0.4], [130, 16, 12, 0.8], [200, 36, 16, 0.95],
                 [240, 90, 22, 1], [255, 150, 34, 1], [255, 205, 70, 1], [255, 242, 180, 1]];
  const palette = Array.from({length: SHADES}, (_, i) => {
    const t = i / (SHADES - 1) * (STOPS.length - 1), k = Math.min(t | 0, STOPS.length - 2);
    return STOPS[k].map((v, j) => v + (STOPS[k + 1][j] - v) * (t - k));
  });
  function blaze(canvas) {
    const ctx = canvas.getContext("2d");
    let w, h, hottest, heat, image;
    const size = () => {
      w = Math.max(8, Math.round(canvas.clientWidth / CELL));
      h = Math.max(8, Math.round(canvas.clientHeight / CELL));
      canvas.width = w;
      canvas.height = h;
      hottest = Math.max(4, Math.round(h * 0.4));
      heat = new Uint8Array(w * h);
      image = ctx.createImageData(w, h);
    };
    const spread = () => {
      heat.fill(hottest, (h - 1) * w);
      for (let y = 1; y < h; y++) for (let x = 0; x < w; x++) {
        const from = y * w + x, r = Math.round(Math.random() * 3) & 3;
        const to = from - w - r + 1;
        if (to >= 0) heat[to] = Math.max(0, heat[from] - (r & 1));
      }
    };
    const paint = () => {
      for (let i = 0; i < heat.length; i++) {
        const [r, g, b, a] = palette[Math.round(heat[i] / hottest * (SHADES - 1))];
        image.data[i * 4] = r;
        image.data[i * 4 + 1] = g;
        image.data[i * 4 + 2] = b;
        image.data[i * 4 + 3] = a * 255;
      }
      ctx.putImageData(image, 0, 0);
    };
    size();
    new ResizeObserver(size).observe(canvas);
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      for (let i = 0; i < h * 3; i++) spread();
      paint();
      return;
    }
    let last = 0;
    const frame = (t) => {
      if (t - last > 45) { spread(); paint(); last = t; }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }
  const wait = setInterval(() => {  // canvas isn't in the DOM yet
    const canvas = document.querySelector(".pf-flames");
    if (canvas) { clearInterval(wait); blaze(canvas); }
  }, 100);
})();
"""

# goes in the page head: gradio's css= scoping breaks media queries on .gradio-container
FRAME_CSS = """
:root { color-scheme: dark; }
.gradio-container { width: 100% !important; max-width: 1080px !important;
  margin: 0 auto !important; padding: 36px 38px 64px !important; overflow: visible !important; }
.gradio-container > main.app { padding: 0 !important; max-width: none !important; }
@media (min-width: 1180px) {
  .gradio-container { max-width: 1720px !important; padding: 40px 38px 72px !important; }
}
@media (max-width: 560px) {
  .gradio-container { padding: 28px 16px 56px !important; }
}
"""

CSS = """
.gradio-container {
  --pf-ink: #0f1730; --pf-ink-raised: #16203f; --pf-rule: #27335a; --pf-track: #1e2a4d;
  --pf-vellum: #ece6d9; --pf-faded: #a4adc2; --pf-link: #8fc3ea; --pf-rubric: #c12440;
  --pf-glow: rgba(236, 230, 217, 0.1);
  --pf-metaphysical: #7e6cd9; --pf-epistemological: #31a0d0; --pf-ethical: #c12440;
  --pf-political: #e66941;
  --pf-serif: "Spectral", Georgia, "Times New Roman", serif;
  --pf-sans: "Schibsted Grotesk", system-ui, sans-serif;
}
footer { display: none !important; }
/* gradio's overflow:auto shows empty scroll bars */
.block:not(#pf-input) { overflow: visible !important; }
#pf-layout { flex-direction: column !important; flex-wrap: nowrap !important; gap: 0; }
#pf-layout > .column { width: 100%; min-width: 0 !important; }
#pf-work { gap: 16px; }
.gradio-container :focus-visible { outline: 2px solid var(--pf-link); outline-offset: 2px; }

/* title and intro */
.pf-top { text-align: center; margin: 0; padding: 16px 0 30px; }
.pf-top::after { content: ""; display: block; width: calc(100% - 100px); height: 2px;
  margin: 22px auto 0; border-radius: 1px; background: var(--pf-rubric); }
.pf-top h1 { font: 400 clamp(2.2rem, 4.4vw, 3.2rem)/1.05 Impact, Anton, "Arial Narrow",
  sans-serif; margin: 0; position: relative; z-index: 1;
  letter-spacing: 0.01em; color: var(--pf-vellum);
  text-shadow: 0 0 3px #2a0505, 0 3px 0 #2a0505, 0 0 22px rgba(20, 0, 0, 0.75); }
/* fire canvas */
.pf-blaze { position: relative; }
.pf-flames { position: absolute; left: 50%; bottom: -0.35em; transform: translateX(-50%);
  width: min(100%, 60rem); height: calc(100% + 2.9rem); z-index: 0; pointer-events: none;
  image-rendering: pixelated;
  -webkit-mask-image: linear-gradient(to right, transparent, #000 18%, #000 82%, transparent),
    linear-gradient(to top, transparent, #000 25%, #000 70%, transparent);
  -webkit-mask-composite: source-in;
  mask-image: linear-gradient(to right, transparent, #000 18%, #000 82%, transparent),
    linear-gradient(to top, transparent, #000 25%, #000 70%, transparent);
  mask-composite: intersect; }
.pf-pilcrow { color: var(--pf-rubric); font-weight: 500; margin-right: 0.14em; }
.pf-lede { font: 400 1.1875rem/1.55 var(--pf-serif); color: var(--pf-vellum); margin: 0 0 14px;
  max-width: 40em; }
.pf-caveat { font: 400 0.9375rem/1.55 var(--pf-sans); color: var(--pf-faded); margin: 0 0 24px;
  max-width: 44em; }
.pf-caveat u { color: inherit !important; font-weight: inherit !important;
  text-decoration-thickness: 1px; text-underline-offset: 3px; }

/* input */
#pf-input textarea { font: 400 1.0625rem/1.6 var(--pf-serif) !important;
  padding: 16px 18px !important; }
#pf-privacy { margin-top: -10px; }
.pf-privacy { margin: 0; text-align: right; font: 400 0.8125rem/1.4 var(--pf-sans);
  color: var(--pf-faded); }
#pf-actions { gap: 10px; align-items: center; flex-wrap: nowrap; }
#pf-actions button { flex: 0 0 auto !important; min-width: 0 !important; padding-inline: 20px;
  border-radius: 0 !important; }
#pf-examples { gap: 4px 16px; align-items: baseline; flex-wrap: wrap; }
#pf-examples > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
#pf-examples .pf-examples-label p { font: 400 0.9375rem var(--pf-sans); color: var(--pf-faded);
  margin: 0; }
#pf-examples button { background: none !important; border: 0 !important;
  box-shadow: none !important; padding: 2px 0 !important; color: var(--pf-link) !important;
  font: 500 0.9375rem var(--pf-sans) !important; text-decoration: underline;
  text-decoration-color: color-mix(in srgb, var(--pf-link) 45%, transparent);
  text-underline-offset: 3px; }
#pf-examples button:hover { text-decoration-color: var(--pf-link); }

/* results */
.pf { font: 400 0.9375rem/1.5 var(--pf-sans); color: var(--pf-vellum); margin-top: 12px; }
.pf p { margin: 0; }
.pf-message { max-width: 46em; font: 400 1.125rem/1.5 var(--pf-serif); }
.pf-summary { margin: 0 0 40px; border-top: 1px solid var(--pf-rule); padding-top: 26px; }
.pf-total { font: 400 1.25rem/1.4 var(--pf-serif); margin-bottom: 18px !important; }
.pf-bars { display: grid; grid-template-columns: max-content 1fr max-content; gap: 8px 14px;
  align-items: center; max-width: 38rem; margin-bottom: 16px; }
.pf-bar-name { display: flex; align-items: center; gap: 8px; }
.pf-swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }
.pf-track { height: 8px; border-radius: 4px; background: var(--pf-track); position: relative; }
.pf-track > span { position: absolute; inset: 0 auto 0 0; border-radius: 4px; }
.pf-count { color: var(--pf-faded); font-variant-numeric: tabular-nums; text-align: right; }
.pf-rest { color: var(--pf-faded); max-width: 46em; }
.pf-rest + .pf-rest { margin-top: 4px !important; }

.metaphysical { background: var(--pf-metaphysical); }
.epistemological { background: var(--pf-epistemological); }
.ethical { background: var(--pf-ethical); }
.political { background: var(--pf-political); }

.pf-passage { display: grid; grid-template-columns: 11rem minmax(0, 54rem); column-gap: 1.5rem;
  margin-bottom: 30px; }
.pf-gloss { grid-column: 1; padding-top: 3px; text-align: right; }
.pf-body { grid-column: 2; position: relative; padding-left: 22px; }
.pf-rules { position: absolute; left: 0; top: 4px; bottom: 4px; display: flex; gap: 2px; }
.pf-rules > span { width: 3px; border-radius: 2px; }
.pf-rules .pf-dashed { background: none; border-left: 2px dashed var(--pf-rule); width: 0; }
.pf-body > p { font: 400 1.0625rem/1.7 var(--pf-serif); color: var(--pf-vellum);
  max-width: 42em; }
.pf-body > p + p { margin-top: 0.8em; }
.pf-quiet .pf-body > p { color: var(--pf-faded); }
.pf-domain { font-weight: 600; display: flex; justify-content: flex-end; align-items: center;
  gap: 7px; }
.pf-domain + .pf-cats { margin-bottom: 8px; }
.pf-cats { list-style: none; margin: 2px 0 0; padding: 0; }
.pf-cats li { color: var(--pf-vellum); line-height: 1.35; margin-top: 3px; padding: 0 2px;
  border-radius: 3px; }
.pf-gloss-note { color: var(--pf-faded); font-size: 0.8125rem; line-height: 1.4;
  margin-top: 6px !important; }

/* evidence sentences */
.pf-s { text-decoration-line: underline; text-decoration-thickness: 2px;
  text-underline-offset: 0.3em; text-decoration-skip-ink: auto; border-radius: 3px;
  transition: background-color 0.15s; }
.pf-u-metaphysical { text-decoration-color: var(--pf-metaphysical); }
.pf-u-epistemological { text-decoration-color: var(--pf-epistemological); }
.pf-u-ethical { text-decoration-color: var(--pf-ethical); }
.pf-u-political { text-decoration-color: var(--pf-political); }
.pf-ref { font: 600 0.62em/0 var(--pf-sans); color: var(--pf-faded); margin-left: 0.15em;
  font-variant-numeric: tabular-nums; }
.pf-refs { color: var(--pf-faded); font-size: 0.75rem; font-weight: 500; margin-left: 5px; }
@media (prefers-reduced-motion: reduce) { .pf-s { transition: none; } }

.pf-scores { margin-top: 12px; max-width: 46em; }
.pf summary { cursor: pointer; color: var(--pf-faded); font-size: 0.8125rem; width: fit-content; }
.pf summary:hover { color: var(--pf-vellum); }
.pf-table { display: grid; gap: 6px 12px; align-items: center; margin-top: 10px;
  grid-template-columns: minmax(9rem, max-content) minmax(6rem, 1fr) 2.6rem 5.5rem;
  font-size: 0.875rem; }
.pf-table .pf-head-row { color: var(--pf-faded); font-size: 0.75rem; }
.pf-table .pf-off { color: var(--pf-faded); }
.pf-meter { height: 6px; border-radius: 3px; background: var(--pf-track); position: relative; }
.pf-meter > .pf-level { position: absolute; inset: 0 auto 0 0; border-radius: 3px;
  background: var(--pf-faded); }
.pf-meter > .pf-level.pf-on { background: var(--pf-vellum); }
.pf-meter > .pf-mark { position: absolute; top: -3px; bottom: -3px; width: 2px; margin-left: -1px;
  background: var(--pf-rubric); }
.pf-num { font-variant-numeric: tabular-nums; text-align: right; }
.pf-f1 { color: var(--pf-faded); font-variant-numeric: tabular-nums; }
.pf-defs { margin-top: 12px; color: var(--pf-faded); font-size: 0.875rem; }
.pf-defs b { color: var(--pf-vellum); font-weight: 600; }
.pf-defs p + p { margin-top: 6px; }

.pf-skipped { margin: 0 0 30px calc(12.5rem + 22px); max-width: 46em; }
.pf-body .pf-skipped { margin: 0.8em 0; }
.pf-skipped ol { list-style: none; margin: 8px 0 0; padding: 0; font-size: 0.8125rem;
  color: var(--pf-faded); }
.pf-skipped li { margin-top: 4px; overflow-wrap: anywhere; }
.pf-skipped li i { font-style: normal; color: var(--pf-faded); opacity: 0.8; }
.pf-end { margin-left: calc(12.5rem + 22px); max-width: 46em; color: var(--pf-faded);
  border-top: 1px solid var(--pf-rule); padding-top: 14px; }

/* accordions */
#pf-about { margin-top: 28px; gap: 0; width: auto !important;
  border-top: 1px solid var(--pf-rule); }
#pf-about > .block { padding: 0 !important; border-bottom: 1px solid var(--pf-rule) !important; }
#pf-about .label-wrap { padding: 14px 0; font: 500 0.9375rem var(--pf-sans); }
#pf-about .label-wrap + div { padding-bottom: 18px; }
#pf-about .block .block { padding: 0 !important; }
#pf-about .prose { font-family: var(--pf-sans); }
#pf-about .prose p, #pf-about .prose li { color: var(--pf-vellum); max-width: 42em; }
#pf-about .prose table { font-size: 0.875rem; border-collapse: collapse; margin: 14px 0 18px; }
#pf-about .prose table, #pf-about .prose tr { border: 0 !important; }
#pf-about .prose th, #pf-about .prose td { border: 0 !important; padding: 7px 20px 7px 0;
  border-bottom: 1px solid var(--pf-rule) !important; background: none !important;
  text-align: left; }
#pf-about .prose th { color: var(--pf-faded); font-weight: 500; font-size: 0.8125rem; }
#pf-about .prose th, #pf-about .prose td { word-break: normal !important;
  overflow-wrap: normal !important; hyphens: manual; }
#pf-about .prose td:last-child, #pf-about .prose th:last-child { padding-right: 0; }
.pf-footer { max-width: 46rem; color: var(--pf-faded); font: 400 0.8125rem/1.5 var(--pf-sans);
  margin-top: 16px; }
.pf-footer a { color: var(--pf-link); }

/* two columns on wide screens */
@media (min-width: 1180px) {
  #pf-layout { display: grid !important; align-items: start;
    grid-template-columns: clamp(19rem, 23vw, 25rem) minmax(0, 1fr);
    grid-template-rows: auto 1fr; grid-template-areas: "intro work" "about work";
    column-gap: clamp(2rem, 3vw, 3.5rem); }
  #pf-intro { grid-area: intro; }
  #pf-work { grid-area: work; }
  #pf-about { grid-area: about; margin-top: 4px; }
  .pf-lede { font-size: 1.125rem; }
  #pf-about .prose table { font-size: 0.8125rem; }
}
@media (max-width: 860px) {
  .pf-passage { grid-template-columns: minmax(0, 1fr); }
  .pf-gloss { grid-column: 1; text-align: left; padding: 0 0 8px 22px; }
  .pf-gloss .pf-domain { justify-content: flex-start; }
  .pf-gloss .pf-domain .pf-swatch { order: -1; }
  .pf-body { grid-column: 1; }
  .pf-skipped, .pf-end { margin-left: 22px; }
}
@media (max-width: 560px) {
  #pf-actions button { padding-inline: 14px; }
  .pf-table { grid-template-columns: minmax(0, 1fr) 4.5rem 2.6rem; }
  .pf-table .pf-f1 { display: none; }
}
""" + "".join(
    # margin number <-> sentence hover highlight
    f'.pf-passage:has([data-refs~="{n}"]:hover) .pf-s[data-ref="{n}"],'
    f'.pf-passage:has(.pf-s[data-ref="{n}"]:hover) .pf-s[data-ref="{n}"],'
    f'.pf-passage:has(.pf-s[data-ref="{n}"]:hover) [data-refs~="{n}"]'
    "{background: var(--pf-glow);}\n"
    for n in range(1, LINKED_REFERENCES + 1))


def domain_label(name: str) -> str:
    return DOMAINS.get(name, (name, ""))[0]


def results_html(read: reading.Reading, results: list[PassageResult]) -> str:
    if read.problem:
        return (f'<div class="pf" id="pf-results"><p class="pf-message">'
                f"{PROBLEMS[read.problem]}</p></div>")
    parts = [_summary(read, results), _reader(read, results), _unread(read)]
    return '<div class="pf" id="pf-results">' + "".join(parts) + "</div>"


# ---- summary ----

def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else plural or word + 's'}"


def _summary(read: reading.Reading, results: list[PassageResult]) -> str:
    n = len(results)
    words = sum(p.words for p in read.passages)
    out = [f'<p class="pf-total">Read {_plural(n, "passage")}, {_plural(words, "word")}.</p>']
    if n > 1:
        found = Counter(d.name for r in results for d in r.domains)
        rows = []
        for key, (label, css) in DOMAINS.items():
            share = 100 * found[key] / n
            rows.append(
                f'<span class="pf-bar-name"><span class="pf-swatch {css}"></span>{label}</span>'
                f'<span class="pf-track" role="img" aria-label="{label}: {found[key]} of {n}">'
                f'<span class="{css}" style="width:{share:.1f}%"></span></span>'
                f'<span class="pf-count">{found[key]} of {n}</span>')
        out.append('<div class="pf-bars">' + "".join(rows) + "</div>")
        rest = []
        none = sum(r.outcome == "no_philosophy" for r in results)
        unclear = sum(r.outcome == "unclear" for r in results)
        if none:
            rest.append(f"{_plural(none, 'passage')} showed no philosophical position.")
        if unclear:
            rest.append(f"{_plural(unclear, 'passage was', 'passages were')} unclear.")
        if rest:
            out.append(f'<p class="pf-rest">{" ".join(rest)}</p>')
    if any(c.evidence for r in results for d in r.domains for c in d.categories):
        out.append('<p class="pf-rest">Underlined: the sentences each position rests on most. '
                   "Their numbers match the margin.</p>")
    return '<section class="pf-summary">' + "".join(out) + "</section>"


# ---- passages ----

def _reader(read: reading.Reading, results: list[PassageResult]) -> str:
    out, i = [], 0
    for passage, result in zip(read.passages, results, strict=True):
        start, end = passage.blocks[0], passage.blocks[-1]
        out.append(_skipped(read.blocks[i:start]))
        out.append(_passage(passage, result, read.blocks[start:end + 1]))
        i = end + 1
    out.append(_skipped(read.blocks[i:]))
    return "".join(out)


def _references(result: PassageResult) -> dict[int, tuple[int, str, list[str]]]:
    """{sentence index: (ref number, css class, positions)}"""
    marks: dict[int, tuple[str, list[str]]] = {}
    for d in _in_order(result):
        for c in d.categories:
            for i in c.evidence:
                marks.setdefault(i, (DOMAINS[d.name][1], []))[1].append(c.name)
    return {i: (n, *marks[i]) for n, i in enumerate(sorted(marks), 1)}


def _passage(passage: reading.Passage, result: PassageResult,
             blocks: tuple[reading.Block, ...]) -> str:
    split = [reading.sentences(b.text) if not b.skip else [] for b in blocks]
    refs = _references(result)
    if refs and sum(map(len, split)) != len(result.sentences):
        refs = {}  # sentence split doesn't match the model's
    body, run, k = [], [], 0
    for block, block_sentences in zip(blocks, split, strict=True):
        if block.skip:
            run.append(block)
            continue
        body.append(_skipped(run))
        run = []
        parts = []
        for sentence in block_sentences:
            if k in refs:
                n, css, names = refs[k]
                parts.append(f'<span class="pf-s pf-u-{css}" data-ref="{n}" '
                             f'title="{escape(", ".join(names))}">{escape(sentence)}</span>'
                             f'<sup class="pf-ref">{n}</sup>')
            else:
                parts.append(escape(sentence))
            k += 1
        body.append(f"<p>{' '.join(parts)}</p>")
    rules = "".join(f'<span class="{DOMAINS[d.name][1]}"></span>' for d in _in_order(result))
    if result.outcome == "unclear":
        rules = '<span class="pf-dashed"></span>'
    quiet = " pf-quiet" if result.outcome != "domains" else ""
    return (f'<article class="pf-passage{quiet}"><div class="pf-gloss">'
            f'{_gloss(passage, result, refs)}</div><div class="pf-body">'
            f'<div class="pf-rules" aria-hidden="true">{rules}</div>'
            f'{"".join(body)}{_scores(result)}</div></article>')


def _in_order(result: PassageResult):
    order = list(DOMAINS)
    return sorted(result.domains, key=lambda d: order.index(d.name) if d.name in order else 99)


def _gloss(passage: reading.Passage, result: PassageResult,
           refs: dict[int, tuple[int, str, list[str]]]) -> str:
    out = []
    if result.outcome == "domains":
        for d in _in_order(result):
            label, css = DOMAINS[d.name]
            items = []
            for c in d.categories:
                numbers = sorted(refs[i][0] for i in c.evidence if i in refs)
                mark = (f'<span class="pf-refs">{", ".join(map(str, numbers))}</span>'
                        if numbers else "")
                items.append(f'<li data-refs="{" ".join(map(str, numbers))}">'
                             f"{escape(c.name)}{mark}</li>")
            out.append(f'<p class="pf-domain">{label}<span class="pf-swatch {css}"></span></p>'
                       f'<ul class="pf-cats">{"".join(items)}</ul>')
    elif result.outcome == "no_philosophy":
        out.append('<p class="pf-gloss-note">No philosophical position found</p>')
    else:
        out.append('<p class="pf-gloss-note">Unclear. Nothing reached its mark; the closest '
                   f"was {escape(result.closest[0].name)}</p>")
    notes = []
    if passage.words < reading.SHORT_WORDS:
        notes.append("Shorter than the passages it learned from")
    if result.truncated:
        notes.append("Too long to read whole: only the start was read")
    out += [f'<p class="pf-gloss-note">{n}</p>' for n in notes]
    return "".join(out)


def _scores(result: PassageResult) -> str:
    flagged = [c for d in result.domains for c in d.categories]
    rows = ['<span class="pf-head-row">Position</span><span class="pf-head-row">Score and mark'
            '</span><span class="pf-head-row pf-num">Score</span>'
            '<span class="pf-head-row pf-f1">F1 in testing</span>']
    for c in sorted([*result.categories, result.ipc], key=lambda c: -c.probability):
        off = "" if c.detected else " pf-off"
        f1 = f"{c.dev_f1:.2f}" if c.dev_f1 is not None else ""
        name = "No philosophical position" if c.id == result.ipc.id else c.name
        rows.append(
            f'<span class="{off.strip()}">{escape(name)}</span>'
            f'<span class="pf-meter" role="img" aria-label="score {c.probability:.2f}, '
            f'marked from {c.threshold:.2f}"><span class="pf-level{"" if off else " pf-on"}" '
            f'style="width:{100 * c.probability:.1f}%"></span>'
            f'<span class="pf-mark" style="left:{100 * c.threshold:.1f}%"></span></span>'
            f'<span class="pf-num{off}">{c.probability:.2f}</span>'
            f'<span class="pf-f1">{f1}</span>')
    defs = "".join(f"<p><b>{escape(c.name)}</b>: {escape(c.description)}</p>"
                   for c in flagged if c.description)
    return ('<details class="pf-scores"><summary>Scores</summary>'
            '<div class="pf-table">' + "".join(rows) + "</div>"
            + (f'<div class="pf-defs">{defs}</div>' if defs else "") + "</details>")


def _skipped(blocks) -> str:
    blocks = [b for b in blocks if b.skip and b.skip != reading.UNREAD]
    if not blocks:
        return ""
    reasons = Counter(SKIPPED[b.skip] for b in blocks)
    why = ", ".join(r for r, _ in reasons.most_common())
    items = "".join(f"<li>{escape(_clip(b.text))} <i>({SKIPPED[b.skip]})</i></li>"
                    for b in blocks[:SKIPPED_LISTED])
    if len(blocks) > SKIPPED_LISTED:
        items += f"<li>and {_plural(len(blocks) - SKIPPED_LISTED, 'more line')}</li>"
    return (f'<details class="pf-skipped"><summary>Skipped {_plural(len(blocks), "line")}: '
            f"{why}</summary><ol>{items}</ol></details>")


def _clip(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _unread(read: reading.Reading) -> str:
    if not read.unread_words:
        return ""
    return (f'<p class="pf-end">The last {_plural(read.unread_words, "word")} weren’t read. The '
            f"app reads up to {reading.MAX_PASSAGES} passages, about "
            f"{reading.MAX_PASSAGES * 200:,} words, at a time.</p>")


# ---- accordions ----

def how_it_reads_markdown(release: dict) -> str:
    train = release.get("training", {}).get("sizes", {}).get("train")
    size = f"{train:,}" if train else "about a thousand"
    return "\n".join([
        f"A RoBERTa model fine-tuned on {size} passages from UK and US news sites, magazines, "
        "think tanks and government websites. One person labelled them against 12 philosophical "
        "positions in 4 domains. A passage can have several positions or none.",
        "",
        f"Your text is split into passages of about {reading.TARGET_WORDS} to 250 words, the "
        "length it was trained on. Headlines, captions, menus, reference lists and text that "
        "isn’t in English are skipped.",
        "",
        "Each position has its own threshold, and a passage is labelled when its score reaches "
        "it. **Scores**, under each passage, shows them all.",
    ])


def reliability_markdown(release: dict) -> str:
    rel = release.get("reliability", {})
    names = {label["id"]: (label["name"], label["domain"]) for label in release["labels"]}
    out = ["**F1** runs from 0 to 1. It’s high when most flags are right and few are missed."]
    development = rel.get("development")
    if development:
        out += ["", f"How it was tested: the {development['n_passages']:,} labelled passages "
                "were split into 9 groups, with each article in only one group. The model was "
                "trained 9 times, each time tested on a group it hadn’t seen. The scores are "
                f"averages over {len(development['seeds'])} repeats."]
    out.append("")
    domains = rel.get("domains")
    if domains:
        out += ["| Domain | F1 in testing | 95% interval |", "|---|---|---|"]
        for r in domains["domains"]:
            if r["domain"] not in DOMAINS:
                continue
            out.append(f"| {domain_label(r['domain'])} | {r['f1']:.2f} | "
                       f"{r['f1_ci95'][0]:.2f} to {r['f1_ci95'][1]:.2f} |")
        out += ["", f"It got exactly the right domains for {domains['exact_match']['mean']:.0%} "
                "of passages.", ""]
    if development:
        out += ["| Position | Domain | F1 in testing |", "|---|---|---|"]
        for r in development["per_label"]:
            name, domain = names[r["label"]]
            if domain not in DOMAINS:
                name, domain = "No philosophical position", ""
            note = " (too rare to measure)" if not r["measurable"] else ""
            out.append(f"| {name} | {domain_label(domain)} | {r['f1']:.2f}{note} |")
        out += ["", f"Average F1 over all labels: "
                f"{development['headline']['macro_f1']['mean']:.2f}."]
        lockbox = rel.get("lockbox")
        if lockbox:
            out.append(f"On {lockbox['n_passages']} passages kept aside until the end: "
                       f"{lockbox['headline']['macro_f1']['mean']:.2f}.")
    return "\n".join(out)
