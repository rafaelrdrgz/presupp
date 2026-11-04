"""A page for re-marking the passages whose evidence couldn't be realigned.

    python -m presupp.data.remark        # writes data/processed/remark_evidence_v1.html

Holds article text, so it's written to the gitignored data/processed/. The CSV it downloads
goes in data/adjudication/.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from presupp import paths
from presupp.labels import IPC
from presupp.taxonomy import Category, Taxonomy, load_taxonomy

OUT = paths.PROCESSED_DIR / "remark_evidence_v1.html"
STORAGE_KEY = "presupp-remark-evidence-v1"
DOMAIN_COLOURS = {"Metaphysical": "#7e6cd9", "Epistemological": "#31a0d0", "Ethical": "#c12440",
                  "Political-Philosophical": "#e66941"}


def _category(category: Category) -> dict:
    return {"id": category.id, "name": category.name,
            "domain": category.domain.replace("-Philosophical", ""),
            "colour": DOMAIN_COLOURS.get(category.domain, "#a4adc2"),
            "description": category.description.split(". ")[0].rstrip(".") + "."}


def to_remark(passages: pd.DataFrame, gold: pd.DataFrame, taxonomy: Taxonomy) -> list[dict]:
    """Passages with evidence that doesn't fit, each with all its non-IPC gold positions."""
    gold = gold[gold["category"] != IPC]
    affected = sorted(set(gold.loc[gold["evidence_reading"] == "doesn't fit", "passage_id"])
                      & set(passages["passage_id"]))
    passages = passages.set_index("passage_id")
    out = []
    for pid in affected:
        positions = []
        for row in gold[gold["passage_id"] == pid].sort_values("category").itertuples():
            fits = row.evidence_reading != "doesn't fit"
            recorded = [int(e) for e in row.evidence]
            if row.evidence_reading.startswith("0-based"):  # stored 0-based: show S-numbers
                recorded = [e + 1 for e in recorded]
            positions.append({
                **_category(taxonomy.get(row.category)),
                "start": [int(i) for i in row.evidence_0based] if fits else [],
                "recorded": recorded,  # the S-numbers clicked
                "fits": fits, "confidence": row.confidence,
            })
        out.append({"id": pid, "source": str(passages.loc[pid, "source"]),
                    "sentences": [str(s) for s in passages.loc[pid, "sentences"]],
                    "positions": positions})
    return out


def page(items: list[dict], taxonomy: Taxonomy, made_on: str | None = None) -> str:
    data = {"passages": items, "storageKey": STORAGE_KEY,
            "madeOn": made_on or date.today().isoformat(),
            "categories": [_category(c) for c in taxonomy.categories if not c.is_exclusion]}
    # stop "</" closing the <script> early
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the evidence re-marking page.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    from presupp.data.dataset import load_dev

    passages, gold = load_dev(paths.PROCESSED_DIR)
    taxonomy = load_taxonomy("v1")
    items = to_remark(passages[passages["is_labelled"]], gold, taxonomy)
    args.out.write_text(page(items, taxonomy), encoding="utf-8")
    positions = sum(len(p["positions"]) for p in items)
    to_fix = sum(not q["fits"] for p in items for q in p["positions"])
    print(f"Wrote {args.out}: {len(items)} passages, {positions} positions "
          f"({to_fix} to re-mark, {positions - to_fix} to check). Open it in a browser.")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Re-mark evidence</title>
<style>
:root { color-scheme: dark; --ink: #0f1730; --raised: #16203f; --rule: #27335a;
  --vellum: #ece6d9; --faded: #a4adc2; --link: #8fc3ea; --rubric: #c12440;
  --serif: Georgia, "Times New Roman", serif; --sans: system-ui, "Segoe UI", sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--ink); color: var(--vellum); font: 15px/1.5 var(--sans); }
button, select { font: inherit; color: inherit; cursor: pointer; }
:focus-visible { outline: 2px solid var(--link); outline-offset: 2px; }
header { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center;
  padding: 14px 24px; border-bottom: 1px solid var(--rule); position: sticky; top: 0;
  background: var(--ink); z-index: 2; }
header h1 { font: 600 1.25rem var(--serif); margin: 0; }
#progress { color: var(--faded); }
.spacer { flex: 1; }
.tool { background: none; border: 1px solid var(--rule); border-radius: 6px; padding: 6px 14px; }
.tool:hover { border-color: var(--faded); }
.tool:disabled { opacity: 0.45; cursor: default; }
.primary { background: #a11d37; border-color: #a11d37; color: #fbf2ee; }
.primary:hover { background: var(--rubric); border-color: var(--rubric); }
#help { max-width: 58rem; margin: 16px 24px 0; color: var(--faded); }
#help p { margin: 0 0 6px; }
kbd { border: 1px solid var(--rule); border-radius: 4px; padding: 0 5px; font-size: 0.85em; }
.layout { display: grid; grid-template-columns: 13rem minmax(0, 1fr); gap: 28px;
  padding: 20px 24px 60px; }
#list { list-style: none; margin: 0; padding: 0; max-height: calc(100vh - 120px);
  overflow-y: auto; position: sticky; top: 80px; }
#list button { width: 100%; text-align: left; background: none; border: 0;
  border-radius: 5px; padding: 4px 8px; color: var(--faded); display: flex; gap: 8px; }
#list button.current { background: var(--raised); color: var(--vellum); }
#list .tick { width: 1em; color: #6fcf97; }
main { max-width: 58rem; }
.title { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; }
main h2 { font: 600 1.375rem var(--serif); margin: 0; }
.meta { color: var(--faded); margin: 4px 0 18px; }
.positions { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; align-items: center; }
.pos { background: var(--raised); border: 1px solid var(--rule); border-radius: 6px;
  padding: 6px 12px; display: flex; gap: 8px; align-items: center; text-align: left; }
.pos.active { border-color: var(--vellum); }
.pos.removed { opacity: 0.5; text-decoration: line-through; }
.dot { width: 10px; height: 10px; border-radius: 2px; flex: none; }
.pos .picked { color: var(--faded); font-size: 0.85em; }
.pos .state { font-size: 0.75rem; color: var(--faded); border: 1px solid var(--rule);
  border-radius: 999px; padding: 0 7px; text-decoration: none; }
.pos .state.todo { color: #f2c94c; border-color: #8a6d1f; }
.pos .state.added { color: #6fcf97; border-color: #2f6b4b; }
select.add { background: none; border: 1px dashed var(--rule); border-radius: 6px;
  padding: 6px 10px; color: var(--faded); }
select.add option, select.add optgroup { background: var(--raised); color: var(--vellum); }
.detail { color: var(--faded); margin: 0 0 10px; }
.detail b { color: var(--vellum); font-weight: 600; }
.confidence { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 0 0 16px; }
.confidence span { color: var(--faded); margin-right: 4px; }
.confidence button { background: none; border: 1px solid var(--rule); border-radius: 999px;
  padding: 2px 12px; }
.confidence button[aria-pressed="true"] { background: var(--vellum); color: var(--ink);
  border-color: var(--vellum); }
.confidence .was { font-size: 0.85em; }
ol.sentences { list-style: none; margin: 0 0 16px; padding: 0; display: grid; gap: 6px; }
ol.sentences button { width: 100%; text-align: left; display: grid;
  grid-template-columns: 2.6rem 1fr auto; gap: 10px; align-items: baseline;
  background: none; border: 1px solid transparent; border-radius: 6px; padding: 8px 10px;
  font: 1.02rem/1.6 var(--serif); }
ol.sentences button:hover { border-color: var(--rule); }
ol.sentences button[aria-pressed="true"] { background: var(--raised);
  border-color: var(--sel, var(--vellum)); box-shadow: inset 4px 0 0 var(--sel, var(--vellum)); }
.num { font: 600 0.85rem var(--sans); color: var(--faded); }
.marks { display: flex; gap: 3px; }
.marks span { width: 8px; height: 8px; border-radius: 2px; }
.row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 8px; }
textarea { width: 100%; min-height: 3.2em; background: var(--raised); color: var(--vellum);
  border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px; font: inherit; }
label.small { color: var(--faded); font-size: 0.85em; display: block; margin: 14px 0 4px; }
.why { color: var(--faded); font-size: 0.85em; }
@media (max-width: 760px) { .layout { grid-template-columns: 1fr; } #list { display: none; } }
</style>
</head>
<body>
<header>
  <h1>Re-mark evidence</h1>
  <span id="progress"></span>
  <span class="spacer"></span>
  <button class="tool" id="toggle-help">How this works</button>
  <label class="tool" for="load">Load a saved file<input id="load" type="file" accept=".csv"
    hidden></label>
  <button class="tool primary" id="download">Download</button>
</header>
<section id="help" hidden>
  <p>These passages' evidence was clicked on a different sentence split from the one stored, so
  the numbers no longer line up. The sentences below are the stored ones.</p>
  <p>Pick a position at the top, then click every sentence that carries it, as in the annotation
  tool, and set its confidence. Positions marked <b>to re-mark</b> start empty; the rest start
  from your original clicks: check them. Confidence starts from your original rating. If no
  sentence carries a position, choose <b>No sentence carries this</b>, or remove the position.
  <b>Add a position</b> adds one the passage holds that wasn't labelled.</p>
  <p>Keys: <kbd>1</kbd>–<kbd>9</kbd> toggle a sentence, <kbd>H</kbd> <kbd>M</kbd> <kbd>L</kbd>
  set confidence, <kbd>←</kbd> <kbd>→</kbd> switch position, <kbd>Enter</kbd> done and next
  passage. Your work is saved in this browser as you go. <b>Download</b> saves
  evidence_remarked_v1.csv: put it in data/adjudication/ and rebuild.</p>
</section>
<div class="layout">
  <ul id="list" aria-label="Passages"></ul>
  <main id="passage"></main>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const PASSAGES = DATA.passages;
const CATEGORIES = Object.fromEntries(DATA.categories.map(c => [c.id, c]));
const LEVELS = ["high", "medium", "low"];

// state[passage id] = {done, note, decidedOn, picked: {position: [0-based sentences]},
// none: {position: true}, confidence: {position: level}, added: [position], removed: {position}}
function freshPassage(p) {
  const picked = {}, confidence = {};
  for (const q of p.positions) { picked[q.id] = [...q.start]; confidence[q.id] = q.confidence; }
  return {done: false, note: "", picked, none: {}, confidence, added: [], removed: {}};
}
function freshState() {
  return Object.fromEntries(PASSAGES.map(p => [p.id, freshPassage(p)]));
}
function loadState() {
  const state = freshState();
  try {
    const saved = JSON.parse(localStorage.getItem(DATA.storageKey) || "{}");
    for (const [pid, s] of Object.entries(saved)) {
      const fresh = state[pid];
      if (!fresh) continue;
      // work saved by an earlier version of this page has no confidence, added or removed
      state[pid] = {...fresh, ...s,
        picked: {...fresh.picked, ...(s.picked || {})}, none: {...(s.none || {})},
        confidence: {...fresh.confidence, ...(s.confidence || {})},
        added: [...(s.added || [])], removed: {...(s.removed || {})}};
    }
  } catch (e) { /* no saved work, or storage blocked: start fresh */ }
  return state;
}
let state = loadState();
let current = Math.max(0, PASSAGES.findIndex(p => !state[p.id].done));
let active = 0;

function positionsOf(p) {  // the passage's labelled positions, then any added
  const added = state[p.id].added.map(id => ({...CATEGORIES[id], start: [], recorded: [],
                                               fits: false, confidence: null, added: true}));
  return [...p.positions, ...added];
}
function save() {
  try { localStorage.setItem(DATA.storageKey, JSON.stringify(state)); } catch (e) { /* ignore */ }
}
function changed(pid) { state[pid].done = false; save(); }
function toggle(pid, qid, i) {
  const s = state[pid], picked = new Set(s.picked[qid] || []);
  picked.has(i) ? picked.delete(i) : picked.add(i);
  s.picked[qid] = [...picked].sort((a, b) => a - b);
  delete s.none[qid];
  changed(pid);
}
function setNone(pid, qid) {
  state[pid].picked[qid] = [];
  state[pid].none[qid] = true;
  changed(pid);
}
function setConfidence(pid, qid, level) {
  state[pid].confidence[qid] = level;
  changed(pid);
}
function addPosition(pid, qid) {
  const s = state[pid];
  if (!s.added.includes(qid)) s.added.push(qid);
  s.picked[qid] = [];
  s.confidence[qid] = null;
  changed(pid);
}
function removePosition(p, q) {
  const s = state[p.id];
  if (q.added) {  // an added position just goes
    s.added = s.added.filter(id => id !== q.id);
    delete s.picked[q.id];
    delete s.confidence[q.id];
  } else if (s.removed[q.id]) {
    delete s.removed[q.id];
  } else {
    s.removed[q.id] = true;
  }
  changed(p.id);
}
function live(p) { return positionsOf(p).filter(q => !state[p.id].removed[q.id]); }
function ready(p) {
  // every remaining position has sentences (or none carries it) and a confidence
  const s = state[p.id];
  return live(p).length > 0 && live(p).every(q =>
    ((s.picked[q.id] || []).length || (s.none[q.id] && !q.added)) && s.confidence[q.id]);
}
function sNumbers(indices) { return indices.map(i => "S" + (i + 1)).join(", "); }

function csvField(v) {
  v = String(v ?? "");
  return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}
const COLUMNS = ["passage_id", "category", "confidence", "evidence_sentences", "status",
                 "decided_on", "note"];
function toCsv() {
  const rows = [COLUMNS];
  for (const p of PASSAGES) {
    const s = state[p.id];
    if (!s.done) continue;
    for (const q of positionsOf(p)) {
      const removed = Boolean(s.removed[q.id]);
      rows.push([p.id, q.id, removed ? "" : s.confidence[q.id],
                 removed ? "" : (s.picked[q.id] || []).map(i => i + 1).join(";"),
                 removed ? "removed" : q.added ? "added" : "original",
                 s.decidedOn || DATA.madeOn, s.note || ""]);
    }
  }
  return rows.map(r => r.map(csvField).join(",")).join("\n") + "\n";
}
function parseCsv(text) {  // enough for the file this page writes
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}
function loadCsv(text) {
  const [header, ...rows] = parseCsv(text);
  const get = (r, name) => (header.indexOf(name) >= 0 ? r[header.indexOf(name)] : "");
  const seen = new Set();
  for (const r of rows) {
    const pid = get(r, "passage_id"), p = PASSAGES.find(x => x.id === pid);
    if (!p) continue;
    if (!seen.has(pid)) { state[pid] = freshPassage(p); seen.add(pid); }
    const s = state[pid], qid = get(r, "category"), status = get(r, "status") || "original";
    const picked = get(r, "evidence_sentences");
    if (status === "added" && !s.added.includes(qid)) s.added.push(qid);
    if (status === "removed") s.removed[qid] = true;
    s.picked[qid] = picked ? picked.split(";").map(n => Number(n) - 1) : [];
    if (!picked && status === "original") s.none[qid] = true;
    if (get(r, "confidence")) s.confidence[qid] = get(r, "confidence");
    s.note = get(r, "note");
    s.decidedOn = get(r, "decided_on") || DATA.madeOn;
    s.done = true;
  }
  save();
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* fall back */ }
  const area = el("textarea", {style: "position:fixed;opacity:0"});
  area.value = text;
  document.body.append(area);
  area.select();
  const ok = document.execCommand("copy");
  area.remove();
  return ok;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;  // attribute left off
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function renderList() {
  const list = document.getElementById("list");
  list.replaceChildren(...PASSAGES.map((p, k) => el("li", {}, el("button", {
    class: k === current ? "current" : "", onclick: () => { current = k; active = 0; render(); }
  }, el("span", {class: "tick"}, state[p.id].done ? "✓" : ""), p.id))));
  const done = PASSAGES.filter(p => state[p.id].done).length;
  document.getElementById("progress").textContent =
    `${done} of ${PASSAGES.length} passages done`;
}

function renderPassage() {
  const p = PASSAGES[current], s = state[p.id], positions = positionsOf(p);
  active = Math.min(active, positions.length - 1);
  const q = positions[active], removed = Boolean(s.removed[q.id]);

  const tabs = positions.map((pos, k) => {
    const picked = s.picked[pos.id] || [];
    const label = s.removed[pos.id] ? "removed" : s.none[pos.id] ? "no sentence"
      : picked.length ? sNumbers(picked) : "nothing yet";
    const badge = pos.added ? ["added", "added"]
      : pos.fits ? ["check", ""] : ["to re-mark", "todo"];
    return el("button", {class: "pos" + (k === active ? " active" : "") +
                                (s.removed[pos.id] ? " removed" : ""),
                         "aria-pressed": String(k === active),
                         onclick: () => { active = k; render(); }},
      el("span", {class: "dot", style: `background:${pos.colour}`}), pos.name,
      el("span", {class: "picked"}, label), el("span", {class: "state " + badge[1]}, badge[0]));
  });
  const present = new Set(positions.map(pos => pos.id));
  const add = el("select", {class: "add", "aria-label": "Add a position",
                            onchange: e => { if (e.target.value) {
                              addPosition(p.id, e.target.value);
                              active = positionsOf(p).length - 1; render(); } }},
                 el("option", {value: ""}, "Add a position…"));
  for (const domain of [...new Set(DATA.categories.map(c => c.domain))]) {
    const group = el("optgroup", {label: domain});
    for (const c of DATA.categories.filter(c => c.domain === domain && !present.has(c.id))) {
      group.append(el("option", {value: c.id}, c.name));
    }
    if (group.children.length) add.append(group);
  }

  const hint = q.added ? "Added here: click the sentences that carry it."
    : q.fits ? `Started from your original clicks (${q.recorded.map(n => "S" + n).join(", ")}), ` +
               "read onto these sentences."
    : `Your original clicks were ${q.recorded.map(n => "S" + n).join(", ")} on a different ` +
      `split, so they don't fit these ${p.sentences.length} sentences.`;
  const detail = el("p", {class: "detail"}, el("b", {}, q.name), ` (${q.domain}): `,
                    q.description, " ", removed ? "Removed from this passage." : hint);

  const confidence = el("div", {class: "confidence", role: "group", "aria-label": "Confidence"},
    el("span", {}, "Confidence"),
    ...LEVELS.map(level => el("button", {"aria-pressed": String(s.confidence[q.id] === level),
                                         disabled: removed ? "" : null,
                                         onclick: () => { setConfidence(p.id, q.id, level);
                                                          render(); }},
                              level[0].toUpperCase() + level.slice(1))),
    q.confidence && !q.added ? el("span", {class: "was"}, `(originally ${q.confidence})`) : "");

  const sentences = el("ol", {class: "sentences"}, ...p.sentences.map((text, i) => {
    const on = !removed && (s.picked[q.id] || []).includes(i);
    const marks = positions.filter(pos => !s.removed[pos.id] &&
                                          (s.picked[pos.id] || []).includes(i))
      .map(pos => el("span", {style: `background:${pos.colour}`, title: pos.name}));
    return el("li", {}, el("button", {"aria-pressed": String(on),
        style: on ? `--sel:${q.colour}` : "",
        onclick: () => { if (!removed) { toggle(p.id, q.id, i); render(); } }},
      el("span", {class: "num"}, "S" + (i + 1)), text, el("span", {class: "marks"}, ...marks)));
  }));

  const note = el("textarea", {id: "note", placeholder: "Optional: anything worth recording " +
                               "about this passage (no need to quote it)."});
  note.value = s.note;
  note.addEventListener("input", () => { s.note = note.value; save(); });
  const doneButton = el("button", {class: "tool primary", onclick: finish},
                        current + 1 < PASSAGES.length ? "Done, next passage" : "Done");
  if (!ready(p)) doneButton.disabled = true;
  const copy = el("button", {class: "tool", onclick: async e => {
    const ok = await copyText(p.sentences.join(" "));
    e.target.textContent = ok ? "Copied" : "Couldn't copy";
    setTimeout(() => { e.target.textContent = "Copy passage text"; }, 1500);
  }}, "Copy passage text");

  const actions = [];
  if (!q.added && !removed) {
    actions.push(el("button", {class: "tool", onclick: () => { setNone(p.id, q.id); render(); }},
                    "No sentence carries this"));
  }
  if (!removed) {
    actions.push(el("button", {class: "tool", onclick: () => {
      s.picked[q.id] = [...q.start]; delete s.none[q.id];
      s.confidence[q.id] = q.confidence; changed(p.id); render(); }}, "Reset this position"));
  }
  const keeps = live(p).length - (removed ? 0 : 1);
  const remove = el("button", {class: "tool", onclick: () => { removePosition(p, q); render(); }},
                    q.added ? "Remove this position" : removed ? "Restore this position"
                                                               : "Remove this position");
  if (!removed && keeps < 1) remove.disabled = true;  // a passage keeps at least one position
  actions.push(remove);

  document.getElementById("passage").replaceChildren(
    el("div", {class: "title"}, el("h2", {}, p.id), copy),
    el("p", {class: "meta"}, `${p.source}. ${live(p).length} position` +
       (live(p).length === 1 ? "" : "s") + (s.done ? ". Done." : "")),
    el("div", {class: "positions"}, ...tabs, add), detail, confidence, sentences,
    el("div", {class: "row"}, ...actions),
    el("label", {class: "small", for: "note"}, "Note"), note,
    el("div", {class: "row"},
       el("button", {class: "tool", onclick: () => move(-1)}, "Previous"), doneButton,
       ready(p) ? "" : el("span", {class: "why"},
                          "Each position needs its sentences (or none) and a confidence.")));
}
function render() { renderList(); renderPassage(); }

function move(step) {
  current = Math.min(PASSAGES.length - 1, Math.max(0, current + step));
  active = 0;
  render();
  window.scrollTo(0, 0);
}
function finish() {
  const p = PASSAGES[current];
  if (!ready(p)) return;
  state[p.id].done = true;
  state[p.id].decidedOn = new Date().toISOString().slice(0, 10);
  save();
  if (current + 1 < PASSAGES.length) move(1); else render();
}

document.addEventListener("keydown", e => {
  if (["TEXTAREA", "INPUT", "SELECT"].includes(e.target.tagName)) return;
  const p = PASSAGES[current], q = positionsOf(p)[active];
  const level = {h: "high", m: "medium", l: "low"}[e.key.toLowerCase()];
  if (/^[1-9]$/.test(e.key) && Number(e.key) <= p.sentences.length) {
    if (!state[p.id].removed[q.id]) { toggle(p.id, q.id, Number(e.key) - 1); render(); }
  } else if (level && !state[p.id].removed[q.id]) {
    setConfidence(p.id, q.id, level);
    render();
  } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    e.preventDefault();
    const n = positionsOf(p).length;
    active = (active + (e.key === "ArrowRight" ? 1 : n - 1)) % n;
    render();
  } else if (e.key === "Enter" && e.target === document.body) {
    e.preventDefault();
    finish();
  }
});
document.getElementById("download").addEventListener("click", () => {
  const blob = new Blob([toCsv()], {type: "text/csv"});
  const a = el("a", {href: URL.createObjectURL(blob), download: "evidence_remarked_v1.csv"});
  document.body.append(a);
  a.click();
  a.remove();
});
document.getElementById("load").addEventListener("change", async e => {
  const file = e.target.files[0];
  if (file) { loadCsv(await file.text()); render(); }
});
document.getElementById("toggle-help").addEventListener("click", () => {
  const help = document.getElementById("help");
  help.hidden = !help.hidden;
});
if (!PASSAGES.some(p => state[p.id].done)) document.getElementById("help").hidden = false;
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
