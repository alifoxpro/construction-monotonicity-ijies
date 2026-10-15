"""A blinded pairwise reversal test — IJIES reviewer 2, point 3.

THE OBJECTION, verbatim: "Obtain a blinded human annotation subset with a frozen
protocol and adjudication of ambiguous boundaries."

WHY NOT JUST RE-LABEL A SUBSET BLIND. Stage labelling needs temporal context: an
annotator shown shuffled frames cannot say which stage a frame belongs to, and an
annotator shown the sequence in order sees a monotone progression and will tend to
label one. Blinding and the task fight each other. So this instrument changes the
task instead of the presentation.

THE INSTRUMENT. The annotator is shown ONE PAIR of frames at a time, side by side,
and answers a single question: which frame is later in the construction? Four
answers are allowed -- LEFT is later, RIGHT is later, SAME stage, CANNOT TELL.
Nothing else is on screen. The pairs come from across the corpus in random order,
the item ids are hashes, and half the pairs are shown with the two frames SWAPPED.
That gives three properties the second annotation pass did not have:

  * The annotator cannot see a monotone progression, because there is no sequence
    to see -- only unordered pairs from thirteen different buildings.
  * The swapped half are PLANTED REVERSALS. If the annotator does not call them
    backwards, the instrument has no power and its null result means nothing. This
    is the control that makes a null interpretable, and it is the reason the test
    is worth running at all.
  * The expected answer is not inferable from the manuscript. Knowing that the
    corpus violation rate is 0.000 tells the annotator nothing about which side of
    a given pair is later.

WHAT IT CAN AND CANNOT SETTLE. A pair whose frames annotator A placed in the same
stage, which the blind annotator calls a clear reversal, is a reversal A's
taxonomy hid -- evidence about explanation (d), the coarse-label blind spot. A
pair A placed across a forward boundary that the blind annotator calls backwards
is a disagreement for adjudication. What the instrument cannot reach is
explanation (b): a reversal the editor cut is not in any frame, so no annotator
can find it. That limit is permanent and is stated in the paper rather than
designed around.

ADJUDICATION. Three classes of item go to a third reading under the frozen
protocol: items answered CANNOT TELL, items where the blind answer contradicts
annotator A, and items where the planted direction was not detected. The
adjudicator sees the protocol and the pair, never the blind annotator's answer or
A's label.

    python src/data_collection/blind_pairs.py prepare --n 200
    #   -> blind_study/task.html   open it, answer, click Save
    #   -> blind_study/key.json    the sealed answer key; do NOT open it first
    python src/data_collection/blind_pairs.py score --answers blind_study/answers.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

from data.sequence_dataset import list_sequence_dirs  # noqa: E402

RAW = REPO / "seq_data_raw"
OUT = REPO / "blind_study"

#: Pair strata. The interior stratum is where explanation (d) lives: both frames
#: carry the same stage label, so A's taxonomy could not have recorded a reversal
#: between them even if one occurred. The boundary stratum is the control: A says
#: these two frames straddle a forward stage change, so a blind annotator who
#: cannot see that is telling us the instrument is noisy.
STRATA = ("interior", "boundary")


def read_labels(path: Path) -> list[tuple[str, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in {"filename", "file", ""}:
                continue
            rows.append((row[0].strip(), int(row[1])))
    return rows


def item_id(site: str, i: int, j: int, salt: str) -> str:
    """A hash, so the annotator cannot sort items back into sequence order."""
    return hashlib.sha256(f"{salt}|{site}|{i}|{j}".encode()).hexdigest()[:12]


def prepare(args) -> int:
    rng = random.Random(args.seed)
    sites = [d.name for d in list_sequence_dirs(REPO / args.data)]

    # Candidate pairs: adjacent ASSESSABLE frames, tagged by stratum. Adjacent
    # because that is the unit the violation rate is defined over -- the paper's
    # 0.000 counts adjacent labelled pairs, so the blind test must ask about the
    # same pairs or it is measuring something else.
    pool: dict[str, list[dict]] = {s: [] for s in STRATA}
    for site in sites:
        rows = [(n, s) for n, s in read_labels(RAW / site / "labels.csv") if s >= 0]
        for k in range(len(rows) - 1):
            (n1, s1), (n2, s2) = rows[k], rows[k + 1]
            stratum = "interior" if s1 == s2 else "boundary"
            if s2 < s1:
                continue          # there are none; a reversal would not be blind
            pool[stratum].append({
                "site": site, "i": k, "j": k + 1,
                "file_a": n1, "file_b": n2, "stage_a": s1, "stage_b": s2,
                "stratum": stratum,
            })

    print("candidate adjacent pairs by stratum:")
    for s in STRATA:
        print(f"  {s:9s} {len(pool[s])}")

    # The boundary stratum is the whole power budget and the corpus only has
    # 40 such pairs, so take ALL of them and spend the rest on interior pairs.
    # Splitting the budget evenly would have thrown away half the control.
    chosen: list[dict] = list(pool["boundary"])
    room = max(0, args.n - len(chosen))
    chosen += rng.sample(pool["interior"], min(room, len(pool["interior"])))

    # Swap exactly half of EACH stratum rather than flipping a coin per item:
    # with 40 boundary pairs, coin flips leave the number of planted reversals
    # to chance, and that number is the instrument's power.
    # Recorded on the dicts themselves, BEFORE the presentation shuffle, so the
    # assignment cannot be scrambled by reordering.
    for s in STRATA:
        group = [p for p in chosen if p["stratum"] == s]
        rng.shuffle(group)
        for k, p in enumerate(group):
            p["swapped"] = k < len(group) // 2
    rng.shuffle(chosen)   # presentation order only

    salt = f"{args.seed}-{len(chosen)}"
    items, key = [], []
    for p in chosen:
        swap = p["swapped"]
        iid = item_id(p["site"], p["i"], p["j"], salt)
        left, right = (p["file_b"], p["file_a"]) if swap else (p["file_a"], p["file_b"])
        items.append({"id": iid,
                      "left": f"../seq_data_raw/{p['site']}/{left}",
                      "right": f"../seq_data_raw/{p['site']}/{right}"})
        key.append({"id": iid, **p,
                    # the correct answer under A's labels: which side is later
                    "truth": ("left" if swap else "right")
                             if p["stage_a"] != p["stage_b"] else "same"})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "key.json").write_text(json.dumps(
        {"seed": args.seed, "salt": salt, "n": len(items),
         "note": "SEALED. Opening this before annotating destroys the blinding.",
         "items": key}, indent=1) + "\n", encoding="utf-8")
    (OUT / "task.json").write_text(json.dumps(items, indent=1) + "\n",
                                   encoding="utf-8")
    write_task_html(items)

    n_swap = sum(1 for k in key if k["swapped"])
    n_int = sum(1 for k in key if k["stratum"] == "interior")
    print(f"\nprepared {len(items)} items")
    print(f"  planted reversals (frames swapped) : {n_swap}")
    print(f"  interior-stratum pairs             : {n_int}")
    print(f"  boundary-stratum pairs             : {len(items) - n_int}")
    print(f"\n  task : {OUT / 'task.html'}   <- open this, answer, click Save")
    print(f"  key  : {OUT / 'key.json'}     <- sealed; do not open first")
    print("\nThe annotator must not read key.json, task.json, or any labels.csv")
    print("before finishing. Nothing else about the study needs withholding.")
    return 0


def write_task_html(items: list[dict]) -> None:
    """A single local page. No network, no build step, no dependencies.

    The annotator opens it from the filesystem, answers each pair, and clicks
    Save to get answers.csv. Progress is kept in localStorage so closing the tab
    does not lose work -- a 200-item task is not finished in one sitting, and
    losing it halfway is the fastest way to end up with a rushed second half.
    """
    payload = json.dumps(items)
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pairwise Reversal Test</title>
<style>
 :root { color-scheme: light dark; --bg:#fff; --fg:#111; --mut:#666; --line:#ddd;
         --btn:#f4f4f4; --sel:#1a5fb4; }
 @media (prefers-color-scheme: dark) { :root {
   --bg:#15171a; --fg:#e8e8e8; --mut:#9aa; --line:#333; --btn:#23262b; } }
 body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5
        system-ui,-apple-system,Segoe UI,sans-serif; }
 header { padding:14px 16px; border-bottom:1px solid var(--line);
          display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
 h1 { font-size:16px; margin:0; font-weight:600; }
 .mut { color:var(--mut); font-size:13px; }
 main { padding:16px; max-width:1100px; margin:0 auto; }
 .pair { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
 .pair figure { margin:0; }
 .pair img { width:100%; border:1px solid var(--line); border-radius:6px;
             display:block; background:var(--btn); }
 .pair figcaption { text-align:center; font-weight:600; padding:6px 0; }
 .btns { display:flex; gap:8px; flex-wrap:wrap; margin:16px 0; }
 button { font:inherit; padding:10px 14px; border:1px solid var(--line);
          border-radius:6px; background:var(--btn); color:var(--fg);
          cursor:pointer; }
 button:hover { border-color:var(--sel); }
 button.save { background:var(--sel); color:#fff; border-color:var(--sel); }
 .q { font-weight:600; margin:14px 0 0; }
 @media (max-width:700px) { .pair { grid-template-columns:1fr; } }
</style></head><body>
<header>
  <h1>Which frame is later in the construction?</h1>
  <span class="mut" id="prog"></span>
  <span class="mut">Answer from the images only. Do not open any label file.</span>
</header>
<main>
  <div class="pair">
    <figure><figcaption>LEFT</figcaption><img id="imgL" alt="left frame"></figure>
    <figure><figcaption>RIGHT</figcaption><img id="imgR" alt="right frame"></figure>
  </div>
  <p class="q">Which shows the building at a LATER point in construction?</p>
  <div class="btns">
    <button data-a="left">LEFT is later</button>
    <button data-a="right">RIGHT is later</button>
    <button data-a="same">Same stage</button>
    <button data-a="cannot">Cannot tell</button>
  </div>
  <div class="btns">
    <button data-nav="-1">&larr; Back</button>
    <button class="save" data-nav="save">Save answers.csv</button>
  </div>
  <p class="mut">A "reversal" answer is not a wrong answer. Some pairs are shown
  in a deliberately swapped order, so calling one backwards is sometimes exactly
  right. Answer what you see.</p>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "blind_pairs_answers_v1";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let at = ITEMS.findIndex(it => !(it.id in answers));
if (at < 0) at = ITEMS.length - 1;

function save() {
  try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {}
}
function render() {
  const it = ITEMS[at];
  document.getElementById("imgL").src = it.left;
  document.getElementById("imgR").src = it.right;
  const done = Object.keys(answers).length;
  document.getElementById("prog").textContent =
    `item ${at + 1} of ${ITEMS.length} — ${done} answered` +
    (answers[it.id] ? ` — this one: ${answers[it.id]}` : "");
}
document.querySelectorAll("button[data-a]").forEach(b => b.onclick = () => {
  answers[ITEMS[at].id] = b.dataset.a;
  save();
  if (at < ITEMS.length - 1) at++;
  render();
});
document.querySelectorAll("button[data-nav]").forEach(b => b.onclick = () => {
  if (b.dataset.nav === "save") {
    const rows = ["item_id,answer"];
    for (const it of ITEMS) if (answers[it.id]) rows.push(it.id + "," + answers[it.id]);
    const blob = new Blob([rows.join("\\n") + "\\n"], {type: "text/csv"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "answers.csv";
    a.click();
    return;
  }
  at = Math.max(0, Math.min(ITEMS.length - 1, at + Number(b.dataset.nav)));
  render();
});
render();
</script></body></html>
"""
    (OUT / "task.html").write_text(html.replace("__ITEMS__", payload),
                                   encoding="utf-8")


def score(args) -> int:
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    by_id = {k["id"]: k for k in key["items"]}
    answers = {}
    with open(args.answers, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answers[row["item_id"]] = row["answer"].strip().lower()
    print(f"{len(answers)} of {key['n']} items answered\n")
    if not answers:
        raise SystemExit("no answers to score")

    # ---- power: were the planted reversals detected? -----------------------
    planted = [k for k in key["items"]
               if k["swapped"] and k["stratum"] == "boundary"
               and k["id"] in answers]
    hit = sum(1 for k in planted if answers[k["id"]] == "left")
    print("POWER — pairs shown swapped across a forward stage boundary.")
    print("The later frame is on the LEFT in these, so 'LEFT is later' is the")
    print("detection. Without detections here, a null result below means nothing.")
    print(f"  planted and answered : {len(planted)}")
    print(f"  detected             : {hit}"
          f"{f' ({100 * hit / len(planted):.0f}%)' if planted else ''}")

    # ---- the finding: reversals inside a single stage label ----------------
    interior = [k for k in key["items"]
                if k["stratum"] == "interior" and k["id"] in answers]
    # For an interior pair A called one stage, order is undetermined under A's
    # labels; a confident directional answer means the annotator SAW a change A
    # could not record. When the pair was swapped, "left is later" is forward and
    # "right is later" is the reversal; unswapped, the reverse.
    rev = 0
    for k in interior:
        a = answers[k["id"]]
        if a in ("same", "cannot"):
            continue
        backwards = (a == "right") if k["swapped"] else (a == "left")
        rev += 1 if backwards else 0
    print("\nFINDING — pairs annotator A placed inside ONE stage, where a")
    print("reversal could not have been recorded by the taxonomy:")
    print(f"  answered             : {len(interior)}")
    print(f"  called a reversal    : {rev}")
    print(f"  called same/unclear  : "
          f"{sum(1 for k in interior if answers[k['id']] in ('same', 'cannot'))}")

    # ---- adjudication queue ------------------------------------------------
    queue = []
    for k in key["items"]:
        if k["id"] not in answers:
            continue
        a = answers[k["id"]]
        reason = None
        planted = k["swapped"] and k["stratum"] == "boundary"
        # Order matters. A missed plant also "contradicts annotator A", because
        # the plant was built from A's own forward boundary -- but calling it a
        # contradiction buries the thing that makes the null interpretable, so
        # the planted check is tested FIRST.
        if planted and a != "left":
            reason = ("planted direction not detected"
                      if a != "cannot" else
                      "planted direction: annotator could not tell")
        elif a == "cannot":
            reason = "annotator could not tell"
        elif k["truth"] != "same" and a in ("left", "right") and a != k["truth"]:
            reason = "contradicts annotator A"
        if reason:
            queue.append({"id": k["id"], "site": k["site"],
                          "file_a": k["file_a"], "file_b": k["file_b"],
                          "stage_a": k["stage_a"], "stage_b": k["stage_b"],
                          "blind_answer": a, "reason": reason})
    (OUT / "adjudication.json").write_text(json.dumps(queue, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"\nADJUDICATION — {len(queue)} items for a third reading under the")
    print("frozen protocol, written to blind_study/adjudication.json:")
    for reason, n in collections.Counter(q["reason"] for q in queue).items():
        print(f"  {reason:34s} {n}")
    print("\nThe adjudicator sees the protocol and the pair, never the blind")
    print("answer or annotator A's label.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="sample pairs and write the task page")
    p.add_argument("--data", default="seq_data")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=prepare)
    s = sub.add_parser("score", help="score a completed answers.csv")
    s.add_argument("--answers", default=str(OUT / "answers.csv"))
    s.set_defaults(func=score)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
