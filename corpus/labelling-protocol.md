# Construction-Stage Labelling Protocol

**Version 1.2 — 2026-08-09.** This is the authoritative protocol for the corpus.
The paper states that the labelling protocol is released; this document is what
that refers to. `src/data_collection/label_tool.py` implements it.

Version 1.1 added the **reversal rule** R1 (§4), which v1.0 lacked. Sequences
labelled under v1.0 remain valid: none of them contains a deliberate removal, so
the rule changes no existing label.

Version 1.2 adds **R2, the non-current-footage rule** (§4.2), after the
second-annotator pass found that the protocol had no rule for retrospective
montage — the source of the only *apparent* reversal in the four sequences
checked. R2 changes no existing label either: both annotators had already
excluded the frame in question by instinct. It is written down so the next
annotator does not have to guess.

Record the protocol version alongside each sequence.

---

## 1. Taxonomy

Five stages, in construction order:

| Index | Stage | The site shows |
|---|---|---|
| 0 | `site_work` | clearing, demolition, excavation, grading, services trenching — ground is being prepared, nothing is being built up |
| 1 | `foundation` | footings, foundation walls, slab, formwork and pours below the ground floor |
| 2 | `structure` | load-bearing frame: floor framing, walls going up, columns, beams, roof framing |
| 3 | `envelope_walls` | the building is being closed: sheathing, weather barrier, windows, doors, roof covering, cladding |
| 4 | `finishing` | interior and final works: services rough-in, plaster, floors, fittings, external finishes |
| −1 | *excluded* | not assessable — title cards, intros, outros, logos, black frames, footage where the works are not visible |

The stage set is **ordered but not a mandatory traversal**. A sequence may begin
at any stage and may never reach the last. A renovation that starts with a
standing building begins at 3 or 4; a sequence that ends at roofing never reaches
4. Neither is a labelling error.

---

## 2. Governing principle — the fresh-viewer test

> **Label each frame as you would if you had never seen the earlier frames.**

Assign the stage that describes the *state of the works visible now*, not the
stage the project has historically reached. This is the single rule from which the
others follow, and it exists for a concrete reason: the model sees one frame at a
time, so a label that depends on unseen history is a label the model cannot
learn.

Applying it consistently is more important than any individual judgement call.

---

## 3. Boundary-based method

Do not label every frame. Scrub the sequence, and at each frame where the stage
**changes**, press the number key for the new stage. Every later frame inherits
that stage until the next mark. A 300-frame sequence typically needs 4–8 marks.

```bash
python src/data_collection/label_tool.py --seq seq_data_raw/seqNNNN
```

Mark the **first** frame that shows the new stage, not the last frame of the old
one. Where the change is gradual, mark the first frame at which you would answer
the fresh-viewer test with the new stage.

---

## 4. When the footage appears to go backwards

Two different things make a sequence look non-monotone, and they need opposite
treatment. **R1** covers the case where the site genuinely went backwards: label
the reversal. **R2** covers the case where the *editing* went backwards while the
site did not: exclude the frames. Confusing the two contaminates the corpus
statistic that this whole corpus exists to measure.

### 4.1 R1 — the reversal rule  *(added in v1.1)*

Construction does not usually run backwards, but it sometimes genuinely does, and
those cases are the reason this corpus exists. The rule:

> **R1 — Deliberate removal reverts the stage.** When completed work is
> deliberately removed — a building demolished, a roof covering stripped,
> cladding taken off, a defective pour broken out — label the affected frames
> with the stage the site would be assigned by a fresh viewer *after* the
> removal. Advance again when the work is re-installed.

This is not a new principle; it is the fresh-viewer test applied to a site that
has gone backwards. A demolished house is site work, whether or not a house stood
there yesterday.

#### What R1 does **not** cover

R1 fires only on **deliberate removal of completed work**. It must not be used
for:

- **Occlusion** — scaffolding, sheeting, a delivery lorry, or a crane hiding the
  works. The stage has not changed; the view has. If the works are not assessable,
  the frame is `−1`, not an earlier stage.
- **Weather and season** — snow cover, mud, poor light.
- **Camera position** — a roaming camera that pans from a finished room to
  ongoing groundworks has not observed a reversal. It has observed two parts of
  one site. Sequences whose viewpoint moves between areas at different stages
  should be rejected at screening (see `docs/reality-track/datasets.md`), because
  they generate spurious reversals.
- **Normal sequencing of trades** — an element being installed in one area while
  another area is further ahead is not re-work.

### 4.2 R2 — the non-current-footage rule  *(added in v1.2)*

Edited construction video routinely shows footage that is not the current state
of the works: a "BEFORE" shot of the bare site, a recap of earlier phases, a
before/after comparison, a flashback illustrating a point. The site did not go
backwards; the edit did.

> **R2 — Footage that does not show the current state of the works is excluded.**
> Label `−1` for retrospective, comparative or recap footage — anything the video
> presents as an *earlier* or *other* state rather than as progress now. Do not
> label it by what it depicts.

**Exclude the whole montage block, not only the frames that obviously look
earlier.** A before/after sequence intercuts past and present, often without a
caption on every frame, and deciding shot by shot which is which is exactly the
judgement a fresh viewer cannot make reliably. Once a video enters a closing
comparison or a recap, mark `−1` from the first frame of the block to the end of
it. Erring toward exclusion costs a few frames; erring toward inclusion
manufactures a reversal that never happened.

**Why this is not just tidiness.** Labelled by what it depicts, a "BEFORE" shot
of a bare field placed after the finished building produces a `4 → 0` transition
— a backward step of four stages, in a sequence where nothing was ever removed.
A single such frame is enough to put a spurious reversal into the corpus
statistic, and the order-violation rate is the number on which the choice between
a rigid and a learnable ordering constraint turns.

**What R2 does not cover.** It applies to footage the *video* presents as
non-current. It is not licence to exclude a frame merely because the works look
less advanced than the previous frame — that is what R1 is for, and genuine
re-work must be labelled, not excluded. If the works really are being taken
apart, R1 applies and the stage reverts.

### 4.3 Granularity limit — state this, do not work around it

R1 can only express a reversal that **crosses a stage boundary**. Real re-work
often does not. Stripping a tiled roof back to its weather membrane is a genuine
reversal of real work, but both states fall inside stage 3, so the five-stage
taxonomy records no change.

**Do not stretch a stage boundary to capture such an event.** Label it by the
fresh-viewer test like any other frame, and record the event in the sequence's
`provenance.json` under `notes` so that it is available if a finer taxonomy is
adopted later. A corpus whose reversals were manufactured by inconsistent
labelling is worse than one with fewer reversals.

---

## 5. Worked examples

**`seq0020` — fire total-loss, demolition and rebuild.** The house stands intact
(4), is demolished (**R1 fires**: 4 → 0), the lot is cleared and excavated (0),
foundations are poured (1), the floor and walls are framed (2), windows and
roofing go on (3). The sequence ends before interior finishing, so it never
returns to 4. One stage-crossing reversal.

**`seq0022` — stone barn, roof re-covered after storm damage.** The narration
records that all newly laid tiles were removed after a storm and the roof left
bare before being re-tiled. This is genuine re-work, but stripping tiles back to
the weather membrane stays inside stage 3, so **no stage-crossing reversal is
labelled**. Per §4.3, the event is recorded in `provenance.json` rather than
forced into a stage change.

**`seq0018` — earthen masonry house, closing before/after montage (R2).** The
build finishes at stage 4 around frame 131. Frame 132 is captioned "BEFORE" and
shows the bare field; frames 133–141 alternate "AFTER" shots of the finished
building with close-ups of planting; 142–144 are end cards. Labelled by what they
depict, frame 132 alone yields a `4 → 0` transition.

Both annotators in the agreement study excluded frame 132 independently, before
R2 existed — but they then diverged on the rest, one excluding the whole block
from 132 to the end, the other re-admitting the "AFTER" frames as stage 4. Both
readings were defensible under v1.1, which is why R2 now fixes the answer:
**exclude from the first frame of the montage to the end of it.** Under R2 the
whole of 132–144 is `−1`.

> The released `labels.B.csv` preserves the v1.1 reading **as measured**. It is
> deliberately not retrofitted to R2: a rule derived from a disagreement must not
> then be used to erase that disagreement. In this instance retrofitting would
> change no published figure anyway — annotator A had already excluded those
> frames, so they never entered the per-frame agreement calculation.

---

## 6. Quality rules

1. **One annotator per pass.** For agreement studies, a second annotator works
   from the raw frames only, never from the first annotator's marks
   (`--annotator B` writes a separate file).
2. **Mark `−1` generously.** An uncertain frame excluded costs one frame; an
   uncertain frame guessed costs a wrong label the model must fit.
3. **Report the gold order-violation rate** for every sequence and for the corpus.
   It determines whether the corpus can test an ordering constraint at all, and it
   costs nothing to compute.
4. **Record the protocol version** used for each sequence.
