"""Copy the task's frames to anonymous filenames — blinding the file paths.

The sampler references frames at their real locations, which look like
`seq_data_raw/seq0005/frames/000412.jpg`. Those paths carry two things an
annotator must not have: the SITE a pair came from, which would let items be
grouped and the sequences partly reconstructed, and the FRAME NUMBER, which
gives the true temporal order directly and would make the whole task trivial.

Each image is therefore copied to `blind_study/frames/<item_id>_L.jpg` and
`_R.jpg`, where the item id is the hash the sampler already assigned. The
mapping back lives only in the sealed key.

The copies stay outside the released package: the source videos are under the
YouTube Standard License and carry no redistribution rights, so frames are not
published anywhere, here or in the repository.

    python src/data_collection/anonymise_blind_frames.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "blind_study"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="blind_study/frames")
    args = ap.parse_args()

    items = json.loads((STUDY / "task.json").read_text(encoding="utf-8"))
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    mapping, missing, total = {}, [], 0
    for it in items:
        entry = {}
        for side, key in (("L", "left"), ("R", "right")):
            src = (STUDY / it[key]).resolve()
            if not src.is_file():
                missing.append(str(src))
                continue
            dst = out / f"{it['id']}_{side}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            entry[side] = {"anonymised": dst.name, "source": str(src)}
            total += 1
        mapping[it["id"]] = entry

    (STUDY / "frame_map.json").write_text(
        json.dumps({"note": "SEALED. Maps anonymised frames back to their "
                            "source files; opening it before annotating "
                            "destroys the blinding.",
                    "frames": mapping}, indent=1) + "\n", encoding="utf-8")

    print(f"copied {total} frames to {out}")
    if missing:
        print(f"MISSING {len(missing)}:")
        for m in missing[:5]:
            print("  ", m)
        return 1
    names = [p.name for p in out.glob("*")]
    leaks = [n for n in names if "seq" in n.lower() or n[:12].isdigit()]
    print(f"filenames carrying a site or frame number: {len(leaks)}")
    print("mapping written to blind_study/frame_map.json (sealed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
