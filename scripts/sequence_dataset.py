"""
SequenceFrameDataset — ordered frame sequences with per-frame stage labels.

Unlike the Milestone-1 ImageFolder (which treats frames independently and loses
temporal order), the sequence-constraint layer needs frames *in time order* grouped
by sequence (e.g. one time-lapse of one building).

Expected layout
---------------
    seq_data/
      train/
        seq0001/
          labels.csv        # rows: path,stage_index   (0-based, construction order)
        seq0002/ ...
      val/
        ...

`stage_index` must use the SAME ordering as the baseline classes (0 = earliest).
Files are read in the order given by labels.csv (which should be temporal order).

Path convention
---------------
The first column is resolved **relative to the directory holding that
labels.csv**, with forward slashes. Frames are not duplicated into the split
dirs, so a split's labels.csv points back at the raw corpus:

    seq_data/train/seq0003/labels.csv
        ../../../seq_data_raw/seq0003/frames/000000.jpg,0

while the raw sequence keeps its frames alongside:

    seq_data_raw/seq0003/labels.csv
        frames/000000.jpg,0

Backslashes are normalised on read, so label files authored on Windows load on
POSIX. Absolute paths still resolve, but they are not portable and the released
corpus must not contain any -- see `src/data_collection/check_label_paths.py`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def list_sequence_dirs(root) -> list[Path]:
    """Every labeled sequence directory under `root`, recursing through splits.

    Accepts either a split dir (``seq_data/train``) or a corpus root holding
    several splits (``seq_data``), so cross-validation can pool all sites and
    re-partition them itself.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    if (root / "labels.csv").exists():
        return [root]
    found: list[Path] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        found.extend(list_sequence_dirs(child))
    return found


class SequenceFrameDataset(Dataset):
    """Ordered frame sequences. Built either from a split dir or an explicit
    list of sequence dirs (``sequence_dirs=``), the latter for cross-validation."""

    def __init__(
        self,
        split_dir=None,
        img_size: int = 224,
        transform=None,
        max_len: int | None = None,
        sequence_dirs=None,
    ):
        self.transform = transform or default_transform(img_size)
        self.max_len = max_len
        self.sequences: list[list[tuple[Path, int]]] = []
        self.names: list[str] = []  # site id per sequence, for per-fold reporting

        if sequence_dirs is not None:
            self.split_dir = None
            seq_dirs = [Path(d) for d in sequence_dirs]
        elif split_dir is not None:
            self.split_dir = Path(split_dir)
            if not self.split_dir.is_dir():
                raise FileNotFoundError(f"Split dir not found: {self.split_dir}")
            seq_dirs = sorted(p for p in self.split_dir.iterdir() if p.is_dir())
        else:
            raise ValueError("pass either split_dir or sequence_dirs")

        for seq_dir in seq_dirs:
            csv_path = seq_dir / "labels.csv"
            if not csv_path.exists():
                continue
            frames: list[tuple[Path, int]] = []
            with open(csv_path, newline="") as f:
                for row in csv.reader(f):
                    if not row or row[0].strip().lower() in {"filename", "file", ""}:
                        continue  # skip header / blank rows
                    fname, stage = row[0].strip(), int(row[1])
                    if stage < 0:
                        continue  # -1 = invalid/unlabeled frames (intros, outros)
                    # normalise Windows separators so label files written on
                    # Windows resolve on POSIX; an absolute path still wins the
                    # join, which is how legacy label files keep working.
                    frames.append((seq_dir / fname.replace("\\", "/"), stage))
            if frames:
                self.sequences.append(frames)
                self.names.append(seq_dir.name)

        if not self.sequences:
            where = self.split_dir if sequence_dirs is None else f"{len(seq_dirs)} given dirs"
            raise RuntimeError(f"No sequences with labels.csv found under {where}")

    def stage_counts(self) -> dict[int, int]:
        """Frames per stage over the whole dataset (class-balance reporting)."""
        counts: dict[int, int] = {}
        for frames in self.sequences:
            for _, stage in frames:
                counts[stage] = counts.get(stage, 0) + 1
        return dict(sorted(counts.items()))

    def gold_order_violation_rate(self) -> float:
        """Backward steps in the *ground-truth* labels.

        Non-zero means the corpus contains genuine re-work (or label noise) —
        i.e. exactly the evidence that makes a hard monotonic constraint wrong
        and a learnable gate necessary.
        """
        bad = total = 0
        for frames in self.sequences:
            stages = [s for _, s in frames]
            for a, b in zip(stages[:-1], stages[1:]):
                total += 1
                bad += int(b < a)
        return bad / max(1, total)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frames = self.sequences[idx]
        if self.max_len is not None and len(frames) > self.max_len:
            # even temporal subsampling — truncation would drop the late stages
            step = (len(frames) - 1) / (self.max_len - 1)
            frames = [frames[round(i * step)] for i in range(self.max_len)]
        imgs, labels = [], []
        for path, stage in frames:
            img = Image.open(path).convert("RGB")
            imgs.append(self.transform(img))
            labels.append(stage)
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


def pad_collate(batch):
    """Pad variable-length sequences to the batch max; return frames, labels, mask."""
    max_t = max(x.shape[0] for x, _ in batch)
    C, H, W = batch[0][0].shape[1:]
    B = len(batch)
    frames = torch.zeros(B, max_t, C, H, W)
    labels = torch.zeros(B, max_t, dtype=torch.long)
    mask = torch.zeros(B, max_t)
    for i, (x, y) in enumerate(batch):
        t = x.shape[0]
        frames[i, :t] = x
        labels[i, :t] = y
        mask[i, :t] = 1.0
    return frames, labels, mask


def scored_stages(sequence_dir, max_len: int | None = 64) -> list[int]:
    """The stages present in the gold sequence the evaluation ACTUALLY scores.

    This is NOT the same as the stages present in the labelled sequence, and the
    difference is a real denominator bug that this function exists to prevent.

    `eval_stage_counts` in a result record is a histogram over every labelled
    frame of the held-out site. The evaluation, however, scores an even 64-frame
    subsample, and a stage occupying very few frames can fail to survive it:
    seq0005 carries stage 1 on 5 of its 830 labelled frames, and no subsampled
    position lands on them, so the scored gold contains four stages while the
    full sequence contains five.

    Any per-stage rate -- onset coverage above all -- must use THIS denominator.
    Using the full-sequence stage set counts a stage the model could not have
    predicted because it was never in the gold it was scored against. Across the
    corpus that is exactly one stage per seed, three terms in a 39-run grid, and
    it moves onset coverage by at most 0.005; small, but wrong in a knowable
    direction, since it can only understate coverage.
    """
    import csv as _csv
    from pathlib import Path as _Path

    rows: list[int] = []
    with open(_Path(sequence_dir) / "labels.csv", newline="", encoding="utf-8") as f:
        for row in _csv.reader(f):
            if not row or row[0].strip().lower() in {"filename", "file", ""}:
                continue
            stage = int(row[1])
            if stage >= 0:
                rows.append(stage)
    if max_len is not None and len(rows) > max_len:
        step = (len(rows) - 1) / (max_len - 1)
        rows = [rows[round(i * step)] for i in range(max_len)]
    return sorted(set(rows))
