"""Trivial baseline: writes a constant-0.5 submission for every target.

Not a real model — just confirms the data is present and the submission
format is correct before building anything smarter.
"""

from pathlib import Path

import pandas as pd

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "submissions"


def main() -> None:
    sample_path = DATA_DIR / "sample_submission.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"{sample_path} not found — run scripts/download_data.sh first."
        )

    sample = pd.read_csv(sample_path)
    for target in TARGETS:
        sample[target] = 0.5

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "submission.csv"
    sample.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(sample)} rows)")


if __name__ == "__main__":
    main()
