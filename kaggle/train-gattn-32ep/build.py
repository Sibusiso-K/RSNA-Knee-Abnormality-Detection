"""Build the pre-registered one-fold, 32-epoch group-attention probe."""
import json
from pathlib import Path
import sys

config = Path(__file__).resolve().parent
repo = config.parents[1]
source = (repo / "notebooks/kaggle_06_train_slots.py").read_text(encoding="utf-8")
replacements = {
    'os.environ.get("HEAD", "slot")': 'os.environ.get("HEAD", "gattn")',
    'os.environ.get("BATCH", "8")': 'os.environ.get("BATCH", "4")',
    'os.environ.get("EPOCHS", "10")': 'os.environ.get("EPOCHS", "32")',
}
for old, new in replacements.items():
    assert source.count(old) == 1, f"missing or ambiguous marker: {old}"
    source = source.replace(old, new)
for expected in ('head=HEAD', 'ALL_GROUPS = HEAD == "gattn"', 'take_input', 'xm.mark_step()'):
    assert expected in source, f"required gattn behavior missing: {expected}"
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
(output / "script.py").write_text(source, encoding="utf-8")
(output / "kernel-metadata.json").write_text((config / "kernel-metadata.json").read_text(), encoding="utf-8")
print(f"Built 32-epoch gattn fold-0 probe -> {output}")
