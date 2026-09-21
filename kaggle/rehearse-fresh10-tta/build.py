"""Rehearsal of the TTA submission over ~1,400 training studies: runtime + parity.

Usage: python kaggle/rehearse-fresh10-tta/build.py OUTPUT_DIRECTORY
Push with --accelerator NvidiaTeslaT4 explicit. Output is rehearsal_predictions.csv,
never submission.csv.
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tta_builder", Path(__file__).resolve().parents[1] / "submit-fresh10-24ep-prob-tta/build.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == "__main__":
    # optional 2nd arg: number of studies (default 1400; 192 = parity-only quick pass)
    mod.build(sys.argv[1], rehearse=True, n_studies=int(sys.argv[2]) if len(sys.argv) > 2 else mod.N_REHEARSAL)
