"""Report -> label extraction.

The competition gives real per-condition labels for only ~58 studies; every other
label has to come out of a free-text, multilingual radiology report. This package
is that extractor. See docs/04-method.md for why it is the highest-leverage part
of the whole project.
"""

from src.extract.rules import ExtractorConfig, RuleExtractor
from src.extract.types import Mention, StudyExtraction

__all__ = ["ExtractorConfig", "Mention", "RuleExtractor", "StudyExtraction"]
