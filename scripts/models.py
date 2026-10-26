from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# =========================================================
# DOCUMENT METADATA
# =========================================================

@dataclass
class PlanMetadata:
    """Plan identifiers displayed above each worksheet's asset table."""

    plan_name: str = ""
    ein: str = ""
    plan_number: str = ""
    plan_year: str = ""


# =========================================================
# ROW MODEL
# =========================================================

@dataclass
class ScheduleRow:
    """Container for one extracted Schedule H line item."""

    identity: str
    description: str
    cost: Optional[str]
    current_value: str

    def to_list(self, include_cost: bool) -> List[str]:
        """Return the row in the column order expected by Excel output."""
        if include_cost:
            return [
                self.identity,
                self.description,
                self.cost or "",
                self.current_value,
            ]

        return [
            self.identity,
            self.description,
            self.current_value,
        ]


# =========================================================
# PAGE RESULT
# =========================================================

@dataclass
class SchedulePageResult:
    """Extraction result for a single PDF page."""
    page_number: int
    rows: List[ScheduleRow]
    has_cost_column: bool
    source: str  # "TEXT" or "OCR"
    is_skewed: bool = False
    legend: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Report whether the page contributed table rows or legend text."""
        return len(self.rows) == 0 and len(self.legend) == 0


# =========================================================
# FULL EXTRACTION RESULT
# =========================================================

@dataclass
class ExtractionResult:
    """Top-level extraction payload returned by each pipeline branch."""
    plan_name: str
    pages: List[SchedulePageResult]

    def is_empty(self) -> bool:
        """Report whether every page result is empty."""
        return all(page.is_empty() for page in self.pages)
