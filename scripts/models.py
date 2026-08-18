from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# =========================================================
# ROW MODEL
# =========================================================

@dataclass
class ScheduleRow:
    """
    Represents a single extracted Schedule H row.
    """

    identity: str
    description: str
    cost: Optional[str]
    current_value: str

    def to_list(self, include_cost: bool) -> List[str]:
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
    page_number: int
    rows: List[ScheduleRow]
    has_cost_column: bool
    source: str  # "TEXT" or "OCR"

    def is_empty(self) -> bool:
        return len(self.rows) == 0


# =========================================================
# FULL EXTRACTION RESULT
# =========================================================

@dataclass
class ExtractionResult:
    plan_name: str
    pages: List[SchedulePageResult]

    def is_empty(self) -> bool:
        return all(page.is_empty() for page in self.pages)