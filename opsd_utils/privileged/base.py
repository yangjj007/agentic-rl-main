from abc import ABC, abstractmethod
from typing import Any, Optional


class PrivilegedContextProvider(ABC):
    """Build teacher-only privileged context from a DyME training sample."""

    @abstractmethod
    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        """Text appended to the teacher user message (invisible to the student)."""

    def build_teacher_images(self, sample: dict[str, Any]) -> Optional[Any]:
        """Optional alternate visual input for the teacher (e.g. crop)."""
        return None
