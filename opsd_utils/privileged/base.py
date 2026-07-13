from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class PrivilegedContextProvider(ABC):
    """Build teacher-only privileged context from a DyME training sample."""

    @abstractmethod
    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        """Text appended to the teacher user message (invisible to the student)."""

    def build_teacher_images(self, sample: dict[str, Any]) -> list[Image.Image]:
        """Optional extra teacher images (e.g. crop). Default: none."""
        return []
