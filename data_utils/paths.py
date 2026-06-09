"""Project-relative data paths for DyME."""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_IMAGES_DIR = os.path.join(DATA_DIR, "images")

CHARTQA_DIR = os.path.join(DATA_IMAGES_DIR, "chartqa")
CHARTQA_IMAGES_DIR = os.path.join(CHARTQA_DIR, "images")
CHARTQA_JSON_DIR = os.path.join(CHARTQA_DIR, "json")

AOKVQA_DIR = os.path.join(DATA_IMAGES_DIR, "aokvqa")
AOKVQA_IMAGES_DIR = os.path.join(AOKVQA_DIR, "images")
AOKVQA_JSON_DIR = os.path.join(AOKVQA_DIR, "json")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Legacy absolute prefixes in JSON files -> canonical project directories
_LEGACY_PREFIX_MAP = [
    ("/chartqa_output/", CHARTQA_DIR + os.sep),
    ("/path/to/chartqa_output/", CHARTQA_DIR + os.sep),
    ("/path/to/data/chartqa_output/", CHARTQA_DIR + os.sep),
    ("/path/to/data/aokvqa/", AOKVQA_DIR + os.sep),
]


def project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_image_path(path: str) -> str:
    """Resolve a stored image path to an existing file under the project tree."""
    if not path:
        return path
    if os.path.exists(path):
        return path

    candidates = []
    for old, new in _LEGACY_PREFIX_MAP:
        if old in path:
            candidates.append(path.replace(old, new))

    if not os.path.isabs(path):
        candidates.append(os.path.join(PROJECT_ROOT, path))

    basename = os.path.basename(path)
    candidates.extend([
        os.path.join(CHARTQA_IMAGES_DIR, basename),
        os.path.join(AOKVQA_IMAGES_DIR, basename),
        os.path.join(PROJECT_ROOT, "chartqa_output", "images", basename),
    ])

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return path
