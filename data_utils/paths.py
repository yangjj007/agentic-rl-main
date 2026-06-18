"""Project-relative data paths for DyME."""
import glob
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


def resolve_model_path(path: str) -> str:
    """Expand ``~`` / relative paths for local dirs; leave HuggingFace repo ids unchanged."""
    if not path:
        return path
    raw = path.strip()
    if not raw:
        return raw
    expanded = os.path.expanduser(raw)
    if os.path.isdir(expanded) or os.path.isfile(expanded):
        return os.path.abspath(expanded)
    if raw.startswith("~") or os.path.isabs(raw):
        return os.path.abspath(expanded)
    return raw


def find_model_weight_files(model_dir: str) -> list[str]:
    """Return weight file paths under a local model directory (HF or ModelScope layout)."""
    patterns = [
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.join(model_dir, pattern)))
    return sorted(set(found))


def validate_local_model_dir(path: str, *, role: str = "model") -> str:
    """
    Ensure a local model directory contains loadable weights for ``from_pretrained``.

    HuggingFace repo ids (e.g. ``llava-hf/...``) are returned unchanged.
    """
    resolved = resolve_model_path(path)
    if not os.path.isdir(resolved):
        return resolved
    if not find_model_weight_files(resolved):
        raise FileNotFoundError(
            f"{role} path '{resolved}' has no model.safetensors / pytorch_model*.bin. "
            "Download weights first (ModelScope --exclude 'onnx/*' or hf download)."
        )
    return resolved


def local_pretrained_kwargs(path: str) -> dict:
    """Use local_files_only for local model directories; no-op for Hub IDs."""
    resolved = resolve_model_path(path)
    if os.path.isdir(resolved):
        return {"local_files_only": True}
    return {}


def discover_local_model(role: str, default: str) -> str:
    """Resolve role-specific model override env vars, otherwise return default."""
    key = f"DYME_{role.upper()}_MODEL"
    override = os.environ.get(key, "").strip()
    if override:
        return validate_local_model_dir(override, role=role)
    return default
