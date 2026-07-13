from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ArtifactSpec:
    id: str
    title: str
    kind: str
    description: str
    outputs: tuple[str, ...]
    tags: tuple[str, ...] = ()
    producer: Callable[[Any], None] | None = None
    aliases: tuple[str, ...] = ()


class ArtifactRegistry:
    def __init__(self) -> None:
        self.specs: list[ArtifactSpec] = []
        self._by_name: dict[str, ArtifactSpec] = {}

    def register(self, spec: ArtifactSpec) -> None:
        self.specs.append(spec)
        for name in (spec.id, *spec.aliases):
            self._by_name[name] = spec

    def resolve(self, targets: set[str]) -> list[ArtifactSpec]:
        if not targets or "all" in targets:
            return [spec for spec in self.specs if spec.producer is not None]
        resolved: list[ArtifactSpec] = []
        seen: set[str] = set()
        for target in sorted(targets):
            spec = self._by_name.get(target)
            if spec is None:
                raise ValueError(f"unknown artifact target: {target}")
            if spec.producer is not None and spec.id not in seen:
                resolved.append(spec)
                seen.add(spec.id)
        return resolved

    def run(self, targets: set[str], ctx: Any) -> list[ArtifactSpec]:
        selected = self.resolve(targets)
        for spec in selected:
            assert spec.producer is not None
            spec.producer(ctx)
        return selected


def _output_entry(path: str) -> dict[str, str]:
    suffix = Path(path).suffix.lower().lstrip(".")
    kind = "data"
    if suffix in {"png", "pdf", "svg"}:
        kind = "figure"
    elif suffix in {"md", "txt"}:
        kind = "text"
    elif suffix in {"html"}:
        kind = "html"
    return {"path": path, "kind": kind}


def write_artifacts_manifest(*, registry: ArtifactRegistry, out_dir: Path) -> None:
    payload = {
        "artifacts": [
            {
                "id": spec.id,
                "title": spec.title,
                "kind": spec.kind,
                "description": spec.description,
                "tags": list(spec.tags),
                "outputs": [_output_entry(path) for path in spec.outputs],
            }
            for spec in registry.specs
        ]
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_dashboard_html(*, out_dir: Path) -> None:
    manifest_path = out_dir / "artifacts_manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else '{"artifacts":[]}'
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Routed OPD Artifact Dashboard</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #222; background: #f7f7f5; }}
    header {{ padding: 20px 28px; border-bottom: 1px solid #ddd; background: white; }}
    h1 {{ margin: 0; font-size: 22px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 22px; display: grid; gap: 18px; }}
    section {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 16px; }}
    h2 {{ margin: 0 0 8px; font-size: 17px; }}
    p {{ margin: 0 0 12px; color: #555; }}
    .tags {{ color: #666; font-size: 12px; margin-bottom: 10px; }}
    img {{ max-width: 100%; border: 1px solid #eee; }}
    a {{ color: #1d5d6a; text-decoration: none; margin-right: 12px; }}
    a:hover {{ text-decoration: underline; }}
    pre {{ white-space: pre-wrap; background: #fafafa; border: 1px solid #eee; padding: 10px; max-height: 360px; overflow: auto; }}
  </style>
</head>
<body>
<header>
  <h1>Routed OPD Artifact Dashboard</h1>
  <p>Local browser for paper figures, tables, CSVs, and reports.</p>
</header>
<main id="app"></main>
<script>
window.__PCD_ARTIFACTS_MANIFEST__ = {manifest_text};
async function loadManifestWithFallback() {{
  try {{
    const response = await fetch('artifacts_manifest.json');
    if (response.ok) return await response.json();
  }} catch (err) {{}}
  return window.__PCD_ARTIFACTS_MANIFEST__;
}}
function renderOutput(output) {{
  const path = output.path;
  if (path.endsWith('.png')) return `<img src="${{path}}" alt="${{path}}" />`;
  if (path.endsWith('.md') || path.endsWith('.csv')) return `<a href="${{path}}">${{path}}</a>`;
  return `<a href="${{path}}">${{path}}</a>`;
}}
loadManifestWithFallback().then((manifest) => {{
  const app = document.getElementById('app');
  app.innerHTML = manifest.artifacts.map((artifact) => `
    <section>
      <h2>${{artifact.title}}</h2>
      <p>${{artifact.description}}</p>
      <div class="tags">${{(artifact.tags || []).join(' · ')}}</div>
      <div>${{artifact.outputs.map(renderOutput).join('')}}</div>
    </section>
  `).join('');
}});
</script>
</body>
</html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
