# HTML master progress report

Evidence snapshot: 15 October 2026.

- [Workspace edition](master-report.html): approximately 630 KB; opens directly in a browser and loads selected media from the existing project output folders.
- [Standalone edition](../outputs/master-report/master-report-standalone.html): approximately 61 MB; embeds all selected images and videos, the main narrative, and the complete text of reports 00–25. Copy this one file to view those contents offline. Raw JSON, checkpoint, source-code and original-file links still require the project workspace.

The report contains 63 searchable experiment records, 26 historical report appendices, 24 video players and 16 images. Videos use controls and do not autoplay. The matched physical comparison can start/pause both rollouts together. Navigation, outcome filtering, source-report expansion and a print view are included. Printing preserves static content; use HTML for video evidence. Expand historical reports before printing if their full text is wanted.

## Content and provenance

The main source is [master-report-content.md](master-report-content.md), with [CSS](master-report.css) and [JavaScript](master-report.js). The experiment records live in [the early/classical ledger](master-report-experiments.json), [the scalar/learning ledger](master-report-learning-experiments.json), and [the 5 ms/four-family ledger](master-report-five-ms-experiments.json). The builder renders the original reports 00–25 into expandable historical appendices and turns their cross-references into in-page links. Original reports and simulation artifacts are not modified.

All selected media paths are checked before building. [The generated manifest](../outputs/master-report/manifest.json) records media SHA256 hashes, source-document hashes, output hashes and sizes. Figures retain source paths and captions distinguishing kinematic replay, native physics, GPU physics and learned control. In particular, historical G1 near-misses are not presented as successful analytical contact, and classical old-body results are not attributed to the latest scalar-body policy.

## Rebuild

From the `human-punching` root, use the isolated report-tool directory. This avoids changing either simulation environment:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install \
  --target .cache/report-tools 'Markdown==3.11' 'playwright==1.63.0'

PYTHONPATH=.cache/report-tools .venv/bin/python -B scripts/build_master_report.py
```

The builder writes `docs/master-report.html` and the standalone edition/manifest under `outputs/master-report/`. Rebuilding embeds the current historical Markdown files; it does not recompute physics or training results.

## Validate

Static validation checks local paths, all internal anchors, unique IDs and media accessibility metadata:

```bash
.venv/bin/python -B scripts/validate_master_report.py
```

For actual Chromium playback and desktop/mobile checks, keep browser caches and temporary profiles inside the project:

```bash
mkdir -p .cache/report-tmp .cache/report-xdg

TMPDIR="$PWD/.cache/report-tmp" XDG_CACHE_HOME="$PWD/.cache/report-xdg" \
PYTHONPATH=.cache/report-tools \
PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/report-browsers" \
  .venv/bin/python -B -m playwright install chromium --only-shell

TMPDIR="$PWD/.cache/report-tmp" XDG_CACHE_HOME="$PWD/.cache/report-xdg" \
PYTHONPATH=.cache/report-tools \
PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/report-browsers" \
  .venv/bin/python -B scripts/validate_master_report.py --browser
```

Validation writes [validation.json](../outputs/master-report/validation.json) and desktop/mobile screenshots under `outputs/master-report/`. The browser checks selected-image decoding, metadata for every video in both editions, actual paired playback, search/outcome filters, empty results, source expansion/deep links, mobile navigation, overflow and JavaScript errors. The standalone check makes no external HTTP(S) requests. This validates the report presentation; the underlying simulation validation remains documented in reports 13, 19 and 20–25, with the latest fresh-trial evidence in report 23.

If a browser disallows workspace video access through `file://`, serve the project locally with `.venv/bin/python -m http.server 8000 --bind 127.0.0.1` from the project root, then visit `http://127.0.0.1:8000/docs/master-report.html`. The standalone edition embeds its media and normally needs only a browser.
