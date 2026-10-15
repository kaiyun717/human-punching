#!/usr/bin/env python3
"""Build the evidence report and an offline edition; requires Markdown 3.11.

PYTHONPATH=.cache/report-tools .venv/bin/python -B scripts/build_master_report.py
Only report artifacts under human-punching are written. Simulation assets are read-only.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import markdown
from markdown.extensions.toc import TocExtension, slugify

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
OUTPUT = ROOT / 'outputs/master-report'
MEDIA: dict[str, dict] = {}
NAV = [
    ('overview', 'Overview'), ('scope', 'Scope & dependencies'),
    ('source-motion', 'DuoBox interpretation'), ('appearance', 'Mesh & glove revision'),
    ('gpu-rendering', 'GPU server & rendering'), ('g1-targeting', 'Standing G1 targeting'),
    ('contact', 'Contact contract'), ('warp-runtime', 'Warp & camera probes'),
    ('classical-physics', 'Classical physical control'), ('timestep', 'Timestep failures'),
    ('sequences', 'Continuous recovery'), ('training-body', 'Constrained training body'),
    ('scalar-reference', 'Scalar IK iterations'), ('learning', 'PPO training'),
    ('paired-result', 'First matched evaluation'), ('five-ms', '5 ms & four punch policies'),
    ('limits', 'Limits & next phases'),
    ('experiments', 'Experiment ledger'), ('reproduce', 'Reproduction & archive'),
]
SOURCE_DOCS = sorted(DOCS.glob('[0-2][0-9]-*.md'))
SOURCE_IDS = {path.name: f'source-{path.stem}' for path in SOURCE_DOCS}


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def href(path: Path, start: Path = DOCS) -> str:
    return quote(os.path.relpath(path, start), safe='/')


def media_path(value: str) -> str:
    path = ROOT / value
    if not path.is_file():
        raise FileNotFoundError(f'Media does not exist: {value}')
    if value not in MEDIA:
        MEDIA[value] = {
            'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'mime': mimetypes.guess_type(path.name)[0],
        }
    return href(path)


def render_media(match: re.Match) -> str:
    fields = match.group(1).split('|')
    kind = fields[0]
    if kind == 'video':
        _, source, poster, title, caption, badge = fields
        content = (f'<video controls playsinline preload="none" poster="{media_path(poster)}" '
                   f'aria-label="{escape(title)}">'
                   f'<source src="{media_path(source)}" type="video/mp4">'
                   'Your browser does not support embedded video. '
                   f'<a href="{href(ROOT/source)}">Open the MP4</a>.</video>')
    else:
        _, source, caption, badge = fields
        title = badge
        content = (f'<img src="{media_path(source)}" alt="{escape(caption)}" '
                   'loading="lazy" decoding="async">')
    return (f'<figure class="media"><div class="media-head">'
            f'<span class="media-title">{escape(title)}</span>'
            f'<span class="badge">{escape(badge)}</span></div>{content}'
            f'<figcaption>{escape(caption)}<span class="asset-path">'
            f'{escape(source)}</span></figcaption></figure>')


def ledger() -> tuple[str, list[dict]]:
    rows = []
    for name in ('master-report-experiments.json', 'master-report-learning-experiments.json',
                 'master-report-five-ms-experiments.json'):
        rows.extend(json.loads((DOCS / name).read_text()))
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate experiment identifiers')
    result = ['<div class="ledger-tools"><label>Search experiments'
              '<input id="experiment-search" type="search" placeholder="Try: 5 ms, glove, Warp, PPO…">'
              '</label><label>Outcome<select id="outcome-filter">'
              '<option value="all">All outcomes</option><option value="passed">Passed</option>'
              '<option value="failed">Failed</option><option value="partial">Partial</option>'
              '<option value="diagnostic">Diagnostic</option></select></label></div>'
              '<p id="ledger-count" role="status"></p>'
              '<p id="no-experiments" class="no-results" hidden>No records match this search.</p>']
    for row in rows:
        evidence = row['evidence']
        if isinstance(evidence, str):
            evidence = [evidence]
        links = []
        for value in evidence:
            path = ROOT / value
            if not path.exists():
                raise FileNotFoundError(f'Ledger evidence does not exist: {value}')
            links.append(f'<a href="{href(path)}">{escape(value)}</a>')
        result.append(
            f'<div class="experiment" data-outcome="{escape(row["outcome"])}">'
            f'<div class="experiment-header"><span class="experiment-id">{escape(row["id"])}</span>'
            f'<span class="outcome {escape(row["outcome"])}">{escape(row["outcome"])}</span>'
            f'<span class="experiment-stage">{escape(row["stage"])}</span></div>'
            f'<p class="method">{escape(row["method"])}</p>'
            f'<h4>{escape(row["question"])}</h4><p>{escape(row["finding"])}</p>'
            f'<div class="evidence-links">{"".join(links)}</div></div>')
    return '\n'.join(result), rows


def source_archive() -> str:
    result = ['<h3 id="source-archive">The complete historical report archive</h3>'
              '<p>These are the original reports, reproduced in full. Their dates and claims belong '
              'to their respective stage; the main narrative above reconciles the current state.</p>'
              '<div class="source-tools"><button id="expand-sources" type="button">Expand all reports</button>'
              '<button id="collapse-sources" type="button" class="secondary">Collapse all</button></div>']
    for path in SOURCE_DOCS:
        raw = path.read_text()
        title = raw.splitlines()[0].lstrip('# ').strip()
        prefix = SOURCE_IDS[path.name]
        converter = markdown.Markdown(extensions=[
            'tables', 'fenced_code', 'sane_lists',
            TocExtension(baselevel=3, slugify=lambda text, separator, p=prefix:
                         p + '-' + slugify(text, separator)),
        ])
        body = converter.convert(raw)
        body = re.sub(r'href="#([^"]+)"',
                      lambda match, p=prefix: f'href="#{p}-{match.group(1)}"', body)
        result.append(
            f'<details class="source-document" id="{prefix}"><summary>'
            f'{escape(path.name[:2])} · {escape(title)}</summary><div class="source-body">'
            '<p class="source-notice"><strong>Historical snapshot.</strong> '
            'Read this in its original stage. Later reports supersede earlier inventory and milestone claims. '
            f'<a href="{quote(path.name)}" data-raw-source="true">Original Markdown file</a>.</p>'
            f'{body}</div></details>')
    return '\n'.join(result)


def internal_source_links(text: str) -> str:
    def replace(match: re.Match) -> str:
        prefix, target = match.groups()
        parts = urlsplit(html.unescape(target))
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        path = (DOCS / unquote(parts.path)).resolve()
        if path.parent == DOCS and path.name in SOURCE_IDS:
            anchor = SOURCE_IDS[path.name]
            if parts.fragment:
                anchor += '-' + parts.fragment
            return f'{prefix}#{anchor}"'
        return match.group(0)
    # Preserve explicit links that offer the raw Markdown file.
    return re.sub(r'(href=")([^"]+)"(?! data-raw-source)', replace, text)


def wrap_tables(text: str) -> str:
    return text.replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')


def make_page(body: str, edition: str, experiment_count: int) -> str:
    nav = ''.join(f'<a href="#{ident}"><span>{index:02d}</span>{escape(label)}</a>'
                  for index, (ident, label) in enumerate(NAV))
    css = (DOCS / 'master-report.css').read_text()
    js = (DOCS / 'master-report.js').read_text()
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Complete Human Punching project progress: motion retargeting, failures, physical control, SOMA body and learned jab, with evidence videos and original reports.">
<title>Human Punching — Master Progress Report</title><style>{css}</style></head><body>
<a class="skip-link" href="#overview">Skip to report</a>
<aside class="sidebar"><div class="brand">human-punching<span>MASTER PROGRESS REPORT</span></div>
<div class="nav-label">The progression</div><nav aria-label="Report sections">{nav}</nav>
<div class="sidebar-footer">Evidence snapshot<br><strong>15 October 2026</strong><br>{experiment_count} experiment records · {len(SOURCE_DOCS)} reports<br>
<a href="#five-ms">Latest measured result →</a><br><a href="#limits">Remaining limits →</a></div></aside>
<header class="topbar"><div class="topbar-title">Human motion → physical control → learning</div>
<div class="topbar-actions"><small>{escape(edition)}</small><button id="print-report" type="button">Print / Save PDF</button></div>
<div id="progress" aria-hidden="true"></div></header>
<main><details class="mobile-nav"><summary>Contents · {len(NAV)} sections</summary><nav aria-label="Mobile report sections">{nav}</nav></details>
<article>{body}</article><footer>Human Punching · Research and implementation record · Evidence, not a formal safety certificate.<br>
Generated from project-owned reports and retained artifacts. No external runtime assets.</footer></main>
<script>{js}</script></body></html>'''


def standalone(text: str) -> str:
    def replace(match: re.Match) -> str:
        attr, encoded = match.groups()
        value = html.unescape(encoded)
        parts = urlsplit(value)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        path = (DOCS / unquote(parts.path)).resolve()
        if attr in ('src', 'poster') and path.is_file():
            mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            data = base64.b64encode(path.read_bytes()).decode('ascii')
            return f'{attr}="data:{mime};base64,{data}"'
        rebased = href(path, OUTPUT)
        if parts.fragment:
            rebased += '#' + parts.fragment
        return f'{attr}="{rebased}"'
    return re.sub(r'(src|poster|href)="([^"]+)"', replace, text)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    content = (DOCS / 'master-report-content.md').read_text()
    content = re.sub(r'\{\{((?:image|video)\|[^\n]+)\}\}', render_media, content)
    rendered_ledger, rows = ledger()
    # Insert trusted HTML after Markdown conversion so source Markdown is parsed only once.
    content = content.replace('{{ledger}}', '<div id="ledger-slot"></div>')
    content = content.replace('{{sources}}', '<div id="sources-slot"></div>')
    body = markdown.markdown(content, extensions=['tables', 'fenced_code', 'md_in_html', 'sane_lists'])
    body = body.replace('<div id="ledger-slot"></div>', rendered_ledger)
    body = body.replace('<div id="sources-slot"></div>', source_archive())
    if '{{' in body:
        raise ValueError('Unexpanded report content marker')
    body = wrap_tables(internal_source_links(body))
    # Include media introduced by the full report appendices in the provenance
    # manifest, not only media selected by narrative template markers.
    for target in re.findall(r'(?:src|poster)="([^"]+)"', body):
        parts = urlsplit(html.unescape(target))
        if not parts.scheme and not parts.netloc and parts.path:
            path = (DOCS / unquote(parts.path)).resolve()
            media_path(str(path.relative_to(ROOT)))
    page = make_page(body, 'Workspace edition', len(rows))
    canonical = DOCS / 'master-report.html'
    canonical.write_text(page)
    offline = OUTPUT / 'master-report-standalone.html'
    offline.write_text(standalone(make_page(body, 'Offline media edition', len(rows))))
    manifest = {
        'schema': 'human_punching_master_report_v1',
        'evidence_snapshot': '2026-10-15',
        'canonical_html': str(canonical.relative_to(ROOT)),
        'standalone_html': str(offline.relative_to(ROOT)),
        'experiment_count': len(rows), 'historical_report_count': len(SOURCE_DOCS),
        'video_player_count': len(re.findall(r'<video\b', page)),
        'image_count': len(re.findall(r'<img\b', page)),
        'media': MEDIA,
        'source_documents': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in SOURCE_DOCS},
        'output_sizes_bytes': {'canonical': canonical.stat().st_size, 'standalone': offline.stat().st_size},
        'output_sha256': {'canonical': hashlib.sha256(canonical.read_bytes()).hexdigest(),
                          'standalone': hashlib.sha256(offline.read_bytes()).hexdigest()},
        'standalone_scope': 'Narrative, historical reports and selected image/video media embedded; raw source/data links require original workspace',
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ('canonical_html', 'standalone_html',
                     'experiment_count', 'historical_report_count', 'output_sizes_bytes')}, indent=2))


if __name__ == '__main__':
    main()
