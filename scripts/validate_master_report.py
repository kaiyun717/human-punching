#!/usr/bin/env python3
"""Check report paths/anchors and optionally exercise the real browser UI.

PYTHONPATH=.cache/report-tools PLAYWRIGHT_BROWSERS_PATH=.cache/report-browsers \
  .venv/bin/python -B scripts/validate_master_report.py --browser
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/master-report'


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.references = []
        self.videos = 0
        self.images = 0
        self.missing_alt = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for key in ('href', 'src', 'poster'):
            if key in attrs:
                self.references.append((key, attrs[key]))
        self.videos += tag == 'video'
        self.images += tag == 'img'
        if tag == 'img' and not attrs.get('alt'):
            self.missing_alt += 1


def static_check(path):
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    parser = ReportParser()
    parser.feed(path.read_text())
    missing_paths, missing_anchors = [], []
    external_media = []
    for attr, target in parser.references:
        parts = urlsplit(target)
        if parts.scheme or parts.netloc:
            if attr in ('src', 'poster') and parts.scheme != 'data':
                external_media.append(target)
            continue
        if parts.path and not (path.parent / unquote(parts.path)).exists():
            missing_paths.append(target)
        if not parts.path and parts.fragment and unquote(parts.fragment) not in parser.ids:
            missing_anchors.append(target)
    result = {
        'file': str(path.relative_to(ROOT)),
        'references_checked': len(parser.references),
        'videos': parser.videos, 'images': parser.images,
        'duplicate_ids': [key for key, count in Counter(parser.ids).items() if count > 1],
        'missing_paths': sorted(set(missing_paths)),
        'missing_anchors': sorted(set(missing_anchors)),
        'missing_image_alt': parser.missing_alt,
        'external_media': external_media,
    }
    for key in ('duplicate_ids', 'missing_paths', 'missing_anchors', 'missing_image_alt', 'external_media'):
        assert not result[key], (path.name, key, result[key])
    assert parser.videos == manifest['video_player_count']
    assert parser.images == manifest['image_count']
    return result


MEDIA_PROBE = """async () => {
  const results = await Promise.all([...document.querySelectorAll('video')].map((video, index) =>
    new Promise(resolve => {
      let timer;
      const finish = (error) => {
        clearTimeout(timer);
        video.removeEventListener('loadedmetadata', loaded);
        video.removeEventListener('error', failed);
        resolve({index, title: video.getAttribute('aria-label'), duration: video.duration,
                 width: video.videoWidth, height: video.videoHeight, error});
      };
      const loaded = () => finish(null);
      const failed = () => finish(video.error?.message || 'media error');
      timer = setTimeout(() => finish('metadata timeout'), 20000);
      video.addEventListener('loadedmetadata', loaded);
      video.addEventListener('error', failed);
      video.preload = 'metadata';
      if (video.readyState >= 1) loaded(); else video.load();
    })));
  return results;
}"""


def browser_check():
    from playwright.sync_api import sync_playwright

    errors = []
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    experiments, reports = manifest['experiment_count'], manifest['historical_report_count']
    with sync_playwright() as playwright:
        # Software painting avoids video-overlay screenshot artifacts on this login host.
        browser = playwright.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1050}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto((ROOT / 'docs/master-report.html').as_uri(), wait_until='load')
        page.add_style_tag(content='html{scroll-behavior:auto!important}')
        assert page.locator('.experiment').count() == experiments
        assert page.locator('.source-document').count() == reports
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path=str(OUTPUT / 'desktop-overview.png'))

        # Validate image decoding even for below-the-fold lazy images.
        images = page.evaluate('''async () => Promise.all([...document.images].map(async image => {
          image.loading = 'eager';
          try { await image.decode(); return {ok: image.naturalWidth > 0, alt: image.alt}; }
          catch (error) { return {ok: false, alt: image.alt, error: String(error)}; }
        }))''')
        assert all(image['ok'] for image in images), images
        media = page.evaluate(MEDIA_PROBE)
        assert all(not item['error'] and item['width'] > 0 and item['duration'] > 0 for item in media), media

        page.locator('#experiment-search').fill('5 ms')
        page.locator('#outcome-filter').select_option('failed')
        failed_timestep_matches = page.locator('.experiment:not([hidden])').count()
        assert failed_timestep_matches > 0
        page.locator('#experiment-search').fill('no-such-experiment-xyz')
        assert page.locator('#no-experiments').is_visible()
        page.locator('#experiment-search').fill('')
        page.locator('#outcome-filter').select_option('all')
        assert page.locator('.experiment:not([hidden])').count() == experiments

        page.locator('#expand-sources').click()
        assert page.locator('.source-document[open]').count() == reports
        page.locator('#collapse-sources').click()
        assert page.locator('.source-document[open]').count() == 0
        page.evaluate("location.hash = '#source-04-glove-rotation-audit'")
        page.wait_for_function("document.getElementById('source-04-glove-rotation-audit').open")
        page.locator('#source-04-glove-rotation-audit').evaluate('(item) => item.open = false')

        page.locator('#compare-play').click()
        page.wait_for_function("[...document.querySelectorAll('#physical-comparison video')].every(v => !v.paused && v.currentTime > 2 && v.readyState >= 2)")
        decoded_frames = page.evaluate('''async () => await Promise.all(
          [...document.querySelectorAll('#physical-comparison video')].map(video => new Promise(resolve => {
            video.requestVideoFrameCallback((now, frame) => resolve({
              time: video.currentTime, frames: frame.presentedFrames, width: frame.width, height: frame.height
            }));
          })))''')
        assert all(frame['frames'] > 0 and frame['width'] > 0 for frame in decoded_frames)
        page.locator('#compare-pause').click()
        assert page.locator('#physical-comparison video').evaluate_all('(items) => items.every(v => v.paused)')
        page.locator('#paired-result').scroll_into_view_if_needed()
        page.evaluate("document.getElementById('paired-result').scrollIntoView({block:'start', behavior:'instant'})")
        page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
        page.screenshot(path=str(OUTPUT / 'desktop-comparison.png'))
        page.get_by_role('heading', name='Four selected policies pass fresh physical testing', exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(OUTPUT / 'desktop-five-ms-results.png'))

        page.set_viewport_size({'width': 390, 'height': 844})
        page.evaluate("window.scrollTo({top:0, behavior:'instant'})")
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path=str(OUTPUT / 'mobile-overview.png'))
        page.locator('.mobile-nav > summary').click()
        page.locator('.mobile-nav a[href="#paired-result"]').click()
        assert not page.locator('.mobile-nav').evaluate('(item) => item.open')
        page.evaluate("document.getElementById('paired-result').scrollIntoView({block:'start', behavior:'instant'})")
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path=str(OUTPUT / 'mobile-comparison.png'))
        page.get_by_role('heading', name='Four selected policies pass fresh physical testing', exact=True).scroll_into_view_if_needed()
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path=str(OUTPUT / 'mobile-five-ms-results.png'))

        offline = browser.new_page(viewport={'width': 1440, 'height': 1050})
        offline.on('pageerror', lambda error: errors.append(str(error)))
        requests = []
        offline.on('request', lambda request: requests.append(request.url))
        offline.goto((OUTPUT / 'master-report-standalone.html').as_uri(), wait_until='load')
        offline.add_style_tag(content='html{scroll-behavior:auto!important}')
        offline_media = offline.evaluate(MEDIA_PROBE)
        assert all(not item['error'] and item['width'] > 0 for item in offline_media), offline_media
        assert offline.locator('.source-document').count() == reports
        assert not [url for url in requests if url.startswith(('http:', 'https:'))]
        offline.locator('#compare-play').click()
        offline.wait_for_function("[...document.querySelectorAll('#physical-comparison video')].every(v => !v.paused && v.currentTime > .1)")
        offline.locator('#compare-pause').click()
        assert not errors, errors
        version = browser.version
        browser.close()
    return {
        'browser': version, 'painting': 'software; headless video-overlay artifacts avoided',
        'javascript_errors': errors,
        'desktop_and_mobile_overflow': False,
        'images_decoded': len(images), 'workspace_video_metadata': media,
        'offline_video_metadata': offline_media,
        'failed_timestep_filter_matches': failed_timestep_matches,
        'ledger_search_outcome_filter_no_results': 'passed',
        'expand_collapse_and_deep_source_link': 'passed',
        'paired_video_play_pause_both_editions': 'passed',
        'decoded_comparison_frames': decoded_frames,
        'mobile_navigation': 'passed', 'offline_network_requests': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', action='store_true')
    args = parser.parse_args()
    result = {'static': [static_check(ROOT / 'docs/master-report.html'),
                         static_check(OUTPUT / 'master-report-standalone.html')]}
    if args.browser:
        result['browser'] = browser_check()
    (OUTPUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'static_editions_passed': len(result['static']),
                      'browser_checks': 'passed' if args.browser else 'not requested',
                      'validation': str((OUTPUT / 'validation.json').relative_to(ROOT))}, indent=2))


if __name__ == '__main__':
    main()
