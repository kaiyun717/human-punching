(() => {
  'use strict';
  const links = [...document.querySelectorAll('.sidebar nav a')];
  const sections = [...document.querySelectorAll('article > section')];
  const progress = document.getElementById('progress');
  const updatePosition = () => {
    const extent = document.documentElement.scrollHeight - innerHeight;
    progress.style.width = `${extent > 0 ? Math.min(100, scrollY / extent * 100) : 0}%`;
    let current = sections[0]?.id;
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= 130) current = section.id;
    }
    for (const link of links) {
      const active = link.hash === `#${current}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  };
  let scheduled = false;
  addEventListener('scroll', () => {
    if (!scheduled) requestAnimationFrame(() => { updatePosition(); scheduled = false; });
    scheduled = true;
  }, {passive: true});
  updatePosition();
  document.getElementById('print-report').addEventListener('click', () => print());

  const search = document.getElementById('experiment-search');
  const outcome = document.getElementById('outcome-filter');
  const experiments = [...document.querySelectorAll('.experiment')];
  const filter = () => {
    const words = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    let count = 0;
    for (const row of experiments) {
      const matches = (outcome.value === 'all' || row.dataset.outcome === outcome.value) &&
        words.every(word => row.textContent.toLowerCase().includes(word));
      row.hidden = !matches;
      count += Number(matches);
    }
    document.getElementById('ledger-count').textContent = `${count} of ${experiments.length} experiment records shown`;
    document.getElementById('no-experiments').hidden = count !== 0;
    updatePosition();
  };
  search.addEventListener('input', filter);
  outcome.addEventListener('change', filter);
  filter();

  const comparison = [...document.querySelectorAll('#physical-comparison video')];
  const status = document.getElementById('compare-status');
  document.getElementById('compare-play').addEventListener('click', async () => {
    comparison.forEach(video => { video.currentTime = 0; video.playbackRate = 1; video.muted = true; });
    const results = await Promise.allSettled(comparison.map(video => video.play()));
    status.textContent = results.every(result => result.status === 'fulfilled')
      ? 'Both physical rollouts started together. Each clip includes ordinary and slowed playback.'
      : 'A browser blocked playback. Use the individual video controls to start the comparison.';
  });
  document.getElementById('compare-pause').addEventListener('click', () => {
    comparison.forEach(video => video.pause());
    status.textContent = 'Comparison paused.';
  });
  document.getElementById('expand-sources').addEventListener('click', () => {
    document.querySelectorAll('.source-document').forEach(item => { item.open = true; });
    updatePosition();
  });
  document.getElementById('collapse-sources').addEventListener('click', () => {
    document.querySelectorAll('.source-document').forEach(item => { item.open = false; });
    updatePosition();
  });
  const revealAnchor = () => {
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (!target) return;
    let parent = target.parentElement;
    while (parent) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
      parent = parent.parentElement;
    }
    if (target instanceof HTMLDetailsElement) target.open = true;
    requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
  };
  addEventListener('hashchange', revealAnchor);
  if (location.hash) revealAnchor();
  document.querySelectorAll('.mobile-nav nav a').forEach(link => {
    link.addEventListener('click', () => { document.querySelector('.mobile-nav').open = false; });
  });
})();
