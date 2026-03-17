---
hide:
  - navigation
  - toc
---

# Option Pricing Library

<div class="portfolio-hero">
  <p class="hero-kicker">Typed pricing, surface repair, and validated numerics</p>
  <p class="hero-copy">This library prices vanilla options, repairs noisy implied-vol surfaces, smooths the Dupire handoff with eSSVI, and validates the local-vol/PDE workflow with repricing and convergence evidence.</p>
  <div class="cta-row cta-row--trio" markdown="1">
[Read the proof path](user_guides/decision_guide.md){ .md-button .md-button--primary }
[Start with the recommended API](user_guides/instruments.md){ .md-button }
[Review performance evidence](performance.md){ .md-button }
  </div>
</div>

<figure markdown class="diagram" style="--diagram-max-width: 1180px">
  ![Reviewer proof panel showing surface repair, eSSVI smoothing, local-vol extraction, and PDE repricing with tracked evidence callouts](assets/generated/showcase/reviewer_proof_panel.light.svg){ .diagram-img .diagram-light }
  ![Reviewer proof panel showing surface repair, eSSVI smoothing, local-vol extraction, and PDE repricing with tracked evidence callouts](assets/generated/showcase/reviewer_proof_panel.dark.svg){ .diagram-img .diagram-dark }
  <figcaption>The fastest single scan: quoted-to-repaired surface evidence, smooth Dupire handoff evidence, and local-vol/PDE validation evidence in one panel.</figcaption>
</figure>

## Validation snapshot

<div class="snapshot-grid">
  <div class="snapshot-card">
    <p class="snapshot-label">Seam control</p>
    <p class="snapshot-copy">Worst observed <code>w_T</code> seam jump</p>
    <p class="metric-pair">
      <span class="metric-number">8.07e-2</span>
      <span class="metric-arrow">&rarr;</span>
      <span class="metric-number metric-number--good">8.17e-5</span>
    </p>
  </div>
  <div class="snapshot-card">
    <p class="snapshot-label">Dupire handoff</p>
    <p class="snapshot-copy"><code>projection_dupire_invalid_count</code></p>
    <p class="metric-line">
      <span class="metric-number metric-number--good">0</span>
    </p>
  </div>
  <div class="snapshot-card">
    <p class="snapshot-label">PDE repricing</p>
    <p class="snapshot-copy">Published repricing sweep</p>
    <p class="metric-line">
      <span class="metric-number">154</span>
      <span class="metric-caption">options repriced</span>
    </p>
    <p class="metric-inline">
      <span class="metric-key">Mean abs price error</span>
      <span class="metric-value">8.1e-4</span>
    </p>
    <p class="metric-inline">
      <span class="metric-key">Max abs IV error</span>
      <span class="metric-value">18.7 bp</span>
    </p>
  </div>
</div>

## First clicks

| If you want to review... | Open this |
| --- | --- |
| The strongest end-to-end proof path | [Decision guide](user_guides/decision_guide.md) |
| The recommended public API | [Instruments](user_guides/instruments.md) |
| The measured runtime and scaling story | [Performance evidence](performance.md) |
| The system design and module boundaries | [Architecture](architecture.md) |
| The generated API reference | [API reference](api/index.md) |

## What makes this repo credible quickly

- The public API is typed and layered rather than notebook-only.
- The surface workflow shows quoted data, repair diagnostics, and fit stress instead of a single polished chart.
- The eSSVI handoff page shows why the time-derivative problem matters before local-vol extraction begins.
- The local-vol/PDE page publishes repricing and convergence evidence, and the benchmark page publishes runtime/error tradeoffs from committed artifacts.

## Continue by intent

- [Installation](installation.md) for a local checkout or editable setup
- [Guides](user_guides/index.md) for the API and workflow map
- [Performance evidence](performance.md) for committed benchmark plots and summaries
- [Architecture](architecture.md) for package structure and dependency direction
- [Future work](roadmap.md) for what is still exploratory
