# Task 09 LaTeX section and Overleaf figures

15 October 2026. The new section is in [main_mjlab.tex](../../main_mjlab.tex:4123), immediately after Task 08 and before the appendix. Its insertion is delimited by `BEGIN TASK09 HUMAN-PUNCHING IMPLEMENTATION RECORD` and `END TASK09 HUMAN-PUNCHING IMPLEMENTATION RECORD`. Removing that delimited insertion reproduces the prior file byte for byte. Existing sections, preamble, bibliography, and appendix are unchanged.

The section covers six completed implementation phases and five proposed next phases, with two tables and three figures (one figure contains four panels). It distinguishes historical kinematic/classical pilots from the final learned release, and fixed-reference tracking accuracy from unimplemented target/TTC commands and adversarial decision-making.

## Upload these six PNGs

These byte-identical evidence copies live in `docs/figures/task09/`, outside ignored `outputs/`, so they can be tracked with the documentation. Their [manifest](figures/task09/manifest.json) records original artifact paths, sizes and SHA256 hashes. No source images or frozen-release files were changed.

Paths in the LaTeX are relative to the directory containing `main_mjlab.tex`, currently the `HumanBox` root. Preserve this directory layout in Overleaf, or change both occurrences of a path—the `\IfFileExists` check and `\includegraphics` command—to the uploaded location. No absolute server paths are required. Placeholders allow compilation before the images are uploaded.

| Figure/panel | Relative path from the LaTeX file | Evidence and scope |
|---|---|---|
| `fig:task09-native-pd` | `human-punching/docs/figures/task09/5ms-controller-comparison.png` | Historical explicit/native PD diagnostic at genuine 5 ms; predates the final learned controllers. |
| `fig:task09-retiming` | `human-punching/docs/figures/task09/hook-retiming-demand.png` | Original and retimed hook effort demand under the stated inverse-dynamics/support approximation. |
| `fig:task09-four-punches`, jab | `human-punching/docs/figures/task09/jab-contact.png` | Final learned jab nominal rollout near closest approach. |
| `fig:task09-four-punches`, cross | `human-punching/docs/figures/task09/cross-contact.png` | Final learned cross nominal rollout near closest approach. |
| `fig:task09-four-punches`, left hook | `human-punching/docs/figures/task09/left-hook-contact.png` | Final learned left-hook nominal rollout near closest approach. |
| `fig:task09-four-punches`, right hook | `human-punching/docs/figures/task09/right-hook-contact.png` | Final learned right-hook nominal rollout near closest approach. |

The four contact stills come from the immutable release, with checkpoint/render provenance recorded in its [manifest](../outputs/four-punch-milestone/manifest.json). They depict recorded physical human states beside a stationary G1 fixture. They do not show a running G1 policy or glove/G1 impact response. The full-cycle numeric evidence is in the results table; a still alone does not prove recovery or tracking accuracy.

The retiming diagnostic also has a [vector SVG](../outputs/reference-dynamics-5931826/retiming-demand.svg), but the LaTeX uses the existing PNG so no new conversion tool or package is needed. Existing `graphicx`, `booktabs`, `array`, `amsmath`, and `hyperref` support all new content. No bibliography entries or additional packages were introduced.

## Result provenance

The final table is derived from `outputs/four-punch-milestone/manifest.json`, including the full fresh 64-trial ranges for every family. Its learned/baseline counts are 64/64 versus 0/64 for each family. The four nominal anchors are shared with validation; the remaining 63 perturbations per family were generated after checkpoint selection. The references, targets, morphology and root poses are not held-out variations.

The original file is preserved for the insertion-only check at `.cache/task09-docs/main_mjlab.before.tex`; this temporary local verification copy is not a required migration asset. The root handoff records any full-document LaTeX build validation separately.


## Validation and review PDF

The Task 09 fragment compiled successfully with the installed `pdflatex` in a two-column `article` harness. Two passes resolved cross-references; there were no overfull-box or unresolved-reference warnings. The six evidence images were included, and the result/table/figure pages were visually inspected.

- [Task 09-only review PDF](../outputs/task09-documentation/task09-preview.pdf) — five pages; approximate review layout, not the full master document's IEEE class.
- [Validation record](../outputs/task09-documentation/validation.json) — original-content preservation, source hashes, labels, figure paths and local-link checks.

The full master document could not be compiled with its exact configuration here because `ieeeconf.cls` and `main.bib` are absent. No class, bibliography, preamble or previous section was replaced to bypass that. Overleaf should use the original complete project and the updated `main_mjlab.tex`; final page/float placement can differ from the review PDF.
