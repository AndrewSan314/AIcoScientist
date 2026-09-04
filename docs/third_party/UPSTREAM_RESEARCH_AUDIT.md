# Upstream research audit

This branch keeps third-party research as reference material or lazy optional boundaries. No upstream source files are vendored into the runtime package.

| Upstream | Pinned revision | License review | Runtime reuse | Integration decision |
|---|---|---|---|---|
| [hypoAL](https://github.com/ziatdinovmax/hypoAL) | `731142998c6b729f613ec22de86484fa88d15d8e` | MIT | None | Reference for hypothesis-driven active learning and HIG framing |
| [gpax](https://github.com/ziatdinovmax/gpax) | `b493ba26c0613e6ade694e5bb6847bfbf4511f36` | MIT | None | Optional lazy backend boundary; no hidden dependency |
| [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer) | `bf32082521e45c0fcf5cf9ae9bd1321e76bf9012` | MIT | None | Optional lazy adapter; CPU fallback is explicit |
| [atomai](https://github.com/pycroscopy/atomai) | `93c88817a577686d6b8a84ab954872ca5cab7fcc` | MIT | None | Optional microscopy boundary; CPU fallback is explicit |
| [CAMEO_NComm](https://github.com/KusneNIST/CAMEO_NComm) | `f4f1ef44a4f9dcb8fd8adba2edbcd91fd10feab` | Mixed/unclear; Graph Cut GPLv2 | None | Reference-only; no source import or vendoring |
| [s4](https://github.com/CederGroupHub/s4) | `58691ce37bc0f659dac29804805c0538f284850b` | MIT | None | Transparent feasibility prior only |
| [SynthesisSimilarity](https://github.com/CederGroupHub/SynthesisSimilarity) | `21f013ff6a1fe1f5eeb8c48e32bf20d601d2fb86` | No explicit license found | None | Reference-only |
| [novel-materials-screening](https://github.com/mattmcdermott/novel-materials-screening) | `a8d7fd435c4bb11648cc286f9946667b561c007c` | No explicit license found | None | Reference-only |

The complete machine-readable manifest is [upstream_research_audit.json](upstream_research_audit.json). The A-Lab data source is treated separately as local replay data; archive-level inventory never implies candidate-level SEM/EDS linkage.
