# Prior-art component comparison

| System | Main contribution | Shared multimodal latent hypotheses | Candidate×modality action costs | Raw-artifact provenance | This branch's boundary |
|---|---|---:|---:|---:|---|
| hypoAL | Hypothesis-driven active learning | Partial/reference | Not the focus | Not the focus | HIG and hypothesis framing only |
| GPax | Gaussian-process tooling | Backend capability | Not the focus | Not the focus | Lazy optional backend |
| CAMEO | Segmentation/communication workflow | No shared scientific contract | No | External workflow | Reference-only baseline |
| A-Lab | Real synthesis/characterization campaign | Domain data | Dataset-specific | Archive/ledger linkage | Offline replay with canonical XRD/refinement |
| AutoXRD | XRD analysis | No shared latent contract | No | Input-dependent | Reference-only unless checkpoint/references are configured; CPU descriptors are separate |
| AtomAI | Microscopy ML | No shared latent contract | No | Input-dependent | Reference-only; no trained A-Lab weights or AtomAI invocation |
| S4 | Synthesis representation/planning | Not this contract | Not this contract | Not this contract | Reference-only; no S4 integration claim |
| AIcoScientist | Falsification-first material discovery | Existing and multimodal contract | Yes | Yes | Shared engine plus explicit evidence limits |

This comparison is descriptive and does not claim that the systems solve identical tasks.
