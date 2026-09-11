# Dataset research

The registry intentionally leaves run/sample counts, file inventory, field names, and hashes as unverified until raw files are acquired. Adapters fail closed until a normalized, source-audited manifest exists.

| ID | Verified source | Evidence | License/status | Intended role |
| --- | --- | --- | --- | --- |
| `drakopoulos_graphite` | Mendeley Data DOI `10.17632/4dh2h3tsf4.1`, version 1 | physical historical | CC BY 4.0 | Multi-factor real manufacturing / recipe replay |
| `warwick_nmc622` | Mendeley Data DOI `10.17632/wwhm2frfmy.1`, version 1 | pilot-line historical | CC0 1.0 | Process and physical/electrochemical multimodal benchmark |
| `warwick_ultrasound` | Mendeley Data DOI `10.17632/c62yn37d9h.1`, version 1 | physical historical | CC BY 4.0 | Signal + process thickness prediction |
| `naion_hte` | Zenodo DOI `10.5281/zenodo.7981011` | physical historical | License must be read from downloaded record/manifest | Heterogeneous process-chain lineage |
| `artistic` | Zenodo DOI `10.5281/zenodo.5956128` | simulated physics | CC BY-NC-SA 4.0; files restricted | Explicitly simulated stage/scale stress |

Acquisition procedure: download under `data/external/<dataset>/<version>/raw/`, retain immutable files, hash every raw file, record source URL/DOI/version/license/timestamp, map source columns, then create `DATASET_AUDIT.md`, `SOURCE_TO_SCHEMA.md`, and the processed-cache manifest. No raw source data were downloaded by this change.
