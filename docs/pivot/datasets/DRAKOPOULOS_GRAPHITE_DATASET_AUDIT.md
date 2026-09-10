# Graphite Process-15 dataset audit

- DOI/version/license: `10.17632/4dh2h3tsf4.1`, version 1, CC BY 4.0.
- Raw inventory and SHA-256: [`manifest.json`](../../../data/external/drakopoulos_graphite/1/manifest.json).
- Parsed source: `AS-Cell_Data-Azar-Stavros corrected FCL-25-01-2021.xlsx`, sheet `FInal_All_Cell_Data`.
- Verified parsed run count: 13 `AS-*` rows with numeric coating speed, coating gap, and the source field `Cell Capacity (372 mAh/g)`.
- Stages: formulation → coating → drying. The source worksheet does not expose a verified calendering pressure; the adapter does not fabricate one.
- Controls: drying temperature, coating speed/gap, active-material/conductive-additive/binder/additive fractions where present.
- Target: `cell_capacity_mah`, mapped from the source's `Cell Capacity (372 mAh/g)` field; source label is retained in serialized provenance.
- Grouping: deterministic hash of the observed recipe controls; no row from the same control tuple may cross the main grouped split.
- Missingness: sparse fraction/temperature fields remain absent, never imputed as observed.
