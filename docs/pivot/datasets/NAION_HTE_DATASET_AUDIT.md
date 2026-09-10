# Na-ion high-throughput process-chain dataset audit

- Source: [Zenodo DOI 10.5281/zenodo.7981011](https://zenodo.org/records/7981011), downloaded 2026-09-11.
- Archive: `Na-upscaling.zip` (660,036,021 bytes); publisher MD5 `955939922331e756e756ef81a6b68fb6`; local SHA-256 `881bf2f9f9fbf59cd2d05b3c996f94940c0e0abe89d9eae6106ea0bee4e7a4f5`.
- Parsed source scope: 72 CSV traces with formation/cycling time-series fields. The archive also contains 65 HDF5 copies, which are preserved raw but not parsed without adding a duplicate-format dependency.
- Verified CSV fields: data point, timestamp/test time, cycle/step indices, current, voltage, power, charge/discharge capacity and energy, plus resistance-derived columns.
- Lineage: run identity comes from each source filename; the top-level source directory is the protocol/batch group. The source notes document C/20, 3-step, pulsed, and C-rate procedures.
- Limitation: the parsed archive does not provide a source-linked per-cell synthesis, coating, or assembly recipe table. Those stages are not fabricated in the normalized contract.
- Leakage boundary: first 256 valid points are a `FORMATION_CURVE`; last 256 points and final observed discharge capacity are attached only to `FINAL_CHARACTERIZATION`.
