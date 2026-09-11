# Warwick NMC622 pilot dataset audit

- DOI/version/license: `10.17632/wwhm2frfmy.1`, version 1, CC0 1.0.
- Raw inventory and SHA-256: [`manifest.json`](../../../data/external/warwick_nmc622/1/manifest.json).
- Parsed source: extracted `Intermediate measurements during calendering.xlsx`, `Cathode` sheet.
- Verified parsed run count: 18 electrode DOE conditions; this adapter does not claim to have joined these rows to the separate half-cell files.
- Stage coverage: coating observations → calendering controls/observations.
- Controls: target coating weight, roll temperature, target density/porosity, roll gap, number of passes.
- Targets: `calendered_density_g_cm3`, `calendered_porosity_pct`; tensile strength/thickness remain stage observations.
- Grouping: deterministic exact DOE-control tuple; grouping avoids condition leakage.
- Raw package also contains Megtec/Mesys and Biologic files. They remain source-linked inventory for future typed time-series/curve adapters, not silently concatenated into this electrode table.
