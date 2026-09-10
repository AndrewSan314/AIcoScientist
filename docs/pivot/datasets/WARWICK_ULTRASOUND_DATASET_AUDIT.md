# Warwick frequency-domain ultrasound V4 audit

- Source: [Mendeley Data, DOI 10.17632/c62yn37d9h.4](https://data.mendeley.com/datasets/c62yn37d9h/4), downloaded 2026-09-11 under CC BY 4.0.
- Archive: `frequency_domain_ultrasound_v4.zip`; SHA-256 `c8a8dbdcc0d7bbe2580ec1d3a0842d13706f794c6c13ee33b9eb1ade9056825d`.
- Scope: 48 paired before/after calendering FFT records: 30 graphite-anode and 18 NMC622-cathode samples.
- Schema checked: each JSON contains `metadata`, `fft_frequency`, and `fft_magnitude`. Metadata carries the process state, source sample identifier, control settings, thickness, and density where supplied.
- Leakage boundary: before- and after-calendering spectra are retained as separate observations. Both are only available after calendering; an information horizon at `CALENDERING` excludes them.
