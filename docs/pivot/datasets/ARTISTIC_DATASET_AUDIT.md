# ARTISTIC physics-stress dataset audit

- Source: [Zenodo DOI 10.5281/zenodo.5956128](https://zenodo.org/records/5956128), checked 2026-09-11; CC BY-NC-SA 4.0.
- Record scope: source code for the ARTISTIC online calculator workflow, run sequentially as slurry, drying, then calendering. The record describes inputs in `user_inputs*` files and 3D electrode/manufacturing outputs.
- Access: the public record exposes metadata but restricts all files behind authenticated access/request access. No archive, schema, sample count, or hash was inferable without credentials.
- Policy: `ArtisticSimulationAdapter` remains fail-closed. It must never synthesize records, label simulated output as physical, or claim an executable benchmark until source files are supplied through authorized access.
