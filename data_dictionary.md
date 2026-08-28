# Data Dictionary

## Raw Process Data

- `sample_id`: Unique electrode sample ID.
- `si_content`: Nano-silicon content in wt%.
- `mxene_content`: Few-layer Ti3C2Tx MXene content in wt%.
- `alginate_content`: Sodium alginate binder content in wt%.
- `carbon_content`: Conductive carbon content in wt%.
- `mixing_time`: Slurry mixing time in minutes.
- `drying_temp`: Electrode drying temperature in Celsius.
- `pressing_pressure`: Electrode pressing pressure in arbitrary lab units.

## SEM Features

- `particle_size_mean`: Mean particle size estimated from SEM-derived features.
- `porosity_score`: Relative porosity score from 0 to 1.
- `agglomeration_index`: Relative aggregation score from 0 to 1.
- `crack_density`: Relative crack density from 0 to 1.
- `surface_uniformity`: Relative surface uniformity from 0 to 1.

## EDX Data

- `si_percent`: Silicon atomic/weight percentage proxy.
- `ti_percent`: Titanium percentage proxy from MXene.
- `c_percent`: Carbon percentage proxy.
- `o_percent`: Oxygen percentage proxy.
- `impurity_percent`: Trace impurity percentage proxy.
- `impurity_score`: `impurity_percent / 100`, retained as a normalized analysis field.

## Electrochemical Data

- `initial_capacity`: First-cycle capacity.
- `capacity_50`: Capacity after 50 cycles.
- `capacity_100`: Capacity after 100 cycles.
- `retention_100`: Capacity retention after 100 cycles, target for the MVP model.
- `coulombic_efficiency`: Coulombic efficiency percentage.
- `rct`: Charge-transfer resistance proxy.

## Engineered Features

- `si_mxene_ratio`: `si_content / mxene_content`.
- `si_ti_ratio`: `si_percent / ti_percent`.
- `c_o_ratio`: `c_percent / o_percent`.
- `capacity_fade`: `initial_capacity - capacity_100`; stored for analysis, not used for recommendations.
