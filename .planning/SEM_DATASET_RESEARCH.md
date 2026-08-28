# SEM Dataset Research

Checked: 2026-07-09

| Work | Public images? | Practical status |
|---|---|---|
| SAM-I-Am (Abebe et al.) | No verified public validation set | The paper describes a manually labeled TEM validation set and data-sharing restrictions. Its cited GitHub URL currently returns 404. |
| micro-sam / Segment Anything for Microscopy | Yes, via cited public microscopy datasets | Code, models, and EM/LM benchmark datasets are public, but these are mainly biological microscopy data rather than battery-electrode SEM. |
| Furat et al. 2022, SEM super-resolution | No direct download | The paper states that data are available from the corresponding authors on reasonable request. |
| Badmos et al. 2020, electrode defect CNN | No verified archive | The paper is indexed in the Aalen repository, but no separately downloadable image dataset was found. |
| Deeg et al. 2024, microstructure to rate capability | No direct download | The paper's data availability statement says data are available from the corresponding author on request. |
| Oh et al. 2024, NCM composition/state CNN | Partial | The official repository includes training code and 51 `NCM622_cycled` SEM images, but no trained checkpoint or complete 12-class dataset. It is a valid image source, not a runnable published classifier. |

## Downloaded Demo

`python -m src.fetch_sem_demo` downloads the repository archive and extracts
three NCM622 cycled SEM images:

- `data/raw/sem_images/NCM622_CYCLED_453.jpg`
- `data/raw/sem_images/NCM622_CYCLED_454.jpg`
- `data/raw/sem_images/NCM622_CYCLED_455.jpg`

The three-image subset is for exercising ingestion and feature extraction only.
It is not large enough for model training or scientific performance claims.

See `SEM_PAPER_METHODS.md` for the audited methods, repository contents, and
the recommended path to a real model-backed web flow.

## Sources

- Oh paper: https://doi.org/10.1038/s41524-024-01279-6
- Oh dataset/code: https://github.com/MIIMSEKAIST/CNN_for_NCM-composition-and-state-prediction
- Furat paper: https://doi.org/10.1038/s41524-022-00749-z
- Deeg paper: https://doi.org/10.3390/batteries10030099
- Badmos paper: https://doi.org/10.1007/s10845-019-01484-x
- SAM-I-Am paper: https://doi.org/10.1016/j.commatsci.2024.113400
- micro-sam paper: https://doi.org/10.1038/s41592-024-02580-4
- micro-sam code: https://github.com/computational-cell-analytics/micro-sam

For a larger public battery-electrode defect dataset, consider the 2025
CoatingVision dataset: https://doi.org/10.6084/m9.figshare.29260121.v1
