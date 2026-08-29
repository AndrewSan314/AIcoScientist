from .base import DatasetSpec


PROCESS_FEATURES = [
    "si_content",
    "mxene_content",
    "alginate_content",
    "carbon_content",
    "mixing_time",
    "drying_temp",
    "pressing_pressure",
]
SEM_FEATURES = [
    "particle_size_mean",
    "porosity_score",
    "agglomeration_index",
    "crack_density",
    "surface_uniformity",
]
EDX_FEATURES = [
    "si_percent",
    "ti_percent",
    "c_percent",
    "o_percent",
    "impurity_percent",
]
ENGINEERED_FEATURES = ["si_mxene_ratio", "si_ti_ratio", "c_o_ratio"]
MODEL_FEATURES = PROCESS_FEATURES + SEM_FEATURES + EDX_FEATURES + ENGINEERED_FEATURES
TARGET = "retention_100"
CANDIDATE_COLUMNS = [
    "si_content",
    "mxene_content",
    "alginate_content",
    "drying_temp",
    "mixing_time",
]
SEARCH_SPACE = {
    "si_content": range(40, 81, 5),
    "mxene_content": range(5, 36, 5),
    "alginate_content": range(5, 19, 1),
    "drying_temp": range(60, 121, 10),
    "mixing_time": range(30, 61, 15),
}
UCB_BETA = 1.0
CHEMISTRY_ALPHA = 0.5

SI_MXENE_SPEC = DatasetSpec(
    name="si_mxene",
    id_column="sample_id",
    feature_columns=MODEL_FEATURES,
    target_column=TARGET,
    objective="maximize",
    candidate_columns=CANDIDATE_COLUMNS,
    pre_experiment_features=PROCESS_FEATURES + ENGINEERED_FEATURES,
    post_experiment_characterization=SEM_FEATURES + EDX_FEATURES,
    targets=[TARGET],
    constraints=["si_plus_mxene_sum", "alginate_bounds"],
    candidate_variables=CANDIDATE_COLUMNS,
    optional_columns=["capacity_fade", "impurity_score"],
    entity_id_column="sample_id",
    source_dataset="si_mxene",
    source_version="1.0",
)
