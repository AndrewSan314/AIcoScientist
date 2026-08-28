def add_edx_features(df):
    required = {"si_percent", "ti_percent", "c_percent", "o_percent", "impurity_percent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing EDX columns: {sorted(missing)}")
    if (df[["ti_percent", "o_percent"]] <= 0).any().any():
        raise ValueError("EDX ratio denominators must be positive")

    result = df.copy()
    result["si_ti_ratio"] = result["si_percent"] / result["ti_percent"]
    result["c_o_ratio"] = result["c_percent"] / result["o_percent"]
    result["impurity_score"] = result["impurity_percent"] / 100
    return result
