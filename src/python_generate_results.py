import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.cluster import HDBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from sklearn.cluster import DBSCAN

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"
RESULT_DIR = ROOT_DIR / "result"
PYTHON_RESULT_DIR = RESULT_DIR / "python"

PYTHON_RESULT_DIR.mkdir(parents=True, exist_ok=True)


DATASET_CONFIGS = {
    "circles": {
        "kind": "preprocessed",
        "path": DATA_DIR / "circles_dataset.csv",
    },
    "moons": {
        "kind": "preprocessed",
        "path": DATA_DIR / "moons_dataset.csv",
    },
    "varied": {
        "kind": "preprocessed",
        "path": DATA_DIR / "varied_dataset.csv",
    },
    "aniso": {
        "kind": "preprocessed",
        "path": DATA_DIR / "aniso_dataset.csv",
    },
    "blobs": {
        "kind": "preprocessed",
        "path": DATA_DIR / "blobs_dataset.csv",
    },
    "no_structure": {
        "kind": "preprocessed",
        "path": DATA_DIR / "no_structure_dataset.csv",
    },
    "heartfailure": {
    "kind": "unprocessed",
    "path": DATA_DIR / "heartfailure.csv",
    "feature_cols": [
        "Age (years)",
        "Male (1=Yes, 0=No)",
        "PHQ-9",
        "Systolic BP (mm Hg)",
        "Estimated glomerular filtration rate",
        "Ejection fraction (%)",
        "Serum sodium (mmol/l)",
        "Blood urea nitrogen (mg/dl)",
        "Etiology HF(1=Yes, 0=No)",
        "Prior diabetes mellitus",
        "Elevated level of BNP/NT-BNP (1=Yes, 0=No)",
    ],
},

"cardiacarrest": {
    "kind": "unprocessed",
    "path": DATA_DIR / "cardiacarrest.csv",
    "feature_cols": [
        "sex_woman",
        "Age_years",
        "Endotracheal_intubation",
        "Functional_status",
        "Asystole",
        "Bystander",
        "Time_min",
        "Cardiogenic",
        "Cardiac_arrest_at_home",
    ],
},

"neuroblastoma": {
    "kind": "unprocessed",
    "path": DATA_DIR / "neuroblastoma.csv",
    "feature_cols": [
        "age",
        "sex",
        "site",
        "stage",
        "time_months",
        "autologous_stem_cell_transplantation",
        "radiation",
        "degree_of_differentiation",
        "UH_or_FH",
        "MYCN_status ",
        "surgical_methods",
    ],
},

"sepsis": {
    "kind": "unprocessed",
    "path": DATA_DIR / "sepsis.csv",
    "feature_cols": [
        "Age",
        "sex_woman",
        "diagnosis_0EC_1M_2_AC",
        "APACHE II",
        "SOFA",
        "CRP",
        "WBCC",
        "NeuC",
        "LymC",
        "EOC",
        "NLCR",
        "PLTC",
        "MPV",
        "LOS-ICU",
    ],
},

"type1diabetes": {
    "kind": "unprocessed",
    "path": DATA_DIR / "type1diabetes.csv",
    "feature_cols": [
        "age",
        "duration.of.diabetes",
        "body_mass_index",
        "TDD",
        "basal",
        "bolus",
        "HbA1c",
        "eGFR",
        "perc.body.fat",
        "adiponectin",
        "free.testosterone",
        "SMI",
        "grip.strength",
        "knee.extension.strength",
        "gait.speed",
        "ucOC",
        "OC",
        "weight_kg",
        "sex_0man_1woman",
    ],
},
}


DATASETS = [
    "circles",
    "moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
    "heartfailure",
    "cardiacarrest",
    "neuroblastoma",
    "sepsis",
    "type1diabetes",]

def vector_to_string(x):
    return " ".join(map(str, x))


def load_dataset(dataset_name):

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    cfg = DATASET_CONFIGS[dataset_name]
    path = cfg["path"]

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {path}")

    if cfg["kind"] == "preprocessed":
        return pd.read_csv(path, header=None).to_numpy(dtype=float)

    elif cfg["kind"] == "unprocessed":
        df = pd.read_csv(path)

        # Keep only the chosen feature columns
        X_df = df[cfg["feature_cols"]].copy()

        # Force numeric; non-numeric junk becomes NaN
        X_df = X_df.apply(pd.to_numeric, errors="coerce")

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_df)

        scaler = RobustScaler()
        X = scaler.fit_transform(X)

        return X

    else:
        raise ValueError(f"Unsupported dataset kind: {cfg['kind']}")



def run_one(dataset_name, algo_name, params):
    print(f"Running Python {algo_name} on {dataset_name}")

    X = load_dataset(dataset_name)

    if algo_name == "hdbscan":
        model = HDBSCAN(**params)
    elif algo_name == "dbscan":
        model = DBSCAN(**params)
    else:
        raise ValueError(f"Unsupported algoritm: {algo_name}")

    t0 = time.perf_counter()
    model.fit(X)
    t1 = time.perf_counter()

    labels = model.labels_.astype(int)
    probabilities = model.probabilities_.astype(float)

    out = pd.DataFrame([{
        "dataset": dataset_name,
        "fit_time_sec": t1 - t0,
        "labels": vector_to_string(labels),
        "probabilities": vector_to_string(probabilities),
    }])

    out_path = PYTHON_RESULT_DIR / f"{dataset_name}_{algo_name}_python.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved {out_path}")

# MAIN

def main(inp_min_cluster_size, inp_min_samples):

    params = dict(
    min_cluster_size=inp_min_cluster_size,
    min_samples=inp_min_samples,
    allow_single_cluster=False,
    copy=True,)

    algo_name = "hdbscan"
    for dataset_name in DATASETS:
        run_one(dataset_name, algo_name, params)

if __name__ == "__main__":

    inp_min_cluster_size = int(sys.argv[1])
    inp_min_samples = int(sys.argv[2])

    main(inp_min_cluster_size, inp_min_samples)