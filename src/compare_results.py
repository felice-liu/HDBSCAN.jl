from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hdbscan.validity import validity_index
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"
RESULT_DIR = ROOT_DIR / "result"

PYTHON_RESULT_DIR = RESULT_DIR / "python"
JULIA_RESULT_DIR = RESULT_DIR / "julia"
COMPARISON_RESULT_DIR = RESULT_DIR / "comparison"

COMPARISON_RESULT_DIR.mkdir(parents=True, exist_ok=True)


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
        "label_cols": {
            "death": "Death (1=Yes, 0=No)",
            "hospitalized": "Hospitalized (1=Yes, 0=No)",
        },
        "extra_cols": {
            "death_time": "Time from HF to Death (days)",
            "hospitalization_time": "Time from HF to hospitalization (days)",
        },
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
        "label_cols": {
            "exitus": "Exitus",
        },
    },

    "neuroblastoma": {
        "kind": "unprocessed",
        "path": DATA_DIR / "neuroblastoma.csv",
        "feature_cols": [
            "age",
            "sex",
            "site",
            "stage",
            "risk",
            "time_months",
            "autologous_stem_cell_transplantation",
            "radiation",
            "degree_of_differentiation",
            "UH_or_FH",
            "MYCN_status ",
            "surgical_methods",
        ],
        "label_cols": {
            "outcome": "outcome",
        },
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
            "Group",
            "LOS-ICU",
        ],
        "label_cols": {
            "mortality": "Mortality",
        },
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
        "label_cols": {
            "insulin_regimen": "insulin_regimen_binary",
        },
    },
}


DATASETS_TO_COMPARE = [
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


BASE_COLORS = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",]


def parse_vector_string(s, dtype=float):
    if pd.isna(s) or str(s).strip() == "":
        return np.array([], dtype=dtype)
    return np.array([dtype(x) for x in str(s).split()])


def load_dataset_bundle(dataset_name):

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    cfg = DATASET_CONFIGS[dataset_name]
    path = cfg["path"]

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    if cfg["kind"] == "preprocessed":
        X = pd.read_csv(path, header=None).to_numpy(dtype=float)
        return X, {}, {}

    elif cfg["kind"] == "unprocessed":
        df = pd.read_csv(path)

        X_df = df[cfg["feature_cols"]].copy()
        X_df = X_df.apply(pd.to_numeric, errors="coerce")

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_df)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        labels_dict = {
            label_name: df[col_name].to_numpy(dtype=int)
            for label_name, col_name in cfg.get("label_cols", {}).items()
        }

        extras_dict = {
            extra_name: df[col_name].to_numpy()
            for extra_name, col_name in cfg.get("extra_cols", {}).items()
        }

        return X, labels_dict, extras_dict

    else:
        raise ValueError(f"Unsupported dataset kind: {cfg['kind']}")


def load_result(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")

    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"{path} should contain exactly one row")

    row = df.iloc[0]
    return {
        "fit_time_sec": float(row["fit_time_sec"]),
        "labels": parse_vector_string(row["labels"], dtype=int),
        "probabilities": parse_vector_string(row["probabilities"], dtype=float),
    }


def safe_dbcv(X, labels):
    labels = np.asarray(labels, dtype=int)

    if len(set(labels) - {-1}) == 0:
        return np.nan

    try:
        return float(validity_index(X, labels))
    except Exception:
        return np.nan


def safe_allclose(a, b, atol=1e-8, rtol=1e-8):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=atol, rtol=rtol)


def labels_to_colors(labels):
    labels = np.asarray(labels, dtype=int)

    non_noise = sorted(l for l in np.unique(labels) if l >= 0)
    color_map = {
        lab: BASE_COLORS[i % len(BASE_COLORS)]
        for i, lab in enumerate(non_noise)
    }

    out = []
    for lab in labels:
        if lab == -1:
            out.append("#000000")
        else:
            out.append(color_map.get(lab, "#000000"))
    return np.array(out)


def project_for_plot(X):

    X = np.asarray(X, dtype=float)
    if X.shape[1] == 2:
        return X
    return PCA(n_components=2, random_state=42).fit_transform(X)


def plot_panel(ax, X_plot, labels, title, fit_time, dbcv):
    colors = labels_to_colors(labels)
    ax.scatter(X_plot[:, 0], X_plot[:, 1], s=10, color=colors)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title, fontsize=12)

    dbcv_text = "nan" if np.isnan(dbcv) else f"{dbcv:.32f}"
    ax.text(
        0.99,
        0.01,
        f"{fit_time:.6f}s\nDBCV={dbcv_text}",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
    )


def compute_external_metrics(labels_dict, cluster_labels):

    out = {}

    if not labels_dict:
        return out

    for label_name, y_true in labels_dict.items():
        try:
            out[f"{label_name}_ari"] = adjusted_rand_score(y_true, cluster_labels)
        except Exception:
            out[f"{label_name}_ari"] = np.nan

        try:
            out[f"{label_name}_nmi"] = normalized_mutual_info_score(y_true, cluster_labels)
        except Exception:
            out[f"{label_name}_nmi"] = np.nan

    return out


def compare_datasets(dataset_names, run_name, julia_algo_name, python_algo_name):
    summary_rows = []

    fig, axes = plt.subplots(len(dataset_names), 2, figsize=(10, 4 * len(dataset_names)))
    if len(dataset_names) == 1:
        axes = np.array([axes])

    for i, dataset_name in enumerate(dataset_names):
        print(f"Comparing {dataset_name}...")

        X, labels_dict, extras_dict = load_dataset_bundle(dataset_name)
        X_plot = project_for_plot(X)

        py = load_result(PYTHON_RESULT_DIR / f"{dataset_name}_{python_algo_name}_python.csv")
        jl = load_result(JULIA_RESULT_DIR / f"{dataset_name}_{julia_algo_name}_julia.csv")

        py_dbcv = safe_dbcv(X, py["labels"])
        jl_dbcv = safe_dbcv(X, jl["labels"])

        row = {
            "dataset": dataset_name,
            "python_fit_time_sec": py["fit_time_sec"],
            "julia_fit_time_sec": jl["fit_time_sec"],
            "python_dbcv": py_dbcv,
            "julia_dbcv": jl_dbcv,
            "same_labels": np.array_equal(py["labels"], jl["labels"]),
            "same_probabilities": safe_allclose(py["probabilities"], jl["probabilities"]),
            "python_n_clusters": len(set(py["labels"]) - {-1}),
            "julia_n_clusters": len(set(jl["labels"]) - {-1}),
            "python_n_noise": int(np.sum(py["labels"] == -1)),
            "julia_n_noise": int(np.sum(jl["labels"] == -1)),
        }

        py_metrics = compute_external_metrics(labels_dict, py["labels"])
        jl_metrics = compute_external_metrics(labels_dict, jl["labels"])

        for k, v in py_metrics.items():
            row[f"python_{k}"] = v
        for k, v in jl_metrics.items():
            row[f"julia_{k}"] = v

        summary_rows.append(row)

        plot_panel(
            axes[i, 0],
            X_plot,
            py["labels"],
            f"{dataset_name} - Python",
            py["fit_time_sec"],
            py_dbcv,
        )

        plot_panel(
            axes[i, 1],
            X_plot,
            jl["labels"],
            f"{dataset_name} - Julia",
            jl["fit_time_sec"],
            jl_dbcv,
        )

    plt.tight_layout()

    summary_df = pd.DataFrame(summary_rows)

    out_dir = COMPARISON_RESULT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{run_name}_comparison_summary_julia_{julia_algo_name}_vs_python_{python_algo_name}.csv"
    plot_path = out_dir / f"{run_name}_clustering_visualized_julia_{julia_algo_name}_vs_python_{python_algo_name}.png"

    summary_df.to_csv(summary_path, index=False, float_format="%.32f")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved plot to:    {plot_path}")
    print()
    print(summary_df)


def main():
    compare_datasets(DATASETS_TO_COMPARE, run_name="all_datasets",
    julia_algo_name="hdbscan", python_algo_name="hdbscan")


if __name__ == "__main__":
    main()