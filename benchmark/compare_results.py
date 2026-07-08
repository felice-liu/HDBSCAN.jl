from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dbcv
from sklearn.decomposition import PCA

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT_DIR = BENCHMARK_DIR.parent

SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
RESULT_DIR = ROOT_DIR / "result"

PYTHON_RESULT_DIR = RESULT_DIR / "python"
JULIA_RESULT_DIR = RESULT_DIR / "julia"
COMPARISON_RESULT_DIR = RESULT_DIR / "comparison"

COMPARISON_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# dataset name : has header?
DATASETS = {"circles": False,
    "moons": False,
    "varied": False,
    "aniso": False,
    "blobs": False,
    "no_structure": False,

    "heartfailure": True,
    "cardiacarrest": True,
    "neuroblastoma": True,
    "sepsis": True,
    "type1diabetes": True,}

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


def load_dataset(dataset_name, has_header):

    path = DATA_DIR / f"{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    
    df = pd.read_csv(path, header=0 if has_header else None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median())

    return df.to_numpy(dtype=float)


def load_result(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")

    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"{path} should contain exactly one row")

    row = df.iloc[0]
    return {
        "average_fit_time_sec": float(row["average_fit_time_sec"]),
        "labels": parse_vector_string(row["labels"], dtype=int),
        "probabilities": parse_vector_string(row["probabilities"], dtype=float),
    }


def safe_dbcv(X, labels):
    labels = np.asarray(labels, dtype=int)

    if len(set(labels) - {-1}) == 0:
        return np.nan

    try:
        return float(dbcv.dbcv(X, labels))
    except Exception:
        return np.nan


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
    ax.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=colors,
        s=18,
        edgecolors="black",
        linewidths=0.2,)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax.tick_params(labelsize=8)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title, fontsize=12)

    dbcv_text = "nan" if np.isnan(dbcv) else f"{dbcv:.15f}"
    ax.text(
        0.99,
        0.01,
        f"{fit_time:.6f}s\nDBCV={dbcv_text}",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
    )


def compare_datasets(dataset_names, run_name):
    summary_rows = []

    fig, axes = plt.subplots(len(dataset_names), 2, figsize=(10, 4 * len(dataset_names)))

    if len(dataset_names) == 1:
        axes = np.array([axes])

    for i, dataset_name in enumerate(dataset_names):
        print(f"Comparing {dataset_name}...")

        X = load_dataset(dataset_name, DATASETS[dataset_name])
        X_plot = project_for_plot(X)

        py = load_result(
            PYTHON_RESULT_DIR / f"{dataset_name}_results.csv"
        )

        jl = load_result(
            JULIA_RESULT_DIR / f"{dataset_name}_results.csv"
        )

        py_dbcv = safe_dbcv(X, py["labels"])
        jl_dbcv = safe_dbcv(X, jl["labels"])

        row = {
            "dataset": dataset_name,

            "python_fit_time_sec": py["average_fit_time_sec"],
            "julia_fit_time_sec": jl["average_fit_time_sec"],

            "python_dbcv": py_dbcv,
            "julia_dbcv": jl_dbcv,

            "same_labels": np.array_equal(py["labels"], jl["labels"]),
            "same_probabilities":
                py["probabilities"].shape == jl["probabilities"].shape
                and np.allclose(
                    py["probabilities"],
                    jl["probabilities"],
                    atol=1e-8,
                    rtol=1e-8,
                ),

            "python_clusters": len(set(py["labels"]) - {-1}),
            "julia_clusters": len(set(jl["labels"]) - {-1}),

            "python_noise": int(np.sum(py["labels"] == -1)),
            "julia_noise": int(np.sum(jl["labels"] == -1)),}

        summary_rows.append(row)

        plot_panel(
            axes[i, 0],
            X_plot,
            py["labels"],
            f"{dataset_name} - Python",
            py["average_fit_time_sec"],
            py_dbcv,
        )

        plot_panel(
            axes[i, 1],
            X_plot,
            jl["labels"],
            f"{dataset_name} - Julia",
            jl["average_fit_time_sec"],
            jl_dbcv,
        )

    fig.tight_layout()

    summary_df = pd.DataFrame(summary_rows)

    out_dir = COMPARISON_RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{run_name}_comparison_summary_julia_vs_python.csv"
    plot_path = out_dir / f"{run_name}_clustering_visualized_julia_vs_python.png"

    summary_df.to_csv(summary_path, index=False, float_format="%.32f")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved plot to:    {plot_path}")
    print()
    print(summary_df)


def main():
    compare_datasets(DATASETS, run_name="all_datasets")


if __name__ == "__main__":
    main()