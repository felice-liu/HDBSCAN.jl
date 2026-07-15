# This file can be run though command line. It can accept hyperparameters in the command 
# line and generate files in hdbscan\result\python by fitting said hypermarameters
# for the 11 datasets in hdbscan\data (as of now can't be modified).

import time
from pathlib import Path
import argparse
from timeit import repeat

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import HDBSCAN

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULT_DIR = ROOT_DIR / "result"
PYTHON_RESULT_DIR = RESULT_DIR / "python"

PYTHON_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset name -> has header?
DATASETS = {
    "circles": False,
    "moons": False,
    "varied": False,
    "aniso": False,
    "blobs": False,
    "no_structure": False,

    "heartfailure": True,
    "cardiacarrest": True,
    "neuroblastoma": True,
    "sepsis": True,
    "type1diabetes": True,
}


def vector_to_string(v):
    return " ".join(map(str, v))


def fill_missing_median(df):
    imputer = SimpleImputer(strategy="median")
    df[df.columns] = imputer.fit_transform(df)
    return df


def load_dataset(dataset_name, has_header):

    path = DATA_DIR / f"{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {path}")

    df = pd.read_csv(path, header=0 if has_header else None)
    

    df = fill_missing_median(df)

    return df.to_numpy(dtype=float)


def generate_results(model,
                     dataset_name,
                     has_header):

    print(f"Running Python HDBSCAN on {dataset_name}")

    X = load_dataset(dataset_name, has_header)

    times = repeat(lambda: model.fit(X), number=1, repeat=10)
    avg = sum(times) / len(times)
    
    model.fit(X)
    labels = model.labels_
    probabilities = model.probabilities_

    out = pd.DataFrame([{
        "dataset": dataset_name,
        "average_fit_time_sec": avg,
        "labels": vector_to_string(labels),
        "probabilities": vector_to_string(probabilities),
    }])

    out_path = PYTHON_RESULT_DIR / f"{dataset_name}_results.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved {out_path}")


def generate_all_examples(model):

    for dataset_name, has_header in DATASETS.items():
        generate_results(
            model,
            dataset_name,
            has_header
        )

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate HDBSCAN benchmark results."
    )

    parser.add_argument("--min_cluster_size", type=int, required=True)
    parser.add_argument("--min_samples", type=int, required=True)

    parser.add_argument(
        "--cluster_selection_epsilon",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max_cluster_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--leaf_size",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--cluster_selection_method",
        type=str,
        choices=["eom", "leaf"],
        default="eom",
    )

    parser.add_argument(
        "--allow_single_cluster",
        action="store_true",
    )

    parser.add_argument(
        "--copy",
        action="store_true",
    )

    return parser.parse_args()

def main():

    # Example of usage

    args = parse_args()

    model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        max_cluster_size=args.max_cluster_size,
        metric=args.metric,
        metric_params=None,
        alpha=args.alpha,
        algorithm=args.algorithm,
        leaf_size=args.leaf_size,
        cluster_selection_method=args.cluster_selection_method,
        allow_single_cluster=args.allow_single_cluster,
        copy=args.copy,
    )

    generate_all_examples(model)


if __name__ == "__main__":
    main()