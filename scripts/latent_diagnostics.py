"""Stage 3 Gate: Latent Diagnostics and Predictability Analysis.

Reports per Section 6.5 & 9.2 of the design document:
1. Per-lambda channel statistics: mean, standard deviation, and sparsity (fraction of zeros).
2. Evaluation of how accurately lambda can be predicted from pooled latent summary statistics.
Writes output to reports/latent_diagnostics.md.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.data.manifest import ParquetManifest
from nrcomp.data.store import LatentStore


def compute_vector_features(arr) -> List[float]:
    """Compute summary statistics for a 3D/4D numpy array without heavy ML dependencies."""
    import numpy as np
    flat = arr.astype(np.float32).ravel()
    n = len(flat)
    if n == 0:
        return [0.0] * 7

    abs_flat = np.abs(flat)
    mean_abs = float(np.mean(abs_flat))
    std = float(np.std(flat))
    zero_frac = float(np.mean(flat == 0))
    l2_norm = float(np.sqrt(np.mean(flat ** 2)))
    p10 = float(np.percentile(abs_flat, 10))
    p50 = float(np.percentile(abs_flat, 50))
    p90 = float(np.percentile(abs_flat, 90))

    return [mean_abs, std, zero_frac, l2_norm, p10, p50, p90]


def nearest_centroid_classifier(
    X_train: List[List[float]], y_train: List[int], X_test: List[List[float]], y_test: List[int]
) -> float:
    """Simple, pure-Python nearest-centroid classifier to assess lambda predictability."""
    # Compute class centroids
    class_vectors = defaultdict(list)
    for feat, label in zip(X_train, y_train):
        class_vectors[label].append(feat)

    centroids: Dict[int, List[float]] = {}
    for label, vecs in class_vectors.items():
        dim = len(vecs[0])
        c = [sum(vec[d] for vec in vecs) / len(vecs) for d in range(dim)]
        centroids[label] = c

    # Evaluate on test set
    correct = 0
    for feat, label in zip(X_test, y_test):
        best_dist = float("inf")
        best_cls = -1
        for cls_lbl, centroid in centroids.items():
            dist = math.sqrt(sum((f - c) ** 2 for f, c in zip(feat, centroid)))
            if dist < best_dist:
                best_dist = dist
                best_cls = cls_lbl
        if best_cls == label:
            correct += 1

    return correct / len(y_test) if len(y_test) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Generate Latent Diagnostics Report.")
    parser.add_argument("--manifest", type=str, default="data/manifest_pilot.parquet", help="Path to manifest.")
    parser.add_argument("--output", type=str, default="reports/latent_diagnostics.md", help="Output markdown path.")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples to analyze.")
    args = parser.parse_args()

    manifest_path = (PROJECT_ROOT / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    output_path = (PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)

    if not manifest_path.exists():
        if manifest_path.suffix == ".parquet" and manifest_path.with_suffix(".csv").exists():
            manifest_path = manifest_path.with_suffix(".csv")
        elif manifest_path.suffix == ".csv" and manifest_path.with_suffix(".parquet").exists():
            manifest_path = manifest_path.with_suffix(".parquet")
        else:
            print(f"[ERROR] Manifest not found: {manifest_path}")
            sys.exit(1)

    manifest = ParquetManifest(manifest_path)
    rows = manifest.get_all_rows()
    print(f"Loaded {len(rows)} samples from {manifest_path}.")

    # Filter rows with existing latent files
    valid_rows = [r for r in rows if Path(r["latent_path"]).exists()]
    if len(valid_rows) == 0:
        print("[ERROR] No existing latent files found from manifest rows.")
        sys.exit(1)

    if len(valid_rows) > args.max_samples:
        import random
        random.seed(42)
        valid_rows = random.sample(valid_rows, args.max_samples)

    latent_store = LatentStore(root_dir=PROJECT_ROOT / "data" / "latents")

    import numpy as np

    # Group by lambda
    lambda_stats = defaultdict(lambda: {"mean_abs": [], "std": [], "zero_frac": []})
    dataset_features = []
    dataset_labels = []

    print(f"Analyzing {len(valid_rows)} latent files...")
    for idx, r in enumerate(valid_rows):
        l_idx = r["lambda_idx"]
        lat_path = Path(r["latent_path"])
        data = latent_store.load_latents(lat_path)
        y_hat = data["y_hat"]

        feats = compute_vector_features(y_hat)
        dataset_features.append(feats)
        dataset_labels.append(l_idx)

        lambda_stats[l_idx]["mean_abs"].append(feats[0])
        lambda_stats[l_idx]["std"].append(feats[1])
        lambda_stats[l_idx]["zero_frac"].append(feats[2])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(valid_rows)} latents.")

    # Train / Test split for predictability evaluation (80/20)
    indices = list(range(len(dataset_features)))
    import random
    random.seed(0)
    random.shuffle(indices)

    split_pt = int(0.8 * len(indices))
    train_idx = indices[:split_pt]
    test_idx = indices[split_pt:]

    X_train = [dataset_features[i] for i in train_idx]
    y_train = [dataset_labels[i] for i in train_idx]
    X_test = [dataset_features[i] for i in test_idx]
    y_test = [dataset_labels[i] for i in test_idx]

    pred_accuracy = nearest_centroid_classifier(X_train, y_train, X_test, y_test)
    num_classes = len(set(dataset_labels))
    chance_accuracy = (1.0 / num_classes) if num_classes > 0 else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Stage 3 Latent Diagnostics & Predictability Report\n\n")
        f.write(f"- **Manifest:** `{manifest_path.name}`\n")
        f.write(f"- **Evaluated Latents:** {len(valid_rows)}\n")
        f.write(f"- **Operating Lambdas:** {num_classes}\n\n")

        f.write("## 1. Lambda Predictability from Pooled Latent Statistics\n\n")
        f.write(
            "Assesses whether simple summary statistics (mean magnitude, standard deviation, sparsity, energy percentiles) "
            "trivially disclose the codec operating point:\n\n"
        )
        f.write(f"- **Predictor Accuracy:** {pred_accuracy * 100.0:.2f}%\n")
        f.write(f"- **Random Chance Baseline:** {chance_accuracy * 100.0:.2f}%\n\n")

        if pred_accuracy > 0.70:
            f.write(
                "> [!NOTE]\n"
                f"> Lambda is highly predictable from latent summary statistics ({pred_accuracy * 100.0:.1f}% accuracy vs {chance_accuracy * 100.0:.1f}% chance). "
                "Per Section 9.2 of the design document, blind mode models can implicitly deduce the operating lambda from channel statistics alone.\n\n"
            )

        f.write("## 2. Per-Lambda Latent Statistics\n\n")
        f.write("| Lambda Idx | Evaluated Samples | Mean Absolute Value | Mean Std | Zero Fraction (Sparsity) |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for l_idx in sorted(lambda_stats.keys()):
            st = lambda_stats[l_idx]
            m_abs = sum(st["mean_abs"]) / len(st["mean_abs"])
            m_std = sum(st["std"]) / len(st["std"])
            m_zero = sum(st["zero_frac"]) / len(st["zero_frac"])
            f.write(f"| {l_idx} | {len(st['mean_abs'])} | {m_abs:.4f} | {m_std:.4f} | {m_zero * 100.0:.2f}% |\n")

    print(f"\nReport generated: {output_path}")
    print(f"Lambda Predictor Accuracy: {pred_accuracy * 100.0:.2f}% (Chance: {chance_accuracy * 100.0:.2f}%)")


if __name__ == "__main__":
    main()
