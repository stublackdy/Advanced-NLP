import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVENT_PHRASE = "Stored 1 new unique documents"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"


def extract_event_times(log_path: Path):
    times = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if EVENT_PHRASE in line:
                ts_str = line.split(" - ", 1)[0].strip()
                times.append(datetime.strptime(ts_str, TIMESTAMP_FORMAT))
    if not times:
        raise ValueError(f"No '{EVENT_PHRASE}' entries found in {log_path}")
    return times


def plot_progress(times, output_path=None, target_docs=None, fit_degree=3):
    start = times[0]
    elapsed_hours = [(t - start).total_seconds() / 3600 for t in times]
    doc_counts = list(range(1, len(times) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(doc_counts, elapsed_hours, marker="o", linewidth=1)
    plt.title("Cumulative Processing Time vs. Documents Completed")
    plt.xlabel("Documents processed (cumulative count)")
    plt.ylabel("Elapsed time since first document (hours)")
    plt.grid(True, alpha=0.3)

    coeffs = np.polyfit(doc_counts, elapsed_hours, deg=1)
    fitted_hours = np.polyval(coeffs, doc_counts)
    degree = min(fit_degree, len(doc_counts) - 1)
    if degree < 1:
        degree = 1

    coeffs = np.polyfit(doc_counts, elapsed_hours, deg=degree)
    fitted_hours = np.polyval(coeffs, doc_counts)
    label = "Polynomial fit (deg=" + str(degree) + ")"
    # plt.plot(doc_counts, fitted_hours, color="red", linestyle="--", label=label)

    prediction_msg = None
    if target_docs is not None:
        predicted_hours = float(np.polyval(coeffs, target_docs))
        prediction_msg = (
            f"Predicted total time for {target_docs} documents: {predicted_hours:.2f} hours"
        )
        # plt.scatter([target_docs], [predicted_hours], color="red", marker="x", zorder=5)
        # plt.annotate(
        #     f"{predicted_hours:.1f} h",
        #     (target_docs, predicted_hours),
        #     textcoords="offset points",
        #     xytext=(5, 5),
        #     fontsize=9,
        #     color="red",
        # )

    plt.legend()
    plt.tight_layout()

    if prediction_msg:
        print(prediction_msg)

    if output_path:
        plt.savefig(output_path, dpi=200)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize processing time for MiniRAG document ingestion."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("../LiHua-World/minirag.log"),
        help="Path to minirag log file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the plot instead of showing it",
    )
    parser.add_argument(
        "--target-docs",
        type=int,
        default=424,
        help="Document count to forecast total processing time for",
    )
    parser.add_argument(
        "--fit-degree",
        type=int,
        default=2,
        help="Polynomial degree for the time fit (1 = linear)",
    )
    args = parser.parse_args()

    times = extract_event_times(args.log)
    plot_progress(times, args.output, args.target_docs, args.fit_degree)


if __name__ == "__main__":
    main()