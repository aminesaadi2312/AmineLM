"""Plot the validation curves of several runs.

    python scripts/plot.py out/base out/mha out/mqa --out assets/ablation_gqa.png
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run folders containing log.csv")
    p.add_argument("--out", default="assets/val_loss.png")
    p.add_argument("--labels", nargs="*", default=None, help="legend labels (default: folder names)")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    plt.figure(figsize=(7, 4.2))
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    for run, label in zip(args.runs, labels):
        rows = list(csv.DictReader(open(os.path.join(run, "log.csv"))))
        plt.plot([int(r["step"]) for r in rows], [float(r["val"]) for r in rows], label=label)
    plt.xlabel("optimizer step")
    plt.ylabel("validation loss")
    if args.title:
        plt.title(args.title)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print("->", args.out)


if __name__ == "__main__":
    main()
