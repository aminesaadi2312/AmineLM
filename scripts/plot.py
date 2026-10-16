"""Séance M7 · Trace les courbes de validation de plusieurs runs.

    python scripts/plot.py out/base out/mha out/mqa --out assets/ablation_gqa.png
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="dossiers contenant log.csv")
    p.add_argument("--out", default="assets/val_loss.png")
    args = p.parse_args()
    plt.figure(figsize=(7, 4.2))
    for run in args.runs:
        rows = list(csv.DictReader(open(os.path.join(run, "log.csv"))))
        plt.plot([int(r["step"]) for r in rows], [float(r["val"]) for r in rows], label=os.path.basename(run.rstrip("/")))
    plt.xlabel("pas d'optimisation")
    plt.ylabel("perte de validation")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print("→", args.out)


if __name__ == "__main__":
    main()
