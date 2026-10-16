"""Séance M5 · Entraînement.

    python scripts/train.py --preset tiny --max_iters 2000            # CPU ou Mac : quelques minutes
    python scripts/train.py --preset small --max_iters 5000 --flash   # GPU
"""
import argparse, contextlib, csv, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minigpt import GPTConfig, PRESETS
from minigpt.model import GPT
from minigpt.data import load_split, get_batch
from minigpt.train_utils import get_lr, estimate_loss


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/shakespeare")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    p.add_argument("--device", default="auto")
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--n_kv_head", type=int, default=None, help="ablation GQA : 1 (MQA) … n_head (MHA)")
    p.add_argument("--no_tie", action="store_true", help="ablation : sans weight tying")
    p.add_argument("--flash", action="store_true", help="F.scaled_dot_product_attention (séance M7)")
    p.add_argument("--no_rope", action="store_true", help="ablation : sans RoPE")
    p.add_argument("--post_norm", action="store_true", help="ablation : blocs post-norm")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else device
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    over = dict(PRESETS[args.preset])
    if args.n_kv_head:
        over["n_kv_head"] = args.n_kv_head
    over["tie_weights"] = not args.no_tie
    over["flash"] = args.flash
    over["use_rope"] = not args.no_rope
    over["post_norm"] = args.post_norm
    cfg = GPTConfig(vocab_size=meta["vocab_size"], **over)
    data = {s: load_split(os.path.join(args.data_dir, f"{s}.bin")) for s in ("train", "val")}

    raw = GPT(cfg).to(device)
    print(f"device {device} · {raw.num_params() / 1e6:.2f} M paramètres · config {cfg}")
    opt = raw.configure_optimizer(args.weight_decay, args.lr, device_type=device_type)
    bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
    model = torch.compile(raw) if args.compile else raw

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    log = open(os.path.join(args.out_dir, "log.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "train", "val", "lr", "tokens_per_s"])
    best, tokens, t0 = math.inf, 0, time.time()

    raise NotImplementedError("séance M5 : à compléter")
    print(f"meilleure perte de validation : {best:.3f} · checkpoint : {ckpt_path}")


if __name__ == "__main__":
    main()
