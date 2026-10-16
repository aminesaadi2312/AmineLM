"""Séance M6 · Génération depuis un checkpoint.

    python scripts/sample.py --prompt "ROMEO:" --n 400 --temperature 0.8 --top_k 50
    python scripts/sample.py --no_cache          # pour comparer la vitesse
"""
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minigpt import GPTConfig
from minigpt.model import GPT
from minigpt.tokenizer import CharTokenizer
from scripts.train import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    tok = CharTokenizer.load(os.path.join(ck["data_dir"], "tokenizer.json"))
    torch.manual_seed(args.seed)
    raise NotImplementedError("séance M6 : à compléter")


if __name__ == "__main__":
    main()
