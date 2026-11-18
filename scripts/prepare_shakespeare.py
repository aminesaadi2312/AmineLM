"""Prepare the data: text -> ids (character tokenizer) -> data/<name>/{train,val}.bin (uint16).

    python scripts/prepare_shakespeare.py                      # Tiny Shakespeare (1.1M characters)
    python scripts/prepare_shakespeare.py --input my_text.txt --name my_corpus
"""
import argparse, json, os, sys, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minigpt.tokenizer import CharTokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="UTF-8 text file (default: Tiny Shakespeare)")
    p.add_argument("--name", default="shakespeare")
    p.add_argument("--val_frac", type=float, default=0.1)
    args = p.parse_args()
    out = os.path.join("data", args.name)
    os.makedirs(out, exist_ok=True)
    if args.input:
        text = open(args.input, encoding="utf-8").read()
    else:
        raw = os.path.join(out, "input.txt")
        if not os.path.exists(raw):
            print("downloading Tiny Shakespeare...")
            urllib.request.urlretrieve(URL, raw)
        text = open(raw, encoding="utf-8").read()
    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.uint16)        # uint16: fine for vocabularies up to 65,535 tokens
    n = int((1 - args.val_frac) * len(ids))        # the last 10% is validation: never seen in training
    ids[:n].tofile(os.path.join(out, "train.bin"))
    ids[n:].tofile(os.path.join(out, "val.bin"))
    tok.save(os.path.join(out, "tokenizer.json"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"vocab_size": tok.vocab_size, "tokenizer": "char", "n_train": n, "n_val": len(ids) - n}, f)
    print(f"{len(text):,} characters · vocab {tok.vocab_size} · train {n:,} · val {len(ids) - n:,} tokens -> {out}/")


if __name__ == "__main__":
    main()
