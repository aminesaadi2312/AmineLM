"""pytest tests/test_tokenizer.py"""
from minigpt.tokenizer import CharTokenizer, BPETokenizer, get_stats, merge

TEXT = "le chat mange, les chats mangent ; le chien mange aussi. " * 8 + "élève, été, Noël 🙂"


def test_char_vocab_sorted_and_inverse():
    tok = CharTokenizer.from_text("hello world")
    assert tok.chars == sorted(set("hello world"))
    assert tok.vocab_size == len(set("hello world"))
    assert all(tok.itos[i] == c for c, i in tok.stoi.items())


def test_char_round_trip(tmp_path):
    tok = CharTokenizer.from_text(TEXT)
    s = "les chats mangent"
    assert tok.decode(tok.encode(s)) == s
    tok.save(tmp_path / "t.json")
    assert CharTokenizer.load(tmp_path / "t.json").encode(s) == tok.encode(s)


def test_get_stats_and_merge():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert merge([7, 7, 7], (7, 7), 5) == [5, 7], "merges do not overlap"


def test_bpe_first_merge_is_most_frequent():
    tok = BPETokenizer().train(TEXT, 256 + 1)
    stats = get_stats(list(TEXT.encode("utf-8")))
    assert list(tok.merges) == [max(stats, key=stats.get)]


def test_bpe_unicode_round_trip_and_compression():
    tok = BPETokenizer().train(TEXT, 256 + 40)
    assert len(tok.merges) == 40 and tok.vocab_size == 296
    for s in ["élève 🙂 Noël", "le chat", "", "zzz"]:
        assert tok.decode(tok.encode(s)) == s
    assert len(tok.encode(TEXT)) < 0.6 * len(TEXT.encode("utf-8"))


def test_bpe_encode_follows_merge_order():
    tok = BPETokenizer()
    tok.merges = {(97, 98): 256, (256, 99): 257, (98, 99): 258}
    for (a, b), i in tok.merges.items():
        tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
    assert tok.encode("abc") == [257], "(a,b) was learned first, so it is merged before (b,c)"
