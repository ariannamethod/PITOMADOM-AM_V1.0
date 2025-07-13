import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub torch and transformers modules before importing genesis2
torch_stub = types.ModuleType("torch")
def _tensor(data, device=None):
    return data
class _Result(float):
    def item(self):
        return float(self)
def _cosine_similarity(a, b, dim=0):
    return _Result((a * b) / (abs(a) * abs(b)))
torch_stub.tensor = _tensor
torch_stub.cosine_similarity = _cosine_similarity
sys.modules.setdefault("torch", torch_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoTokenizer = type("AutoTokenizer", (), {})
sys.modules.setdefault("transformers", transformers_stub)

model_stub = types.ModuleType("model")
model_stub.Transformer = type("Transformer", (), {})
sys.modules.setdefault("model", model_stub)

from inference import genesis2

class DummyTorch:
    @staticmethod
    def tensor(data, device=None):
        return data

    @staticmethod
    def cosine_similarity(a, b, dim=0):
        # simple cosine for scalars
        val = (a * b) / (abs(a) * abs(b))
        class Result(float):
            def item(self):
                return float(self)
        return Result(val)

class DummyTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.vocab = {}
        self.reverse = {}

    def encode(self, text):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                idx = len(self.vocab) + 1
                self.vocab[word] = idx
                self.reverse[idx] = word
            ids.append(self.vocab[word])
        return ids

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(self.reverse.get(t, "") for t in tokens)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return self.encode(messages[-1]["content"])

class DummyModel:
    def embed_tokens(self, tokens):
        # convert list of ints to floats
        return DummyEmbedding([float(t) for t in tokens])

class DummyEmbedding(list):
    def mean(self, dim=0):
        return sum(self) / len(self)

def test_genesis2_resonance_loop(monkeypatch):
    tokenizer = DummyTokenizer()
    model = DummyModel()
    monkeypatch.setattr(genesis2, "torch", DummyTorch)

    def dummy_generate_fn(model, prompts, max_new_tokens, eos_token_id, temperature):
        return [tokenizer.encode("echo")]

    result = genesis2.genesis2_resonance_loop(
        model,
        tokenizer,
        "hello",
        dummy_generate_fn,
        iterations=5,
        resonance_threshold=0.5,
    )
    assert result["layers"] == 2
    assert result["final_resonance"] == "echo"
    assert result["evolution"] == 1.0

def test_random_delay(monkeypatch):
    called = []
    monkeypatch.setattr(genesis2.random, "randint", lambda a, b: 1)
    monkeypatch.setattr(genesis2, "time", types.SimpleNamespace(sleep=lambda x: called.append(x)))
    genesis2.random_delay(min_seconds=1, max_seconds=5)
    assert called == [1]
