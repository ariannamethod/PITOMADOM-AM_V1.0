import sys
import types
import math
import pytest

class SimpleTensor:
    def __init__(self, data):
        self.data = data

    def mean(self, dim=0):
        if dim != 0:
            raise NotImplementedError
        if isinstance(self.data[0], list):
            cols = len(self.data[0])
            return SimpleTensor([
                sum(row[i] for row in self.data) / len(self.data) for i in range(cols)
            ])
        return SimpleTensor([sum(self.data) / len(self.data)])

    def item(self):
        return self.data[0]


def tensor(data, device=None):
    return SimpleTensor(data)


def cosine_similarity(a, b, dim=0):
    dot = sum(x * y for x, y in zip(a.data, b.data))
    norm_a = math.sqrt(sum(x * x for x in a.data))
    norm_b = math.sqrt(sum(x * x for x in b.data))
    return SimpleTensor([dot / (norm_a * norm_b)])


torch_stub = types.SimpleNamespace(tensor=tensor, cosine_similarity=cosine_similarity)
sys.modules['torch'] = torch_stub

embedding_map = {1: [1.0, 0.0], 3: [1.0, 1.0]}

class Transformer:
    def embed_tokens(self, tokens):
        embeds = [embedding_map[int(t)] for t in tokens.data]
        return tensor(embeds)

model_module = types.SimpleNamespace(Transformer=Transformer)
sys.modules['model'] = model_module

class DummyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return '\n'.join(m['content'] for m in messages)

    def encode(self, text):
        return [token_id_map[t] for t in text.split()]

    def decode(self, tokens, skip_special_tokens=True):
        inverse = {v: k for k, v in token_id_map.items()}
        return ' '.join(inverse[t] for t in tokens)

token_id_map = {'hello': 1, 'hi': 3}

transformers_stub = types.SimpleNamespace(AutoTokenizer=DummyTokenizer)
sys.modules['transformers'] = transformers_stub

import importlib.util
from pathlib import Path

def load_genesis2():
    path = Path(__file__).parents[1] / 'inference' / 'genesis2.py'
    spec = importlib.util.spec_from_file_location('genesis2', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['genesis2'] = module
    spec.loader.exec_module(module)
    return module

def make_generate_fn(outputs):
    out_iter = iter(outputs)
    def _fn(model, tokens, max_new, eos_id, temp):
        return [next(out_iter)]
    return _fn

def test_genesis2_resonance_loop_breaks_on_similarity():
    genesis2 = load_genesis2()
    tokenizer = DummyTokenizer()
    model = Transformer()
    gen_fn = make_generate_fn([[1], [3]])
    result = genesis2.genesis2_resonance_loop(
        model,
        tokenizer,
        'hello',
        gen_fn,
        iterations=3,
        resonance_threshold=0.7,
    )
    assert result['layers'] == 2
    assert result['final_resonance'] == 'hi'
    expected_sim = 1 / math.sqrt(2)
    assert result['evolution'] == pytest.approx(expected_sim, rel=1e-6)

