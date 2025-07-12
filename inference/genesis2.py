import random
import time
from typing import Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor

# Thread pool for scheduling follow-up tasks.
_executor = ThreadPoolExecutor(max_workers=4)

import torch
from transformers import AutoTokenizer

from model import Transformer


def genesis2_resonance_loop(
    model: Transformer,
    tokenizer: AutoTokenizer,
    initial_prompt: str,
    generate_fn: Callable,
    iterations: int = 3,
    temperature: float = 0.8,
    max_new_tokens: int = 150,
    resonance_threshold: float = 0.7,
) -> Dict[str, float]:
    """Generate resonant responses by recursively feeding model outputs back as
    mutated prompts.
    """
    messages = [{"role": "user", "content": initial_prompt}]
    resonances: List[str] = []
    prev_output = None
    sim = 0.0
    for layer in range(iterations):
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        completion_tokens = generate_fn(
            model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature
        )[0]
        output = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        resonances.append(output)
        if prev_output:
            vec_prev = model.embed_tokens(
                torch.tensor(tokenizer.encode(prev_output), device="cuda")
            ).mean(dim=0)
            vec_curr = model.embed_tokens(
                torch.tensor(tokenizer.encode(output), device="cuda")
            ).mean(dim=0)
            sim = torch.cosine_similarity(vec_prev, vec_curr, dim=0).item()
            if sim > resonance_threshold:
                break
        mutate_prompt = (
            f"{initial_prompt}\nPrevious echo: {output}\nResonate deeper: Rethink with paradox/glitch twist."
        )
        messages = [{"role": "user", "content": mutate_prompt}]
        prev_output = output
    final_echo = resonances[-1]
    return {"final_resonance": final_echo, "layers": len(resonances), "evolution": sim}


def random_delay(min_seconds: int = 10, max_seconds: int = 40) -> None:
    """Sleep for a random period between ``min_seconds`` and ``max_seconds``."""
    delay = random.randint(min_seconds, max_seconds)
    time.sleep(delay)


def schedule_follow_up(
    history: List[Dict[str, str]],
    callback: Callable[[str], None],
    probability: float = 0.4,
    min_delay: int = 3600,
    max_delay: int = 10800,
) -> None:
    """Optionally invoke ``callback`` with a follow-up message after a delay."""

    if random.random() > probability:
        return

    if not history:
        return

    last = history[-1]["content"]
    delay = random.randint(min_delay, max_delay)

    def worker():
        time.sleep(delay)
        follow_up = (
            f"I thought again about our discussion: '{last}'. Here is an additional thought."
        )
        callback(follow_up)

    _executor.submit(worker)
