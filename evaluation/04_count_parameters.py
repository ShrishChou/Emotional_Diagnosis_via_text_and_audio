"""Count the parameters of every model on the inference path -> evals/system/params.json.

Counts *total* parameters (sum of numel over model.parameters()), loaded from the
same weights the app uses (the classifier heads named in models/meta.json).
Asserts the total is at most 6e9, the brief's limit.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"  # count the cached weights; never download here
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModel, AutoModelForCausalLM, WavLMModel, WhisperForConditionalGeneration

from config import CFG, write_json
from src.fusion import load_deployed

LIMIT = 6_000_000_000
DEPLOYED = json.loads((CFG.models_dir / "meta.json").read_text())["deployed"]


def count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    loaders = {
        "text_encoder": (CFG.text_model, lambda: AutoModel.from_pretrained(CFG.text_model, add_pooling_layer=False)),
        "audio_encoder": (CFG.audio_model, lambda: WavLMModel.from_pretrained(CFG.audio_model)),
        "text_only_head": (f"models/{DEPLOYED['text_only']}", lambda: load_deployed()["text_only"][0]),
        "fused_head": (f"models/{DEPLOYED['fused']}", lambda: load_deployed()["fused"][0]),
        "asr": (CFG.asr_model, lambda: WhisperForConditionalGeneration.from_pretrained(CFG.asr_model)),
        "reply_llm": (CFG.llm_model, lambda: AutoModelForCausalLM.from_pretrained(CFG.llm_model)),
        "tts": (CFG.tts_model, lambda: __import__("src.tts", fromlist=["Speaker"]).Speaker().model),
    }
    components = {}
    for name, (source, load) in loaders.items():
        model = load()
        components[name] = {"source": source, "params": count(model)}
        print(f"{name:15s} {source:35s} {components[name]['params']:>15,d}")
        del model
        gc.collect()

    total = sum(c["params"] for c in components.values())
    print(f"{'total':15s} {'':35s} {total:>15,d}  (limit {LIMIT:,d})")
    assert total <= LIMIT, "over the 6B parameter limit"
    write_json(CFG.out("system", "params.json"), {
        "components": components,
        "total": total,
        "limit": LIMIT,
        "fraction_of_limit": round(total / LIMIT, 4),
        "note": "Total parameters (not trainable or active). ASR runs only when no transcript is given, "
                "and TTS only when the chat page is not muted; both are included in the total. "
                "No VAD (push-to-talk instead).",
    })


if __name__ == "__main__":
    main()
