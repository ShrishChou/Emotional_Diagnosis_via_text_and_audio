"""Record the hardware and library versions to results/hardware.json, then
download every model's weights into the local Hugging Face cache (the only
step that needs the network; everything after runs offline)."""
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil
import torch

from config import CFG, write_json


def cpu_model() -> str:
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return out.stdout.strip()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def lib_versions() -> dict:
    import importlib.metadata as md
    names = ["torch", "torchaudio", "transformers", "accelerate", "soundfile", "librosa",
             "scikit-learn", "pandas", "numpy", "ftfy", "psutil"]
    return {n: md.version(n) for n in names}


def main():
    info = {
        "device": CFG.device,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "system_ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "cpu_model": cpu_model(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "libraries": lib_versions(),
        "chosen_llm": CFG.llm_model,
        "llm_dtype": str(CFG.llm_dtype),
        "audio_train_subsample": CFG.audio_train_subsample,
    }
    if CFG.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 1e9, 1)
    elif CFG.device == "mps":
        # Apple Silicon: the GPU shares unified memory with the CPU.
        info["gpu_name"] = f"{cpu_model()} integrated GPU (MPS, unified memory)"
        info["gpu_vram_gb"] = "shared with system RAM"

    write_json(CFG.results_dir / "hardware.json", info)
    for k, v in info.items():
        print(f"{k}: {v}")

    from huggingface_hub import snapshot_download
    for repo in (CFG.text_model, CFG.audio_model, CFG.asr_model, CFG.llm_model):
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt",
                                                "vocab*", "tokenizer*"])
        print(f"cached: {repo}")
    # Speech output: Kokoro's weights are a .pth file plus one voice.
    snapshot_download(CFG.tts_model, allow_patterns=["config.json", "*.pth", f"voices/{CFG.tts_voice}.pt"])
    print(f"cached: {CFG.tts_model} (voice {CFG.tts_voice})")


if __name__ == "__main__":
    main()
