"""Model registry, downloader and loader.

All weights are stored in ``<repo>/models`` (override with ``VIDEO_UPSCALER_MODELS``).
Every model here is loaded through spandrel, so the architecture (ESRGAN/RRDB,
SRVGGNetCompact, SPAN, RCAN, DAT, ...) is detected from the state dict.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from spandrel import ImageModelDescriptor, ModelLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("VIDEO_UPSCALER_MODELS", REPO_ROOT / "models"))

_RE = "https://github.com/xinntao/Real-ESRGAN/releases/download"
_HF = "https://huggingface.co"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    scale: int
    arch: str
    license: str
    description: str
    speed: str  # "fast" | "medium" | "slow"


MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        # --- Anime / cartoon -------------------------------------------------
        ModelSpec(
            "realesr-animevideov3",
            "realesr-animevideov3.pth",
            f"{_RE}/v0.2.5.0/realesr-animevideov3.pth",
            4,
            "SRVGGNetCompact",
            "BSD-3-Clause",
            "Real-ESRGAN model trained specifically for anime *video*; very fast, temporally stable.",
            "fast",
        ),
        ModelSpec(
            "2x-AnimeSharpV4-Fast",
            "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
            f"{_HF}/Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
            2,
            "RCAN (PixelUnshuffle)",
            "CC-BY-NC-SA-4.0",
            "2025 anime model tuned for video (handles MPEG2/H264 artifacts); faithful and sharp.",
            "medium",
        ),
        ModelSpec(
            "RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x4plus_anime_6B.pth",
            f"{_RE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            4,
            "ESRGAN (RRDB 6B)",
            "BSD-3-Clause",
            "Classic Real-ESRGAN anime illustration model; strong line cleanup, flatter shading.",
            "medium",
        ),
        ModelSpec(
            "4x-AnimeSharp",
            "4x-AnimeSharp.safetensors",
            f"{_HF}/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.safetensors",
            4,
            "ESRGAN (RRDB)",
            "CC-BY-NC-SA-4.0",
            "Sharp anime/cartoon model that keeps more texture than Real-ESRGAN anime.",
            "slow",
        ),
        # --- Live action: films / series ---------------------------------------
        ModelSpec(
            "4x-ClearRealityV1",
            "4x-ClearRealityV1.safetensors",
            f"{_HF}/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.safetensors",
            4,
            "SPAN",
            "CC-BY-NC-SA-4.0",
            "Fast SPAN model for natural/real-world footage; good default for TV series.",
            "fast",
        ),
        ModelSpec(
            "realesr-general-x4v3",
            "realesr-general-x4v3.pth",
            f"{_RE}/v0.2.5.0/realesr-general-x4v3.pth",
            4,
            "SRVGGNetCompact",
            "BSD-3-Clause",
            "Small general real-world model with strong denoising; blended with -wdn for denoise control.",
            "fast",
        ),
        ModelSpec(
            "realesr-general-wdn-x4v3",
            "realesr-general-wdn-x4v3.pth",
            f"{_RE}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            4,
            "SRVGGNetCompact",
            "BSD-3-Clause",
            "Weak-denoise twin of realesr-general-x4v3; keeps grain/texture when blended in.",
            "fast",
        ),
        ModelSpec(
            "RealESRGAN_x2plus",
            "RealESRGAN_x2plus.pth",
            f"{_RE}/v0.2.1/RealESRGAN_x2plus.pth",
            2,
            "ESRGAN (RRDB)",
            "BSD-3-Clause",
            "Real-ESRGAN 2x for degraded real-world footage (1080p->4K, 720p->1440p).",
            "slow",
        ),
        ModelSpec(
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus.pth",
            f"{_RE}/v0.1.0/RealESRGAN_x4plus.pth",
            4,
            "ESRGAN (RRDB)",
            "BSD-3-Clause",
            "Real-ESRGAN 4x for heavily degraded real-world footage (SD/480p movies).",
            "slow",
        ),
        ModelSpec(
            "4x_foolhardy_Remacri",
            "4x_foolhardy_Remacri.pth",
            f"{_HF}/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
            4,
            "ESRGAN (RRDB)",
            "CC-BY-NC-SA-4.0",
            "Detail-preserving real-world model; keeps grain/texture rather than smoothing.",
            "slow",
        ),
        # --- General / abstract ------------------------------------------------
        ModelSpec(
            "4x-UltraSharpV2-Lite",
            "4x-UltraSharpV2_Lite.safetensors",
            f"{_HF}/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2_Lite.safetensors",
            4,
            "PLKSR",
            "CC-BY-NC-SA-4.0",
            "2025 general-purpose sharpener; good for CGI, games, abstract and art content.",
            "medium",
        ),
        ModelSpec(
            "4x-UltraSharpV2",
            "4x-UltraSharpV2.safetensors",
            f"{_HF}/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2.safetensors",
            4,
            "DAT",
            "CC-BY-NC-SA-4.0",
            "Highest-quality general model in the set (transformer); slow on video.",
            "slow",
        ),
    ]
}


@dataclass(frozen=True)
class ContentType:
    name: str
    description: str
    model: str  # fast default, practical for full-length video
    hq: str  # higher quality, much slower (--hq)
    alternatives: tuple[str, ...] = ()
    denoise: float | None = None  # realesr-general-x4v3 blend (0..1), see load_model

    @property
    def all_models(self) -> list[str]:
        names = [self.model, self.hq, *self.alternatives]
        if DENOISE_PAIR[0] in names:
            names.append(DENOISE_PAIR[1])
        return list(dict.fromkeys(names))


DENOISE_PAIR = ("realesr-general-x4v3", "realesr-general-wdn-x4v3")

CONTENT_TYPES: dict[str, ContentType] = {
    t.name: t
    for t in [
        ContentType(
            "anime",
            "Japanese animation: flat colours, hard line art.",
            model="realesr-animevideov3",
            hq="2x-AnimeSharpV4-Fast",
            alternatives=("RealESRGAN_x4plus_anime_6B", "4x-AnimeSharp"),
        ),
        ContentType(
            "cartoon",
            "Western 2D cartoons, older cel animation.",
            model="realesr-animevideov3",
            hq="4x-AnimeSharp",
            alternatives=("RealESRGAN_x4plus_anime_6B", "2x-AnimeSharpV4-Fast"),
        ),
        ContentType(
            "series",
            "Modern live-action TV (digital, mostly clean sources).",
            model="4x-ClearRealityV1",
            hq="RealESRGAN_x2plus",
            alternatives=("realesr-general-x4v3",),
        ),
        ContentType(
            "movie",
            "Live-action film (DVD/Blu-ray/web rips); keeps some film grain.",
            model="realesr-general-x4v3",
            hq="4x_foolhardy_Remacri",
            alternatives=("4x-ClearRealityV1", "RealESRGAN_x2plus"),
            denoise=0.5,
        ),
        ContentType(
            "old-movie",
            "Traditional/old film: grain, noise, VHS/SD and compression damage.",
            model="realesr-general-x4v3",
            hq="RealESRGAN_x4plus",
            alternatives=("4x_foolhardy_Remacri",),
            denoise=0.8,
        ),
        ContentType(
            "abstract",
            "CGI, games, motion graphics, art and abstract visuals.",
            model="4x-ClearRealityV1",
            hq="4x-UltraSharpV2-Lite",
            alternatives=("4x-UltraSharpV2",),
        ),
        ContentType(
            "general",
            "Unknown/mixed content; safe all-rounder.",
            model="4x-ClearRealityV1",
            hq="4x-UltraSharpV2-Lite",
            alternatives=("realesr-general-x4v3",),
        ),
    ]
}


def model_path(name: str) -> Path:
    return MODELS_DIR / MODELS[name].filename


def download(name: str, force: bool = False) -> Path:
    spec = MODELS[name]
    dest = model_path(name)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "video-upscaler"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {name}") as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    shutil.move(tmp, dest)
    return dest


def download_for_type(type_name: str) -> list[Path]:
    return [download(n) for n in CONTENT_TYPES[type_name].all_models]


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            return sd[key]
    return sd


def load_model(name: str, denoise: float | None = None) -> ImageModelDescriptor:
    """Load a model by registry name (or a path to any spandrel-supported file).

    ``denoise`` (0..1) only applies to realesr-general-x4v3: it linearly blends the
    weights with the wdn (weak denoise) variant, like Real-ESRGAN's --denoise_strength.
    1.0 = full denoise, 0.0 = keep all grain.
    """
    if name not in MODELS:
        path = Path(name).expanduser()
        if not path.exists():
            raise SystemExit(f"Unknown model '{name}'. Use --list-models.")
        model = ModelLoader().load_from_file(path)
    elif name == DENOISE_PAIR[0] and denoise is not None and denoise < 1:
        a = _load_state_dict(download(DENOISE_PAIR[0]))
        b = _load_state_dict(download(DENOISE_PAIR[1]))
        mixed = {k: a[k] * denoise + b[k] * (1 - denoise) for k in a}
        model = ModelLoader().load_from_state_dict(mixed)
    else:
        model = ModelLoader().load_from_file(download(name))
    if not isinstance(model, ImageModelDescriptor):
        raise SystemExit(f"{name} is not a single-image super-resolution model.")
    return model
