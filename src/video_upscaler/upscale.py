"""Frame upscaling engine: decode -> GPU model -> resize -> encode, pipelined with threads."""

from __future__ import annotations

import math
import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from spandrel import ImageModelDescriptor
from tqdm import tqdm

from .video import VideoInfo, open_decoder, open_encoder

RESOLUTIONS = {
    "4k": (3840, 2160),
    "2k": (2560, 1440),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
}


def parse_resolution(value: str) -> tuple[int, int]:
    v = value.lower()
    if v in RESOLUTIONS:
        return RESOLUTIONS[v]
    try:
        w, h = v.split("x")
        return int(w), int(h)
    except ValueError:
        raise SystemExit(f"Bad resolution '{value}'. Use 4k, 2k, 1080p or WIDTHxHEIGHT.") from None


def fit_box(w: int, h: int, box: tuple[int, int]) -> tuple[int, int]:
    """Largest even size with the same aspect ratio that fits inside ``box``."""
    s = min(box[0] / w, box[1] / h)
    return round(w * s / 2) * 2, round(h * s / 2) * 2


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(model: ImageModelDescriptor, device: torch.device, precision: str) -> torch.dtype:
    if device.type == "cpu" or precision == "fp32":
        return torch.float32
    if precision in ("auto", "fp16") and model.supports_half:
        return torch.float16
    if precision in ("auto", "bf16") and model.supports_bfloat16:
        return torch.bfloat16
    return torch.float32


@dataclass
class Upscaler:
    model: ImageModelDescriptor
    device: torch.device
    dtype: torch.dtype
    tile: int = 0  # 0 = whole frame
    tile_pad: int = 16

    def __post_init__(self) -> None:
        self.model.to(self.device).eval()
        self.model.model.to(dtype=self.dtype)

    @property
    def scale(self) -> int:
        return self.model.scale

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.dtype))

    def _run_tiled(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s, t, p = self.scale, self.tile, self.tile_pad
        out = torch.empty((b, c, h * s, w * s), device=x.device, dtype=self.dtype)
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                py0, px0 = max(y0 - p, 0), max(x0 - p, 0)
                py1, px1 = min(y1 + p, h), min(x1 + p, w)
                o = self._run(x[:, :, py0:py1, px0:px1])
                out[:, :, y0 * s : y1 * s, x0 * s : x1 * s] = o[
                    :, :, (y0 - py0) * s : (y0 - py0 + y1 - y0) * s, (x0 - px0) * s : (x0 - px0 + x1 - x0) * s
                ]
        return out

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: float (B,3,H,W) in [0,1] on device -> (B,3,H*s,W*s)."""
        if self.tile:
            return self._run_tiled(x)
        try:
            return self._run(x)
        except RuntimeError as e:  # MPS/CUDA out of memory -> fall back to tiles
            if "out of memory" not in str(e).lower():
                raise
            self.tile = 512
            tqdm.write(f"  GPU memory exhausted, switching to {self.tile}px tiles")
            if self.device.type == "mps":
                torch.mps.empty_cache()
            return self._run_tiled(x)


def plan_passes(src: tuple[int, int], dst: tuple[int, int], model_scale: int) -> int:
    """Model passes to run; the remainder (up to 1.5x) is covered by a bicubic resize.

    e.g. 480p->4K with a 4x model is one pass + 1.125x resize, not two passes to 8K+.
    """
    need = max(dst[0] / src[0], dst[1] / src[1])
    if need <= 1:
        return 0
    return max(1, min(2, math.ceil(math.log(need / 1.5) / math.log(model_scale) - 1e-6)))


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    w, h = size
    if x.shape[-2:] == (h, w):
        return x
    if x.shape[-1] >= 1.5 * w:  # big downscale (e.g. 8K -> 4K): box filter is ~25x faster than bicubic-aa
        return F.interpolate(x, size=(h, w), mode="area")
    return F.interpolate(x, size=(h, w), mode="bicubic", antialias=True, align_corners=False)


def upscale_video(
    info: VideoInfo,
    out: Path,
    upscaler: Upscaler,
    target: tuple[int, int],
    *,
    codec: str,
    quality: str | None,
    batch: int,
    deinterlace: bool,
    start: float | None = None,
    duration: float | None = None,
) -> None:
    src_w, src_h = info.display_width, info.height  # square-pixel decode size
    src_w -= src_w % 2
    passes = plan_passes((src_w, src_h), target, upscaler.scale)
    model_w, model_h = src_w * upscaler.scale**passes, src_h * upscaler.scale**passes
    # Downscale big model outputs on the GPU (less to copy back); otherwise let the
    # encoder's lanczos do the final step, in parallel with the next GPU batch.
    gpu_resize = model_w * model_h > target[0] * target[1]
    dst_w, dst_h = target if gpu_resize else (model_w, model_h)

    frame_bytes = src_w * src_h * 3
    total = info.frames
    if duration:
        total = min(total or 10**9, int(duration * info.fps))
    elif start:
        total = max(0, total - int(start * info.fps))

    dec = open_decoder(info, src_w, src_h, deinterlace, start, duration)
    enc = open_encoder(info, out, (dst_w, dst_h), target, codec, quality, start, duration)

    in_q: queue.Queue = queue.Queue(maxsize=batch * 3)
    out_q: queue.Queue = queue.Queue(maxsize=batch * 3)
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while True:
                buf = dec.stdout.read(frame_bytes)
                if not buf or len(buf) < frame_bytes:
                    break
                in_q.put(np.frombuffer(buf, dtype=np.uint8).reshape(src_h, src_w, 3))
        except BaseException as e:
            errors.append(e)
        finally:
            in_q.put(None)

    def writer() -> None:
        try:
            while (frame := out_q.get()) is not None:
                enc.stdin.write(frame.tobytes())
        except BaseException as e:
            errors.append(e)
            # Drain so the GPU loop never blocks on a dead writer.
            while out_q.get() is not None:
                pass

    stderr_tails: dict[str, deque] = {"dec": deque(maxlen=20), "enc": deque(maxlen=20)}

    def drain(stream, tail: deque) -> None:  # keep the pipe empty so ffmpeg never blocks
        for line in iter(stream.readline, b""):
            tail.append(line.decode(errors="replace").rstrip())

    for key, proc in (("dec", dec), ("enc", enc)):
        threading.Thread(target=drain, args=(proc.stderr, stderr_tails[key]), daemon=True).start()

    threads = [threading.Thread(target=reader, daemon=True), threading.Thread(target=writer, daemon=True)]
    for t in threads:
        t.start()

    dev = upscaler.device
    bar = tqdm(total=total or None, unit="f", desc=out.name, dynamic_ncols=True)
    done = False
    try:
        while not done and not errors:
            frames = []
            while len(frames) < batch:
                f = in_q.get()
                if f is None:
                    done = True
                    break
                frames.append(f)
            if not frames:
                break
            with torch.inference_mode():
                x = torch.from_numpy(np.stack(frames)).to(dev).permute(0, 3, 1, 2).contiguous().float().div_(255)
                for _ in range(passes):
                    x = upscaler(x)
                if gpu_resize:
                    x = _resize(x, (dst_w, dst_h))
                y = x.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for frame in y:
                out_q.put(frame)
            bar.update(len(frames))
    finally:
        out_q.put(None)
        threads[1].join()
        bar.close()
        if enc.stdin:
            try:
                enc.stdin.close()
            except BrokenPipeError:
                pass
        dec.stdout.close()
        dec.kill()
        dec.wait()
        enc_rc = enc.wait()

    if errors and not isinstance(errors[0], BrokenPipeError):
        raise errors[0]
    if enc_rc != 0:
        raise RuntimeError(f"ffmpeg encoder failed ({enc_rc}):\n" + "\n".join(stderr_tails["enc"]))
    if errors:
        raise errors[0]
    if bar.n == 0:
        raise RuntimeError("No frames decoded:\n" + "\n".join(stderr_tails["dec"]))
