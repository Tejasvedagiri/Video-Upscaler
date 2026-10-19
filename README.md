# Video-Upscaler

Upscale anime, cartoons, TV series, movies (including old film) and abstract/CGI video to **4K** or **2K**
on Apple Silicon. Models run on the Mac GPU through PyTorch **Metal (MPS)** in fp16/bf16. Decoding and
encoding use ffmpeg with the **VideoToolbox** hardware codecs. Audio, subtitles, chapters and metadata
are carried over from the source.

## Setup

```bash
brew install ffmpeg uv
uv sync                                  # Python 3.12 + deps
uv run video-upscaler --download all     # ~460 MB of weights -> ./models
```

Models are also downloaded automatically the first time they're needed. Set `VIDEO_UPSCALER_MODELS` to
store them elsewhere.

## Usage

```bash
# Single file -> 4K (output defaults to <name>_4k.mp4 next to the input)
uv run video-upscaler ~/Movies/episode01.mkv --type anime

# Whole folder (recursive) -> 2K, into a chosen output folder (mirrors sub-folders)
uv run video-upscaler ~/Movies/Show --type series --resolution 2k --output ~/Movies/Show_2k

# Old film, explicit output file, higher-quality (slow) model
uv run video-upscaler old.avi --type old-movie --hq -o old_4k.mkv

# Preview 10 seconds before committing to a full run
uv run video-upscaler film.mp4 --type movie --start 600 --duration 10 -o preview.mp4
```

| Argument | Meaning |
|---|---|
| `path` | Video file or folder |
| `-t/--type` | `anime`, `cartoon`, `series`, `movie`, `old-movie`, `abstract`, `general` |
| `-o/--output` | Output file (single input) or folder |
| `-r/--resolution` | `4k` (3840×2160), `2k` (2560×1440), `1080p`, or `WxH`. Aspect ratio is kept, so output fits inside the box |
| `--hq` | Use the type's high-quality model (much slower) |
| `-m/--model` | Any registry model name, or a path to any [spandrel](https://github.com/chaiNNer-org/spandrel)-supported `.pth`/`.safetensors` (e.g. from [OpenModelDB](https://openmodeldb.info)) |
| `--denoise 0..1` | Denoise strength for `realesr-general-x4v3` (movie/old-movie). 1 = clean, 0 = keep grain |
| `-c/--codec` | `hevc` (default, 10-bit, hardware), `h264`, `prores` (hardware), `x265`, `x264` (CPU) |
| `-q/--quality` | VideoToolbox `-q:v` (1–100, default 65) or CRF for x264/x265 |
| `--container` | `mp4` (default), `mkv` (keeps every subtitle format), `mov` |
| `-b/--batch`, `--tile`, `--precision` | Performance and memory tuning |
| `--deinterlace auto/on/off` | `bwdif` deinterlace (auto = only streams flagged interlaced) |
| `--overwrite` | Re-process existing outputs (they're skipped by default) |
| `--list-types`, `--list-models` | Show what's available |

## Types and models

| Type | Default (fast) | `--hq` | Notes |
|---|---|---|---|
| anime | realesr-animevideov3 | 2x-AnimeSharpV4-Fast | animevideov3 is trained for anime *video* |
| cartoon | realesr-animevideov3 | 4x-AnimeSharp | |
| series | 4x-ClearRealityV1 (SPAN) | RealESRGAN_x2plus | |
| movie | realesr-general-x4v3, denoise 0.5 | 4x_foolhardy_Remacri | Remacri keeps grain/texture |
| old-movie | realesr-general-x4v3, denoise 0.8 | RealESRGAN_x4plus | heavy noise/compression cleanup |
| abstract | 4x-ClearRealityV1 | 4x-UltraSharpV2-Lite | also `-m 4x-UltraSharpV2` (DAT, very slow) |
| general | 4x-ClearRealityV1 | 4x-UltraSharpV2-Lite | |

Measured on an M6 (model only, fp16/bf16, frames per second):

| Model | 480p in | 720p in | 1080p in |
|---|---|---|---|
| realesr-animevideov3 | 25.7 | 10.8 | |
| 4x-ClearRealityV1 | 21.9 | 9.0 | 4.0 |
| realesr-general-x4v3 | 13.6 | 5.8 | |
| 2x-AnimeSharpV4-Fast | 2.3 | 0.9 | 0.4 |
| RealESRGAN_x4plus_anime_6B | 2.0 | 0.7 | |
| 4x-UltraSharpV2-Lite | 1.6 | 0.6 | |
| RealESRGAN_x4plus / Remacri / 4x-AnimeSharp | 0.6 | 0.2 | |
| 4x-UltraSharpV2 (DAT) | 0.13 | 0.05 | |

End-to-end speed matches model speed (480p anime → 4K ≈ 23 fps, 1080p series → 4K ≈ 3.8 fps), because
decode, inference and encode run in parallel.

**Licences:** Real-ESRGAN models are BSD-3-Clause. The Kim2091 models (AnimeSharp, UltraSharpV2,
ClearReality) and Remacri are **CC BY-NC-SA 4.0** (non-commercial).

## How it works

1. `ffprobe` reads size, fps, sample aspect ratio (anamorphic DVDs are un-squeezed) and interlacing.
2. ffmpeg decodes to RGB, deinterlacing with `bwdif` when needed.
3. The SR model runs on MPS. If one pass can't reach the target (e.g. 240p → 4K) it runs a second
   pass. Leftover scale up to 1.5× is a resize.
4. Oversized model outputs (1080p × 4 = 8K) are box-downscaled on the GPU. Smaller ones are
   lanczos-resized inside the encoder, in parallel with the next GPU batch.
5. ffmpeg encodes 10-bit BT.709 HEVC (VideoToolbox) and muxes the source's audio, subtitles and chapters.

## Why PyTorch MPS instead of MLX?

No video-capable super-resolution models (ESRGAN, SPAN, RCAN, DAT, ...) have MLX ports, while
`spandrel` + PyTorch MPS runs the whole OpenModelDB ecosystem on the Apple GPU today. Diffusion-based
video SR (e.g. SeedVR2) gives better results on very damaged footage but is far too slow on a Mac for
feature-length video.
