"""Command line entry point: ``video-upscaler PATH --type anime --resolution 4k -o OUT``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .models import CONTENT_TYPES, MODELS, MODELS_DIR, download, download_for_type, load_model, model_path
from .upscale import Upscaler, fit_box, parse_resolution, pick_device, pick_dtype, plan_passes, upscale_video
from .video import CODECS, VIDEO_EXTS, probe, require_ffmpeg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-upscaler",
        description="Upscale a video file or a folder of videos to 4K/2K with GAN/transformer SR models "
        "accelerated on the Apple GPU (Metal/MPS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("path", nargs="?", type=Path, help="Video file or folder of videos.")
    p.add_argument(
        "-t", "--type", default="general", choices=sorted(CONTENT_TYPES), help="Content type; picks the model."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (single input) or folder. Default: next to the input with a _<res> suffix; "
        "for a folder input, <folder>_upscaled/.",
    )
    p.add_argument(
        "-r",
        "--resolution",
        default="4k",
        help="Target box: 4k, 2k (1440p), 1080p or WIDTHxHEIGHT. "
        "Aspect ratio is kept, so the output fits inside the box.",
    )
    p.add_argument("--hq", action="store_true", help="Use the type's high-quality model (5-50x slower).")
    p.add_argument(
        "-m",
        "--model",
        help="Override the model: a registry name (see --list-models) or a path to "
        "any spandrel-supported .pth/.safetensors file.",
    )
    p.add_argument(
        "--denoise",
        type=float,
        help="0..1 denoise strength for realesr-general-x4v3 (1 = full denoise, 0 = keep grain). Defaults per type.",
    )
    p.add_argument(
        "-c",
        "--codec",
        default="hevc",
        choices=list(CODECS),
        help="hevc/h264/prores use the VideoToolbox hardware encoder; x265/x264 are CPU encoders.",
    )
    p.add_argument(
        "-q", "--quality", help="Encoder quality (-q:v 1-100 for VideoToolbox, CRF for x264/x265, profile for prores)."
    )
    p.add_argument(
        "--container",
        default="mp4",
        choices=["mp4", "mkv", "mov"],
        help="Container when the output name is derived (mkv keeps every subtitle format).",
    )
    p.add_argument("-b", "--batch", type=int, default=2, help="Frames per GPU batch.")
    p.add_argument(
        "--tile", type=int, default=0, help="Tile size in px for low memory (0 = whole frame, auto-falls back on OOM)."
    )
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--device", help="torch device (default: mps, then cuda, then cpu).")
    p.add_argument(
        "--deinterlace",
        choices=["auto", "on", "off"],
        default="auto",
        help="bwdif deinterlacing; auto = only for streams flagged interlaced.",
    )
    p.add_argument("--start", type=float, help="Start time in seconds (for quick previews).")
    p.add_argument("--duration", type=float, help="Only process this many seconds (for quick previews).")
    p.add_argument("--overwrite", action="store_true", help="Re-process outputs that already exist.")
    p.add_argument("--no-recursive", action="store_true", help="Don't descend into sub-folders.")
    p.add_argument("--list-models", action="store_true", help="Show available models and exit.")
    p.add_argument("--list-types", action="store_true", help="Show content types and their models and exit.")
    p.add_argument(
        "--download", nargs="?", const="all", metavar="TYPE|MODEL|all", help="Download weights into models/ and exit."
    )
    return p


def list_models() -> None:
    print(f"Models directory: {MODELS_DIR}\n")
    for m in MODELS.values():
        have = "✓" if model_path(m.name).exists() else " "
        print(f" [{have}] {m.name:28s} x{m.scale}  {m.arch:22s} {m.speed:6s} {m.license}")
        print(f"       {m.description}")


def list_types() -> None:
    for t in CONTENT_TYPES.values():
        print(f"{t.name:10s} {t.description}")
        print(
            f"{'':10s} default: {t.model}"
            + (f" (denoise={t.denoise})" if t.denoise is not None else "")
            + f"   --hq: {t.hq}"
        )
        if t.alternatives:
            print(f"{'':10s} also good (--model): {', '.join(t.alternatives)}")


def collect_inputs(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Not found: {path}")
    it = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._"))


def output_for(src: Path, root: Path, args: argparse.Namespace, res_tag: str) -> Path:
    ext = ".mov" if args.codec == "prores" else f".{args.container}"
    name = f"{src.stem}_{res_tag}{ext}"
    if root.is_file():
        if args.output is None:
            return src.with_name(name)
        if args.output.suffix.lower() in VIDEO_EXTS:
            return args.output
        return args.output / name
    out_root = args.output or root.with_name(f"{root.name}_upscaled")
    return out_root / src.parent.relative_to(root) / name


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        list_models()
        return 0
    if args.list_types:
        list_types()
        return 0
    if args.download:
        if args.download == "all":
            for n in MODELS:
                download(n)
        elif args.download in CONTENT_TYPES:
            download_for_type(args.download)
        elif args.download in MODELS:
            download(args.download)
        else:
            raise SystemExit(f"Unknown type/model '{args.download}'")
        print(f"Models ready in {MODELS_DIR}")
        return 0
    if args.path is None:
        build_parser().print_help()
        return 1

    require_ffmpeg()
    if args.codec == "prores" and args.output and args.output.suffix.lower() in (".mp4", ".mkv"):
        raise SystemExit("ProRes needs a .mov output.")

    box = parse_resolution(args.resolution)
    res_tag = args.resolution.lower()
    root = args.path.expanduser().resolve()
    inputs = collect_inputs(root, recursive=not args.no_recursive)
    if not inputs:
        raise SystemExit(f"No videos found in {root}")

    device = pick_device(args.device)
    ctype = CONTENT_TYPES[args.type]
    denoise = args.denoise if args.denoise is not None else ctype.denoise
    name = args.model or (ctype.hq if args.hq else ctype.model)
    print(f"Device: {device}  |  type: {args.type}  |  target box: {box[0]}x{box[1]}  |  {len(inputs)} video(s)")

    upscalers: dict[str, Upscaler] = {}
    failures = 0
    for i, src in enumerate(inputs, 1):
        out = output_for(src, root, args, res_tag)
        prefix = f"[{i}/{len(inputs)}] {src.name}"
        if out.exists() and not args.overwrite:
            print(f"{prefix}: exists, skipping ({out})")
            continue
        try:
            info = probe(src)
        except Exception as e:
            print(f"{prefix}: cannot read ({e})", file=sys.stderr)
            failures += 1
            continue

        src_w = info.display_width - info.display_width % 2
        target = fit_box(src_w, info.height, box)
        if target[0] <= src_w and target[1] <= info.height:
            print(f"{prefix}: already {src_w}x{info.height} >= {target[0]}x{target[1]}, skipping")
            continue

        if name not in upscalers:
            model = load_model(name, denoise)
            upscalers[name] = Upscaler(model, device, pick_dtype(model, device, args.precision), tile=args.tile)
        up = upscalers[name]
        passes = plan_passes((src_w, info.height), target, up.scale)
        deint = args.deinterlace == "on" or (args.deinterlace == "auto" and info.interlaced)
        print(
            f"{prefix}: {info.width}x{info.height}@{float(info.fps):.3f} -> {target[0]}x{target[1]}  "
            f"model={name} (x{up.scale} x{passes} pass{'es' if passes > 1 else ''}, {str(up.dtype)[6:]})"
            f"{'  deinterlace' if deint else ''}"
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.partial{out.suffix}")
        t0 = time.time()
        try:
            upscale_video(
                info,
                tmp,
                up,
                target,
                codec=args.codec,
                quality=args.quality,
                batch=args.batch,
                deinterlace=deint,
                start=args.start,
                duration=args.duration,
            )
            tmp.replace(out)
            print(f"  -> {out}  ({time.time() - t0:.0f}s)")
        except KeyboardInterrupt:
            tmp.unlink(missing_ok=True)
            print("\nInterrupted.")
            return 130
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"  FAILED: {e}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
