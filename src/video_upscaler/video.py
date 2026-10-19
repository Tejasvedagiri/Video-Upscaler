"""ffmpeg/ffprobe helpers: probe, raw RGB frame decode and encode pipes."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".m4v",
    ".webm",
    ".wmv",
    ".flv",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".vob",
    ".3gp",
    ".ogv",
}

# Audio codecs that can be stream-copied into mp4/mov without re-encoding.
MP4_AUDIO = {"aac", "mp3", "alac", "ac3", "eac3", "opus", "flac"}
TEXT_SUBS = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}

CODECS = {
    # name: (ffmpeg encoder, pix_fmt, quality flag, default quality, extra args)
    "hevc": ("hevc_videotoolbox", "p010le", "-q:v", "65", ["-tag:v", "hvc1", "-profile:v", "main10"]),
    "h264": ("h264_videotoolbox", "yuv420p", "-q:v", "65", []),
    "x265": ("libx265", "yuv420p10le", "-crf", "18", ["-preset", "slow", "-tag:v", "hvc1"]),
    "x264": ("libx264", "yuv420p", "-crf", "16", ["-preset", "slow"]),
    "prores": ("prores_videotoolbox", "p210le", "-profile:v", "hq", []),
}


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found on PATH. Install with: brew install ffmpeg")


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    frames: int
    interlaced: bool
    sar: Fraction
    audio_codecs: list[str]
    sub_codecs: list[str]

    @property
    def display_width(self) -> int:
        """Width after applying the sample aspect ratio (anamorphic DVDs etc.)."""
        return round(self.width * self.sar)


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = json.loads(out)
    streams = data["streams"]
    v = next(
        (s for s in streams if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")), None
    )
    if v is None:
        raise ValueError(f"No video stream in {path}")

    fps = Fraction(v.get("avg_frame_rate") or "0/1")
    if fps == 0:
        fps = Fraction(v.get("r_frame_rate") or "24/1")
    frames = int(v.get("nb_frames") or 0)
    if not frames:
        duration = float(v.get("duration") or data["format"].get("duration") or 0)
        frames = int(duration * fps)
    sar_s = v.get("sample_aspect_ratio", "1:1")
    sar = Fraction(*map(int, sar_s.split(":"))) if sar_s not in ("0:1", "N/A") else Fraction(1)
    return VideoInfo(
        path=path,
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        frames=frames,
        interlaced=v.get("field_order", "progressive") not in ("progressive", "unknown"),
        sar=sar or Fraction(1),
        audio_codecs=[s["codec_name"] for s in streams if s["codec_type"] == "audio"],
        sub_codecs=[s.get("codec_name", "") for s in streams if s["codec_type"] == "subtitle"],
    )


def _trim_args(start: float | None, duration: float | None) -> list[str]:
    args = []
    if start:
        args += ["-ss", str(start)]
    if duration:
        args += ["-t", str(duration)]
    return args


def open_decoder(
    info: VideoInfo, width: int, height: int, deinterlace: bool, start: float | None, duration: float | None
) -> subprocess.Popen:
    """Decode to packed RGB24 at (width, height) on stdout.

    Deinterlacing and anamorphic correction happen here, before the model sees the frame.
    """
    filters = []
    if deinterlace:
        filters.append("bwdif=mode=send_frame:deint=interlaced")
    if (width, height) != (info.width, info.height):
        filters.append(f"scale={width}:{height}:flags=lanczos")
    filters += ["setsar=1", "format=rgb24"]
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-hwaccel",
        "videotoolbox",
        *_trim_args(start, duration),
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-vf",
        ",".join(filters),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 24)


def open_encoder(
    info: VideoInfo,
    out: Path,
    frame_size: tuple[int, int],
    target: tuple[int, int],
    codec: str,
    quality: str | None,
    start: float | None,
    duration: float | None,
) -> subprocess.Popen:
    """Encode RGB24 frames of ``frame_size`` from stdin, scaled to ``target``.

    Audio, subtitles, chapters and metadata are muxed from the source file.
    """
    encoder, pix_fmt, qflag, qdefault, extra = CODECS[codec]
    container = out.suffix.lower()
    is_mp4 = container in (".mp4", ".m4v", ".mov")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "{}x{}".format(*frame_size),
        "-r",
        str(info.fps),
        "-i",
        "-",
        *_trim_args(start, duration),
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-vf",
        "scale={}:{}:flags=lanczos:out_color_matrix=bt709:out_range=tv".format(*target),
        "-c:v",
        encoder,
        "-pix_fmt",
        pix_fmt,
        qflag,
        quality or qdefault,
        *extra,
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
    ]
    # Audio: copy when the container allows it, otherwise transcode to AAC.
    if not is_mp4 or all(c in MP4_AUDIO for c in info.audio_codecs):
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "320k"]
    # Subtitles: mkv takes anything; mp4 only text subs (converted to mov_text).
    if info.sub_codecs:
        if not is_mp4:
            cmd += ["-map", "1:s?", "-c:s", "copy"]
        elif all(c in TEXT_SUBS for c in info.sub_codecs):
            cmd += ["-map", "1:s?", "-c:s", "mov_text"]
    if is_mp4:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(out))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 24)
