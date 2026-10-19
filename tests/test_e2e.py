"""End-to-end: upscale two short Blender open-movie clips (CC-BY 3.0) to 4K.

Big Buck Bunny (cartoon) and Sintel (anime-style), (c) Blender Foundation, blender.org.
Run with: uv run pytest -m e2e
"""

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest
import torch

from video_upscaler.cli import main

DATA = Path(__file__).parent / "data"
CLIPS = {
    "Big_Buck_Bunny_360_10s_1MB.mp4": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
    "Sintel_360_10s_1MB.mp4": "https://test-videos.co.uk/vids/sintel/mp4/h264/360/Sintel_360_10s_1MB.mp4",
}

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed"),
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU (MPS)"),
]


def _stream(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)["streams"][0]


@pytest.fixture(scope="session")
def clips_dir() -> Path:
    src = DATA / "clips"
    src.mkdir(parents=True, exist_ok=True)
    for name, url in CLIPS.items():
        if not (src / name).exists():
            urllib.request.urlretrieve(url, src / name)
    return src


def test_folder_to_4k(clips_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "out"
    assert main([str(clips_dir), "--type", "anime", "--resolution", "4k", "--output", str(out_dir)]) == 0

    for name in CLIPS:
        src = _stream(clips_dir / name)
        out = out_dir / name.replace(".mp4", "_4k.mp4")
        assert out.exists(), out
        dst = _stream(out)
        assert (dst["width"], dst["height"]) == (3840, 2160)  # 16:9 640x360 -> exact UHD
        assert dst["codec_name"] == "hevc"
        assert dst["r_frame_rate"] == src["r_frame_rate"]
        assert int(dst["nb_read_frames"]) == int(src["nb_read_frames"])
