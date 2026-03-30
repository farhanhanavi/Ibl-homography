"""
Video processing: download, clip, and combine videos.
"""
import os
import subprocess
from pathlib import Path

import cv2
import yt_dlp


def download_video(url: str, job_id: str, output_dir: str) -> dict:
    """Download a YouTube video and extract its first frame as a thumbnail."""
    output_dir = Path(output_dir)
    video_path = str(output_dir / f"tmp_{job_id}.mp4")

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": video_path,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # Read video metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open downloaded video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Save first frame as thumbnail
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame from video")

    thumb_path = str(output_dir / f"thumb_{job_id}.jpg")
    cv2.imwrite(thumb_path, frame)

    return {
        "video_path": video_path,
        "thumb_path": thumb_path,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
    }


def clip_video(video_path: str, job_id: str, output_dir: str,
               start_sec: float, end_sec: float) -> dict:
    """Clip a video to the specified time range using ffmpeg."""
    output_dir = Path(output_dir)
    clip_path = str(output_dir / f"clip_{job_id}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", video_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        clip_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip failed: {result.stderr[-500:]}")

    # Measure actual duration
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frames / fps

    return {"clip_path": clip_path, "duration": duration}


def combine_videos(clip_path: str, court_anim_path: str,
                   output_dir: str, job_id: str) -> dict:
    """Stack original clip (top) and court animation (bottom) vertically."""
    output_dir = Path(output_dir)
    output_path = str(output_dir / f"output_{job_id}.mp4")

    # Get clip dimensions to scale court animation to same width
    cap = cv2.VideoCapture(clip_path)
    clip_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    cap2 = cv2.VideoCapture(court_anim_path)
    anim_w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap2.release()

    # If widths differ, scale court animation to match clip width
    if anim_w != clip_w:
        scale_filter = f"[1:v]scale={clip_w}:-2[court];"
        vstack_filter = f"{scale_filter}[0:v][court]vstack=inputs=2[out]"
    else:
        vstack_filter = "[0:v][1:v]vstack=inputs=2[out]"

    cmd = [
        "ffmpeg", "-y",
        "-i", clip_path,
        "-i", court_anim_path,
        "-filter_complex", vstack_filter,
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-r", str(clip_fps),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg combine failed: {result.stderr[-500:]}")

    return {"output_path": output_path}
