"""
Court Vision - Basketball Analysis App
FastAPI backend
"""
import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from video_processor import download_video, clip_video, combine_videos
from homography import compute_homography, process_video_with_homography

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Court Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

executor = ThreadPoolExecutor(max_workers=2)

# In-memory job store
jobs: dict = {}


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class FetchVideoRequest(BaseModel):
    url: str

class ClipRequest(BaseModel):
    job_id: str
    start: str   # "MM:SS" or "HH:MM:SS"
    end: str

class CourtPointsRequest(BaseModel):
    job_id: str
    # List of {x, y} pixel coords on the displayed thumbnail
    points: list[dict]
    # Displayed image dimensions so we can scale to original resolution
    display_width: int
    display_height: int

class ProcessRequest(BaseModel):
    job_id: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def parse_timestamp(ts: str) -> float:
    """Convert MM:SS or HH:MM:SS or bare seconds to float seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    raise ValueError(f"Cannot parse timestamp: {ts!r}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/fetch-video")
async def fetch_video(req: FetchVideoRequest):
    """Download YouTube video and return first frame + metadata."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "downloading",
        "progress": 0,
        "video_path": None,
        "clip_path": None,
        "thumb_path": None,
        "homography_matrix": None,
        "output_path": None,
        "orig_width": None,
        "orig_height": None,
        "fps": None,
        "error": None,
    }

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: download_video(req.url, job_id, str(STATIC_DIR))
        )
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=400, detail=str(e))

    jobs[job_id].update({
        "status": "downloaded",
        "video_path": result["video_path"],
        "thumb_path": result["thumb_path"],
        "orig_width": result["width"],
        "orig_height": result["height"],
        "fps": result["fps"],
        "duration": result["duration"],
    })

    return {
        "job_id": job_id,
        "thumb_url": f"/static/{Path(result['thumb_path']).name}",
        "width": result["width"],
        "height": result["height"],
        "fps": result["fps"],
        "duration": result["duration"],
    }


@app.post("/api/clip")
async def clip(req: ClipRequest):
    """Clip the downloaded video to the specified timestamps."""
    job = jobs.get(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.get("video_path"):
        raise HTTPException(status_code=400, detail="Video not yet downloaded")

    try:
        start_sec = parse_timestamp(req.start)
        end_sec = parse_timestamp(req.end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if end_sec <= start_sec:
        raise HTTPException(status_code=400, detail="End time must be after start time")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: clip_video(job["video_path"], req.job_id, str(STATIC_DIR), start_sec, end_sec)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    job["clip_path"] = result["clip_path"]
    job["clip_duration"] = result["duration"]
    job["status"] = "clipped"

    return {
        "job_id": req.job_id,
        "clip_url": f"/static/{Path(result['clip_path']).name}",
        "duration": result["duration"],
    }


@app.post("/api/set-court-points")
async def set_court_points(req: CourtPointsRequest):
    """Accept clicked court points and compute homography matrix."""
    job = jobs.get(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if len(req.points) < 4:
        raise HTTPException(status_code=400, detail="At least 4 court points required")

    orig_w = job["orig_width"]
    orig_h = job["orig_height"]

    # Scale points from display size back to original resolution
    scale_x = orig_w / req.display_width
    scale_y = orig_h / req.display_height
    scaled_points = [
        {"x": p["x"] * scale_x, "y": p["y"] * scale_y, "label": p.get("label", "")}
        for p in req.points
    ]

    try:
        matrix = compute_homography(scaled_points)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Homography failed: {e}")

    job["homography_matrix"] = matrix.tolist()
    job["court_points"] = scaled_points
    job["status"] = "ready"

    return {"job_id": req.job_id, "status": "ready"}


@app.post("/api/process")
async def process(req: ProcessRequest):
    """Start background processing: player tracking + output video render."""
    job = jobs.get(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.get("clip_path"):
        raise HTTPException(status_code=400, detail="Video not yet clipped")
    if job.get("homography_matrix") is None:
        raise HTTPException(status_code=400, detail="Court points not yet set")

    job["status"] = "processing"
    job["progress"] = 0

    asyncio.create_task(_run_processing(req.job_id))

    return {"job_id": req.job_id, "status": "processing"}


async def _run_processing(job_id: str):
    job = jobs[job_id]
    loop = asyncio.get_event_loop()
    try:
        import numpy as np
        matrix = np.array(job["homography_matrix"], dtype=np.float64)

        def _track():
            return process_video_with_homography(
                clip_path=job["clip_path"],
                homography_matrix=matrix,
                output_dir=str(STATIC_DIR),
                job_id=job_id,
                progress_callback=lambda p: jobs[job_id].update({"progress": p}),
            )

        result = await loop.run_in_executor(executor, _track)

        # Combine original clip + court animation
        def _combine():
            return combine_videos(
                clip_path=job["clip_path"],
                court_anim_path=result["court_anim_path"],
                output_dir=str(STATIC_DIR),
                job_id=job_id,
            )

        combine_result = await loop.run_in_executor(executor, _combine)

        job["output_path"] = combine_result["output_path"]
        job["status"] = "done"
        job["progress"] = 100

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        import traceback
        traceback.print_exc()


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    resp = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "error": job.get("error"),
    }
    if job["status"] == "done" and job.get("output_path"):
        resp["output_url"] = f"/static/{Path(job['output_path']).name}"
    return resp


@app.get("/download/{filename}")
async def download(filename: str):
    """Force-download a processed video."""
    path = STATIC_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        str(path),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    frontend = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(str(frontend))
