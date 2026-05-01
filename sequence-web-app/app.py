from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request, Response, send_file
from PIL import Image

APP_NAME = "Sequence Video & Audio Tool"

# ------------------------------------------------------------
# Dirs
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "sequence_uploads"
RENDER_DIR = Path(tempfile.gettempdir()) / "sequence_renders"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RENDER_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB max upload

# ------------------------------------------------------------
# Render state
# ------------------------------------------------------------
STOP_EVENT = threading.Event()
CURRENT_PROC_LOCK = threading.Lock()
CURRENT_PROC: Optional[subprocess.Popen] = None
RENDER_SESSIONS: Dict[str, Dict] = {}  # session_id -> {status, files, ...}
# Job tracking uses files on disk so all gunicorn workers can read/write
JOBS_DIR = Path(tempfile.gettempdir()) / "sequence_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"

def _read_job(job_id: str) -> Optional[Dict]:
    p = _job_path(job_id)
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except: return None

def _write_job(job_id: str, job: Dict) -> None:
    _job_path(job_id).write_text(json.dumps(job))

def set_current_proc(p: Optional[subprocess.Popen]) -> None:
    global CURRENT_PROC
    with CURRENT_PROC_LOCK:
        CURRENT_PROC = p

def stop_current_proc() -> None:
    with CURRENT_PROC_LOCK:
        p = CURRENT_PROC
    if p and p.poll() is None:
        try:
            p.terminate()
            try: p.wait(timeout=2)
            except subprocess.TimeoutExpired: p.kill()
        except Exception: pass

# ------------------------------------------------------------
# FFmpeg
# ------------------------------------------------------------
def _can_exec(path: str) -> bool:
    try:
        r = subprocess.run([path, "-version"], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False

def find_tool(name: str) -> str:
    for candidate in [
        Path("/opt/homebrew/bin") / name,
        Path("/usr/local/bin") / name,
        Path("/usr/bin") / name,
    ]:
        if candidate.exists() and _can_exec(str(candidate)):
            return str(candidate)
    found = shutil.which(name)
    if found and _can_exec(found):
        return found
    return name

FFMPEG = find_tool("ffmpeg")
FFPROBE = find_tool("ffprobe")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
VISUAL_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".mp4"}
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".mp3"}
Image.MAX_IMAGE_PIXELS = None

def is_16_9(w: int, h: int) -> bool:
    return abs((w / h) - (16 / 9)) <= 0.02

def _safe_float(v: Any, default: float = 0.0) -> float:
    try: return float(v)
    except: return default

def _next_available(path: Path) -> Path:
    if not path.exists(): return path
    stem, suf = path.stem, path.suffix
    for i in range(2, 5000):
        p = path.with_name(f"{stem}_{i}{suf}")
        if not p.exists(): return p
    return path.with_name(f"{stem}_{next(tempfile._get_candidate_names())}{suf}")

def ffprobe_wh(path: Path) -> Tuple[int, int]:
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "").strip()
    if "x" not in out:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    w, h = out.split("x", 1)
    return int(w), int(h)

def audio_duration(path: Path) -> float:
    cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return float((r.stdout or "0").strip() or 0)

def get_session_dir(session_id: str) -> Path:
    d = UPLOAD_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def cleanup_session(session_id: str) -> None:
    """Remove uploaded files for a session to free disk space."""
    d = UPLOAD_DIR / session_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    rd = RENDER_DIR / session_id
    if rd.exists():
        shutil.rmtree(rd, ignore_errors=True)
    RENDER_SESSIONS.pop(session_id, None)

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "ffmpeg": _can_exec(FFMPEG),
        "ffprobe": _can_exec(FFPROBE),
    })

# ------------------------------------------------------------
# Upload visual
# ------------------------------------------------------------
@app.route("/upload_visual", methods=["POST"])
def upload_visual():
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    f = request.files.get("visual")
    if not f or not f.filename:
        return jsonify({"error": "No file provided"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in VISUAL_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    sess_dir = get_session_dir(session_id)
    # Clear any existing visual
    for old in sess_dir.glob("visual_*"):
        old.unlink(missing_ok=True)

    save_path = sess_dir / f"visual_{f.filename}"
    f.save(str(save_path))

    # Get dimensions
    try:
        if ext in {".jpg", ".jpeg", ".png"}:
            with Image.open(save_path) as img:
                w, h = img.size
        else:
            w, h = ffprobe_wh(save_path)
    except Exception as e:
        return jsonify({"error": f"Could not read visual: {e}"}), 500

    v_is_169 = is_16_9(w, h)

    # Auto-rescale large square images
    is_square = (w == h)
    rescaled = False
    if ext in {".jpg", ".jpeg", ".png"} and is_square and w >= 4000:
        new_path = sess_dir / f"visual_{Path(f.filename).stem}_2k{ext}"
        with Image.open(save_path) as img:
            img.resize((2000, 2000), Image.Resampling.LANCZOS).save(new_path, quality=95)
        save_path = new_path
        w, h = 2000, 2000
        rescaled = True

    return jsonify({
        "session_id": session_id,
        "filename": save_path.name,
        "width": w,
        "height": h,
        "is_16_9": v_is_169,
        "rescaled": rescaled,
    })

# ------------------------------------------------------------
# Upload audio folder (multiple files)
# ------------------------------------------------------------
@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    files = request.files.getlist("audio")
    if not files:
        return jsonify({"error": "No audio files provided"}), 400

    sess_dir = get_session_dir(session_id)
    audio_dir = sess_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    saved = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            continue
        bare_name = Path(f.filename).name
        audio_dir.mkdir(parents=True, exist_ok=True)
        save_path = audio_dir / bare_name
        f.save(str(save_path))
        saved.append(f.filename)

    if not saved:
        return jsonify({"error": "No supported audio files found"}), 400

    return jsonify({
        "session_id": session_id,
        "files": sorted(saved),
        "count": len(saved),
    })

# ------------------------------------------------------------
# Audio stream (for waveform preview)
# ------------------------------------------------------------
@app.route("/audio_stream")
def audio_stream():
    session_id = request.args.get("session_id", "")
    name = request.args.get("name", "")
    if not session_id or not name:
        return Response("Missing params", status=400)
    p = get_session_dir(session_id) / "audio" / name
    if not p.exists():
        return Response("Not found", status=404)
    return send_file(str(p), conditional=True)

# ------------------------------------------------------------
# Inspect (after uploads)
# ------------------------------------------------------------
@app.route("/inspect", methods=["POST"])
def inspect():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "")
    audio_only = bool(data.get("audio_only", False))

    sess_dir = get_session_dir(session_id)
    audio_dir = sess_dir / "audio"

    audio_files = sorted([
        f.name for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    ]) if audio_dir.exists() else []

    result = {
        "audio": {"count": len(audio_files), "files": audio_files},
        "ffmpeg_ok": _can_exec(FFMPEG),
        "ffprobe_ok": _can_exec(FFPROBE),
        "audio_only": audio_only,
    }

    if audio_only:
        result["visual"] = {"selected_name": None, "is_16_9": False, "width": None, "height": None}
        return jsonify(result)

    # Find visual
    visuals = [f for f in sess_dir.iterdir()
               if f.name.startswith("visual_") and f.suffix.lower() in VISUAL_EXTS]
    if not visuals:
        return Response("No visual uploaded yet.", status=400)

    vis = visuals[0]
    ext = vis.suffix.lower()
    try:
        if ext in {".jpg", ".jpeg", ".png"}:
            with Image.open(vis) as img:
                w, h = img.size
        else:
            w, h = ffprobe_wh(vis)
    except Exception as e:
        return Response(f"Could not read visual: {e}", status=500)

    result["visual"] = {
        "selected_name": vis.name,
        "width": w, "height": h,
        "is_16_9": is_16_9(w, h),
    }
    return jsonify(result)

# ------------------------------------------------------------
# Stop
# ------------------------------------------------------------
@app.route("/stop", methods=["POST"])
def stop():
    STOP_EVENT.set()
    stop_current_proc()
    return Response("Stop requested.", status=200)

# ------------------------------------------------------------
# Render
# ------------------------------------------------------------
# Resolution targets: (landscape_w, landscape_h, square_side)
RESOLUTIONS = {
    "4k":    (3840, 2160, 4000),
    "2k":    (2560, 1440, 2000),
    "1080p": (1920, 1080, 1080),
    "720p":  (1280,  720,  720),
    "480p":  ( 854,  480,  480),
    "original": None,  # no scaling
}

def build_scale_filter(w: int, h: int, res_key: str) -> str:
    """
    Build an ffmpeg scale filter that:
    - Preserves the original aspect ratio exactly
    - Scales DOWN to fit within the chosen resolution (never upscales)
    - Works for any shape: landscape, square, portrait, or unusual
    - Ensures width and height are divisible by 2 (required by libx264)
    """
    res = RESOLUTIONS.get(res_key, RESOLUTIONS["1080p"])
    wide_w, wide_h, square_side = res

    is_square = (w == h)
    is_portrait = (h > w)

    if is_square:
        target = square_side
        # Never upscale — if image is already smaller, keep original size
        target = min(target, w)
        # Round down to even
        target = target - (target % 2)
        return f"scale={target}:{target}"
    elif is_portrait:
        # Portrait: longest edge is height
        target_h = min(wide_w, h)  # never upscale
        target_h = target_h - (target_h % 2)
        return f"scale=-2:{target_h}"
    else:
        # Landscape
        target_w = min(wide_w, w)  # never upscale
        target_w = target_w - (target_w % 2)
        return f"scale={target_w}:-2"

def _run_render_job(job_id: str, data: dict) -> None:
    job = _read_job(job_id)
    session_id = data.get("session_id", "")
    audio_only = bool(data.get("audio_only", False))
    export_mp3 = bool(data.get("export_mp3", False))
    sess_dir = get_session_dir(session_id)
    audio_dir = sess_dir / "audio"
    render_out = RENDER_DIR / session_id
    render_out.mkdir(parents=True, exist_ok=True)

    try:
        audio_files = sorted([
            f for f in audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        ]) if audio_dir.exists() else []

        if not audio_files:
            job["status"] = "error"; job["error"] = "No audio files found."; return
            _write_job(job_id, job)

        # ---- AUDIO ONLY ----
        if audio_only:
            clips = data.get("clips") or []
            clips_dir = render_out / "CLIPS"
            clips_dir.mkdir(exist_ok=True)
            job["total"] = len(clips)
            _write_job(job_id, job)
            created = 0
            for idx, clip in enumerate(clips, start=1):
                if STOP_EVENT.is_set(): break
                name = (clip.get("selected_audio") or "").strip()
                match = [a for a in audio_files if a.name == name]
                if not match: continue
                audio = match[0]
                seg_len = _safe_float(clip.get("segment_length_sec"), 30.0)
                seg_start = _safe_float(clip.get("segment_start_sec"), 0.0)
                fade = bool(clip.get("fade", False))
                out_ext = ".mp3" if export_mp3 else ".wav"
                out_path = _next_available(clips_dir / f"{audio.stem}_clip_{idx}{out_ext}")
                job["videos"].append({"name": audio.name, "status": "rendering", "url": None})
                _write_job(job_id, job)
                v_idx = len(job["videos"]) - 1
                _write_job(job_id, job)
                cmd = [FFMPEG, "-y", "-ss", str(seg_start), "-t", str(seg_len), "-i", str(audio)]
                afilters = []
                if fade:
                    f_in = _safe_float(clip.get("fade_in"), 0.0)
                    f_out = _safe_float(clip.get("fade_out"), 0.0)
                    afilters += [f"afade=t=in:st=0:d={f_in}:curve=tri",
                                 f"afade=t=out:st={max(seg_len-f_out,0.0)}:d={f_out}:curve=tri"]
                if afilters: cmd += ["-af", ",".join(afilters)]
                if export_mp3: cmd += ["-c:a", "libmp3lame", "-b:a", "320k", str(out_path)]
                else: cmd += ["-c:a", "pcm_s16le", str(out_path)]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                set_current_proc(p); p.communicate(); set_current_proc(None)
                if p.returncode == 0:
                    created += 1
                    job["videos"][v_idx]["status"] = "done"
                    _write_job(job_id, job)
                    job["videos"][v_idx]["url"] = f"/download/{session_id}/CLIPS/{out_path.name}"
                    _write_job(job_id, job)
                else:
                    job["videos"][v_idx]["status"] = "error"
                    _write_job(job_id, job)
                job["done"] = created
                _write_job(job_id, job)
            if created > 0:
                zip_path = render_out / "clips.zip"
                shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(clips_dir))
                job["status"] = "done"
                _write_job(job_id, job)
                job["zip_url"] = f"/download/{session_id}/clips.zip"
                _write_job(job_id, job)
                job["message"] = f"Created {created} clip(s)."
                _write_job(job_id, job)
            else:
                job["status"] = "error"; job["error"] = "No clips were created."
                _write_job(job_id, job)
            return

        # ---- VIDEO MODE ----
        visuals = [f for f in sess_dir.iterdir()
                   if f.name.startswith("visual_") and f.suffix.lower() in VISUAL_EXTS]
        if not visuals:
            job["status"] = "error"; job["error"] = "No visual found."; return
            _write_job(job_id, job)

        visual_path = visuals[0]
        ext = visual_path.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            with Image.open(visual_path) as img:
                w, h = img.size
        else:
            w, h = ffprobe_wh(visual_path)

        visual_is_169 = is_16_9(w, h)
        selected_audio_name = (data.get("selected_audio") or "").strip()
        sixteen_nine_mode = (data.get("sixteen_nine_mode") or "all").strip().lower()
        mode = (data.get("mode") or "full").strip().lower()
        use_segment = (mode == "segment")
        fade = bool(data.get("fade", False))
        seg_len = _safe_float(data.get("segment_length_sec"), 0.0)
        seg_start = _safe_float(data.get("segment_start_sec"), 0.0)
        res_key = (data.get("resolution") or "1080p").strip().lower()
        if res_key not in RESOLUTIONS: res_key = "1080p"

        # For 16:9 classic mode — pre-scale to 1080p ONCE, then reuse for all renders
        # This is much faster than applying the scale filter per-frame during encoding
        if visual_is_169 and sixteen_nine_mode == "all":
            scaled_path = sess_dir / f"{visual_path.stem}_1080p{visual_path.suffix}"
            if not scaled_path.exists():
                job["message"] = "Pre-scaling image to 1080p..."
                _write_job(job_id, job)
                scale_cmd = [
                    FFMPEG, "-y", "-i", str(visual_path),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
                    "-an", str(scaled_path)
                ]
                subprocess.run(scale_cmd, capture_output=True)
            visual_path = scaled_path
            scale_filter = None  # already scaled, no filter needed
        elif res_key == "original":
            scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        else:
            scale_filter = build_scale_filter(w, h, res_key)

        separate_169 = (visual_is_169 and sixteen_nine_mode == "separate")
        if (not visual_is_169) or separate_169:
            match = [a for a in audio_files if a.name == selected_audio_name]
            if not match:
                job["status"] = "error"; job["error"] = "Please choose a valid audio file."; return
                _write_job(job_id, job)
            render_audio_files = match
        else:
            render_audio_files = audio_files
            use_segment = False
            fade = False

        job["total"] = len(render_audio_files)
        _write_job(job_id, job)
        created = 0

        for audio in render_audio_files:
            if STOP_EVENT.is_set(): break
            output_video = render_out / f"{audio.stem}.mp4"
            job["videos"].append({"name": audio.stem + ".mp4", "status": "rendering", "url": None})
            _write_job(job_id, job)
            v_idx = len(job["videos"]) - 1
            _write_job(job_id, job)

            if visual_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                input_v = ["-loop", "1", "-i", str(visual_path)]
            else:
                input_v = ["-stream_loop", "-1", "-i", str(visual_path)]

            cmd = [FFMPEG, "-y", *input_v]
            if use_segment:
                cmd += ["-ss", str(max(seg_start, 0.0))]
                if seg_len > 0: cmd += ["-t", str(seg_len)]
            cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0"]
            if scale_filter:
                cmd += ["-vf", scale_filter]
            if fade:
                dur = seg_len if (use_segment and seg_len > 0) else audio_duration(audio)
                f_in = _safe_float(data.get("fade_in"), 0.0)
                f_out = _safe_float(data.get("fade_out"), 0.0)
                cmd += ["-af", f"afade=t=in:st=0:d={f_in}:curve=tri,afade=t=out:st={max(dur-f_out,0.0)}:d={f_out}:curve=tri"]
            cmd += [
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-r", "24", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "320k",
                "-movflags", "+faststart", "-shortest",
                str(output_video)
            ]

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            set_current_proc(p)
            _, err = p.communicate()
            set_current_proc(None)

            if p.returncode == 0:
                created += 1
                job["videos"][v_idx]["status"] = "done"
                _write_job(job_id, job)
                job["videos"][v_idx]["url"] = f"/download/{session_id}/{output_video.name}"
                _write_job(job_id, job)
            else:
                job["videos"][v_idx]["status"] = "error"
                _write_job(job_id, job)
                job["videos"][v_idx]["error"] = err[:200]
                _write_job(job_id, job)
            job["done"] = created
            _write_job(job_id, job)

        if created > 1:
            zip_path = render_out / "videos.zip"
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(render_out))
            job["zip_url"] = f"/download/{session_id}/videos.zip"
            _write_job(job_id, job)

        job["status"] = "done"
        _write_job(job_id, job)
        job["message"] = f"Done. Created {created} video(s)."
        _write_job(job_id, job)

    except Exception as e:
        job["status"] = "error"; job["error"] = str(e)


@app.route("/render", methods=["POST"])
def render():
    STOP_EVENT.clear()
    data = request.get_json(force=True) or {}
    job_id = str(uuid.uuid4())
    _write_job(job_id, {
        "status": "running", "total": 0, "done": 0,
        "videos": [], "message": "", "error": "", "zip_url": None,
    })
    threading.Thread(target=_run_render_job, args=(job_id, data), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/render_progress/<job_id>")
def render_progress(job_id: str):
    job = _read_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)
@app.route("/merge_clips", methods=["POST"])
def merge_clips():
    STOP_EVENT.clear()
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "")
    clips = data.get("clips") or []

    sess_dir = get_session_dir(session_id)
    audio_dir = sess_dir / "audio"
    render_out = RENDER_DIR / session_id / "CLIPS"
    render_out.mkdir(parents=True, exist_ok=True)

    audio_files = sorted([f for f in audio_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in AUDIO_EXTS])

    tmp_dir = Path(tempfile.mkdtemp())
    processed = []

    try:
        for i, clip in enumerate(clips):
            if STOP_EVENT.is_set(): break
            name = (clip.get("selected_audio") or "").strip()
            match = [a for a in audio_files if a.name == name]
            if not match: continue
            audio = match[0]
            seg_len = _safe_float(clip.get("segment_length_sec"), 30.0)
            seg_start = _safe_float(clip.get("segment_start_sec"), 0.0)
            f_in = 8.0 if i == 0 else 5.0
            f_out = 8.0 if i == len(clips) - 1 else 5.0

            raw_clip = tmp_dir / f"clip_{i}.wav"
            cmd = [FFMPEG, "-y", "-ss", str(seg_start), "-t", str(seg_len), "-i", str(audio),
                   "-af", f"afade=t=in:st=0:d={f_in}:curve=tri,afade=t=out:st={max(seg_len-f_out,0.0)}:d={f_out}:curve=tri",
                   "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(raw_clip)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            set_current_proc(p)
            p.communicate()
            set_current_proc(None)
            if p.returncode == 0: processed.append(raw_clip)

        if not processed:
            return Response("No clips processed.", status=400)

        out_ext = ".mp3" if bool(data.get("export_mp3", False)) else ".wav"
        out_path = _next_available(render_out / f"MERGED_AUDIO{out_ext}")
        concat_txt = tmp_dir / "concat.txt"
        concat_txt.write_text("\n".join([f"file '{str(p.resolve())}'" for p in processed]))

        final_cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt)]
        if out_ext == ".mp3": final_cmd += ["-c:a", "libmp3lame", "-b:a", "320k", str(out_path)]
        else: final_cmd += ["-c:a", "pcm_s16le", str(out_path)]

        p_final = subprocess.Popen(final_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        set_current_proc(p_final)
        p_final.communicate()
        set_current_proc(None)

        if p_final.returncode == 0:
            return jsonify({"ok": True, "message": f"Merge complete: {out_path.name}",
                           "downloads": [{"filename": out_path.name,
                                         "url": f"/download/{session_id}/CLIPS/{out_path.name}"}]})
        return Response("Merge failed.", status=500)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ------------------------------------------------------------
# Socials Animations
# ------------------------------------------------------------
SOCIALS_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SOCIALS_W, SOCIALS_H = 1080, 1920
SOCIALS_FG_W = int(SOCIALS_W * 0.68)  # 734
SOCIALS_FG_W -= SOCIALS_FG_W % 2
SOCIALS_PAD = 40  # shadow padding around the square
SOCIALS_BG_OVERSIZE = 1.4  # for pan animations: 1.4x of frame -> 2688 wide
SOCIALS_BG_PAN_W = int(SOCIALS_W * SOCIALS_BG_OVERSIZE)
SOCIALS_BG_PAN_W -= SOCIALS_BG_PAN_W % 2  # 1512
SOCIALS_BG_PAN_FILL = int(SOCIALS_H * SOCIALS_BG_OVERSIZE)
SOCIALS_BG_PAN_FILL -= SOCIALS_BG_PAN_FILL % 2  # 2688


def _socials_session_dir(session_id: str) -> Path:
    d = UPLOAD_DIR / session_id / "socials"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _socials_find_image(session_id: str) -> Optional[Path]:
    d = _socials_session_dir(session_id)
    for f in d.iterdir():
        if f.is_file() and f.name.startswith("image_") and f.suffix.lower() in SOCIALS_IMAGE_EXTS:
            return f
    return None


def _socials_find_audio(session_id: str) -> Optional[Path]:
    d = _socials_session_dir(session_id)
    for f in d.iterdir():
        if f.is_file() and f.name.startswith("audio_") and f.suffix.lower() in AUDIO_EXTS:
            return f
    return None


@app.route("/upload_socials_image", methods=["POST"])
def upload_socials_image():
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify({"error": "No file provided"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in SOCIALS_IMAGE_EXTS:
        return jsonify({"error": f"Unsupported image type: {ext}"}), 400

    d = _socials_session_dir(session_id)
    # Clear any existing image
    for old in d.glob("image_*"):
        old.unlink(missing_ok=True)

    save_path = d / f"image_{Path(f.filename).name}"
    f.save(str(save_path))

    try:
        with Image.open(save_path) as img:
            w, h = img.size
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 500

    # Auto-rescale large square images to keep ffmpeg fast
    if w == h and w >= 4000:
        new_path = d / f"image_{Path(f.filename).stem}_2k{ext}"
        with Image.open(save_path) as img:
            img.resize((2000, 2000), Image.Resampling.LANCZOS).save(new_path, quality=95)
        save_path.unlink(missing_ok=True)
        save_path = new_path
        w, h = 2000, 2000

    return jsonify({
        "session_id": session_id,
        "filename": save_path.name,
        "width": w, "height": h,
        "is_square": (w == h),
    })


@app.route("/upload_socials_audio", methods=["POST"])
def upload_socials_audio():
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify({"error": "No audio file provided"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in AUDIO_EXTS:
        return jsonify({"error": f"Unsupported audio type: {ext}"}), 400

    d = _socials_session_dir(session_id)
    for old in d.glob("audio_*"):
        old.unlink(missing_ok=True)

    save_path = d / f"audio_{Path(f.filename).name}"
    f.save(str(save_path))

    try:
        dur = audio_duration(save_path)
    except Exception:
        dur = 0.0

    return jsonify({
        "session_id": session_id,
        "filename": save_path.name,
        "duration": dur,
    })


def _socials_overlay_chain(overlays: list, in_label: str, out_label: str, duration: float) -> str:
    """
    Build filter segments applying background overlays in sequence.
    [in_label] → … → [out_label], all at 1080×1920.
    Returns a semicolon-joined string (no leading/trailing semicolons).
    """
    if not overlays:
        return f"[{in_label}]copy[{out_label}]"

    bw, bh = SOCIALS_W, SOCIALS_H
    parts: List[str] = []
    cur = in_label

    for i, ov in enumerate(overlays):
        nxt = out_label if i == len(overlays) - 1 else f"ov{i}"
        t = (ov.get("type") or "").lower()

        if t == "blur":
            n = max(5, min(100, int(_safe_float(ov.get("intensity"), 40))))
            sigma = round(n * 0.25, 1)   # 5→1.25, 100→25
            parts.append(f"[{cur}]gblur=sigma={sigma}[{nxt}]")

        elif t == "grain":
            n = max(5, min(100, int(_safe_float(ov.get("intensity"), 35))))
            parts.append(f"[{cur}]noise=alls={int(n*0.6)}:allf=t+u[{nxt}]")

        elif t == "glitch":
            n = max(5, min(100, int(_safe_float(ov.get("intensity"), 50))))
            shift = max(1, int(round(n * 0.30)))
            parts.append(f"[{cur}]format=rgba,rgbashift=rh={shift}:rv=0:bh=-{shift}:bv=0[{nxt}]")

        elif t == "vignette":
            n = max(10, min(100, int(_safe_float(ov.get("strength"), 60))))
            angle = 3.1416 / 5.0 + (3.1416 / 2.5 - 3.1416 / 5.0) * (n / 100.0)
            parts.append(f"[{cur}]vignette={angle:.4f}[{nxt}]")

        elif t == "scanlines":
            n = max(10, min(100, int(_safe_float(ov.get("intensity"), 50))))
            mult = round(1.0 - n / 200.0, 3)
            parts.append(
                f"[{cur}]geq="
                f"lum='lum(X\\,Y)*if(mod(Y\\,3)\\,1\\,{mult})'"
                f":cb='cb(X\\,Y)':cr='cr(X\\,Y)'[{nxt}]"
            )

        else:
            parts.append(f"[{cur}]copy[{nxt}]")

        cur = nxt

    return ";".join(parts)


def _socials_build_filter(
    bg_treatment: str,
    animation: str,
    strobe_op: float,
    pan_dur: float,
    duration: float,
    strobe_zoom: float = 1.0,
    strobe_offset_x: float = 0.0,
    strobe_offset_y: float = 0.0,
    overlays: Optional[list] = None,
) -> str:
    """
    Build the ffmpeg filter_complex string for the socials render.
    Single image input on [0:v]. Output label: [outv].

    For strobe mode, the strobe overlay is composited BEHIND the foreground square
    (over the background only). strobe_zoom (>=1.0) and strobe_offset_x/y (-1..1)
    let the user reframe the background; the constraint that bg height >= frame
    height is enforced by clamping zoom to >= 1.0.
    """
    bg_w, bg_h = SOCIALS_W, SOCIALS_H
    fg_w = SOCIALS_FG_W
    pad = SOCIALS_PAD
    shadow_outer = fg_w + pad * 2  # 814 for fg=734

    # 1) Background: scale to fill 9:16 (cover) then crop. For pan, oversize to 1.4x.
    is_pan = animation in ("horizontal", "diagonal", "diamond")
    is_strobe = (animation == "strobe")

    if is_pan:
        # Oversized square fill so we have headroom to pan in any direction
        bg_chain = (
            f"[0:v]scale={SOCIALS_BG_PAN_FILL}:{SOCIALS_BG_PAN_FILL}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={SOCIALS_BG_PAN_FILL}:{SOCIALS_BG_PAN_FILL}"
        )
    elif is_strobe:
        # Strobe mode: build a bg sized to cover 9:16 at the user's zoom factor,
        # then crop a 1080x1920 window with offset. Zoom is clamped >= 1.0 so
        # height is never less than the frame height.
        z = max(1.0, min(3.0, strobe_zoom))
        # Scale so the SHORTER edge of the bg equals (frame_short * z); the longer
        # edge will be larger. Using force_original_aspect_ratio=increase against
        # bg_w x bg_h scaled by z gives exactly that.
        sw = int(round(bg_w * z));  sw -= sw % 2
        sh = int(round(bg_h * z));  sh -= sh % 2
        bg_chain = (
            f"[0:v]scale={sw}:{sh}:force_original_aspect_ratio=increase"
        )
    else:
        bg_chain = (
            f"[0:v]scale={bg_w}:{bg_h}:force_original_aspect_ratio=increase,"
            f"crop={bg_w}:{bg_h}"
        )

    # Background treatment
    if bg_treatment == "inverted":
        bg_chain += ",negate"
    elif bg_treatment == "bw":
        bg_chain += ",hue=s=0"

    # For strobe with zoom/offset, append the offset crop now (after treatment).
    if is_strobe:
        # Slack between scaled bg and the 1080x1920 window. We can't know the
        # post-aspect-ratio scaled size without ffprobe; use ffmpeg expressions.
        # iw/ih here refer to the chain INPUT, which is the original [0:v].
        # After scale=force_original_aspect_ratio=increase, the actual pixel
        # dimensions depend on input aspect. Use `ow`/`oh` would be wrong inside
        # crop since they reference the crop output. Easiest: do scale-with-pad
        # to a known size first.
        #
        # Simpler approach: scale square-ish input to a known oversize box, then
        # crop with explicit slack. We'll use a target size of (bg_w*z, bg_h*z)
        # using force_original_aspect_ratio=increase, then crop bg_w:bg_h with
        # an offset based on (in_w-bg_w) and (in_h-bg_h).
        ox = max(-1.0, min(1.0, strobe_offset_x))
        oy = max(-1.0, min(1.0, strobe_offset_y))
        # x,y offset expressed using crop's `in_w`/`in_h` (= the scaled bg dims).
        # Centre is (in_w-bg_w)/2; user offset shifts within ±centre.
        x_expr_s = f"(in_w-{bg_w})/2 + ({ox})*(in_w-{bg_w})/2"
        y_expr_s = f"(in_h-{bg_h})/2 + ({oy})*(in_h-{bg_h})/2"
        bg_chain += f",crop={bg_w}:{bg_h}:'{x_expr_s}':'{y_expr_s}'"

    bg_chain += "[bg]"

    # 2) Foreground square (centred, fg_w x fg_w)
    fg_chain = f"[0:v]scale={fg_w}:{fg_w}:force_original_aspect_ratio=increase,crop={fg_w}:{fg_w}[fg]"

    # 3) Shadow: black square the same size as fg, padded by `pad` on each side, gblur
    shadow_chain = (
        f"color=c=black@0.55:s={fg_w}x{fg_w}:d={duration:.3f},"
        f"pad={shadow_outer}:{shadow_outer}:{pad}:{pad}:color=black@0.0,"
        f"gblur=sigma=20[shadow]"
    )

    # 4) Pan: crop the oversized bg with a moving window
    max_dx = SOCIALS_BG_PAN_FILL - bg_w   # e.g. 2688-1080 = 1608
    max_dy = SOCIALS_BG_PAN_FILL - bg_h   # e.g. 2688-1920 = 768
    cx = max_dx / 2.0
    cy = max_dy / 2.0
    ax = max_dx / 2.0
    ay = max_dy / 2.0
    omega = f"(2*PI*t/{pan_dur})"

    if animation == "horizontal":
        x_expr = f"({cx})+({ax})*sin({omega})"
        y_expr = f"{cy}"
    elif animation == "diagonal":
        x_expr = f"({cx})+({ax})*sin({omega})"
        y_expr = f"({cy})+({ay})*sin({omega})"
    elif animation == "diamond":
        # True diamond: triangle waves on x and y, 90° out of phase.
        # tri(theta) = (2/PI)*asin(sin(theta)) -> ranges -1..+1 with linear edges,
        # so the trace between extremes is a straight line, giving sharp diamond corners.
        tri_x = f"(2/PI)*asin(sin({omega}))"
        tri_y = f"(2/PI)*asin(sin({omega}-PI/2))"
        x_expr = f"({cx})+({ax})*({tri_x})"
        y_expr = f"({cy})+({ay})*({tri_y})"
    else:
        x_expr = y_expr = None

    if is_pan:
        bg_chain = bg_chain[:-len("[bg]")] + (
            f",crop={bg_w}:{bg_h}:'{x_expr}':'{y_expr}'[bg]"
        )

    # 5) Compose
    sx = (bg_w - shadow_outer) // 2
    sy = (bg_h - shadow_outer) // 2
    fx = (bg_w - fg_w) // 2
    fy = (bg_h - fg_w) // 2

    ovl = overlays or []

    if is_strobe:
        op = max(0.0, min(1.0, strobe_op))
        strobe = (
            f"color=c=black:s={bg_w}x{bg_h}:d={duration:.3f},"
            f"format=rgba,colorchannelmixer=aa={op:.3f}[sblk];"
            f"color=c=white:s={bg_w}x{bg_h}:d={duration:.3f},"
            f"format=rgba,colorchannelmixer=aa={op:.3f}[swht];"
            f"[bg][sblk]overlay=0:0:enable='lt(mod(floor(t*10)\\,2)\\,1)'[bgs1];"
            f"[bgs1][swht]overlay=0:0:enable='gte(mod(floor(t*10)\\,2)\\,1)'[bgstrobed]"
        )
        ov_chain = _socials_overlay_chain(ovl, "bgstrobed", "bgo", duration)
        compose = (
            f"[bgo][shadow]overlay={sx}:{sy}:format=auto[bgs2];"
            f"[bgs2][fg]overlay={fx}:{fy}:format=auto[outv]"
        )
        return f"{bg_chain};{fg_chain};{shadow_chain};{strobe};{ov_chain};{compose}"

    # No strobe: bg → overlays → shadow → fg
    ov_chain = _socials_overlay_chain(ovl, "bg", "bgo", duration)
    compose = (
        f"[bgo][shadow]overlay={sx}:{sy}:format=auto[bgs];"
        f"[bgs][fg]overlay={fx}:{fy}:format=auto[outv]"
    )
    return f"{bg_chain};{fg_chain};{shadow_chain};{ov_chain};{compose}"


def _socials_append_waveform(
    base_fc: str,
    wave_input_idx: int,
    in_label: str,
    out_label: str,
    seg_start: float,
    seg_len: float,
    total_dur: float,
    strip_h: int,
    title: str,
    duration: float,
) -> str:
    """
    Append waveform overlay filters to an existing filter complex string.
    base_fc produces [in_label] at 1080×1920. Returns extended fc producing [out_label].

    strip_h controls how far from the bottom of the frame the strip sits (vertical position).
    No dark background panel. White waveform bars, red playhead line.
    """
    fw, fh = SOCIALS_W, SOCIALS_H
    pad = 30
    # strip_h is repurposed as the bottom offset in pixels (how high up from the bottom)
    # Range 60-220 → bottom gap 60-220px from the bottom of the frame
    bottom_gap = max(40, strip_h)
    title_h = 38 if title else 0
    wave_w = fw - 2 * pad   # 1020
    # Fixed waveform bar height regardless of position slider
    wave_h = 100
    strip_total_h = title_h + wave_h + (10 if title else 0)

    wave_y = fh - bottom_gap - wave_h
    title_y = wave_y - title_h - 4 if title else wave_y

    seg_len_s = max(0.001, seg_len)
    # Playhead x position at time T: sweeps from pad to pad+wave_w
    ph_x0 = pad
    ph_speed = wave_w / seg_len_s  # pixels per second

    parts: List[str] = [base_fc]

    # 1) Overlay waveform image (white bars on transparent bg from showwavespic)
    parts.append(f"[{in_label}][{wave_input_idx}:v]overlay={pad}:{wave_y}[wf1]")

    # 2) Red playhead line using geq (drawbox x-expressions are static in ffmpeg 6.x)
    # geq reads each pixel: if X is within 2px of the playhead position, paint red.
    # wave_y_top = wave_y (top of waveform), wave_y_bot = wave_y + wave_h (bottom)
    wave_y_bot = wave_y + wave_h
    ph_expr = f"{ph_x0}+{ph_speed:.4f}*T"
    parts.append(
        f"[wf1]geq="
        f"r='if(between(X\\,{ph_expr}\\,{ph_expr}+2)*between(Y\\,{wave_y}\\,{wave_y_bot})\\,220\\,r(X\\,Y))':"
        f"g='if(between(X\\,{ph_expr}\\,{ph_expr}+2)*between(Y\\,{wave_y}\\,{wave_y_bot})\\,30\\,g(X\\,Y))':"
        f"b='if(between(X\\,{ph_expr}\\,{ph_expr}+2)*between(Y\\,{wave_y}\\,{wave_y_bot})\\,30\\,b(X\\,Y))'"
        f"[wf2]"
    )

    # 3) Song title (if provided)
    if title:
        safe_title = title.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")
        parts.append(
            f"[wf2]drawtext=text='{safe_title}':fontsize=22:fontcolor=white@0.85:"
            f"x={pad}:y={title_y}[{out_label}]"
        )
    else:
        parts.append(f"[wf2]copy[{out_label}]")

    return ";".join(parts)


def _run_socials_render_job(job_id: str, data: dict) -> None:
    job = _read_job(job_id)
    session_id = data.get("session_id", "")

    try:
        image_path = _socials_find_image(session_id)
        if not image_path:
            job["status"] = "error"; job["error"] = "No image uploaded."
            _write_job(job_id, job); return

        bg_treatment = (data.get("bg_treatment") or "normal").strip().lower()
        if bg_treatment not in ("normal", "inverted", "bw"):
            bg_treatment = "normal"
        animation = (data.get("animation") or "none").strip().lower()
        if animation not in ("none", "strobe", "horizontal", "diagonal", "diamond"):
            animation = "none"
        strobe_op = _safe_float(data.get("strobe_opacity"), 60.0) / 100.0
        pan_dur = max(0.5, _safe_float(data.get("pan_duration"), 4.0))
        strobe_zoom = max(1.0, min(3.0, _safe_float(data.get("strobe_zoom"), 1.0)))
        strobe_off_x = max(-1.0, min(1.0, _safe_float(data.get("strobe_offset_x"), 0.0)))
        strobe_off_y = max(-1.0, min(1.0, _safe_float(data.get("strobe_offset_y"), 0.0)))
        output_format = (data.get("output_format") or "mp4").strip().lower()
        if output_format not in ("mp4", "gif", "both"):
            output_format = "mp4"

        # Parse overlays list from payload
        valid_ov = {"blur", "grain", "glitch", "vignette", "scanlines"}
        raw_ovs = data.get("overlays") or []
        overlays: List[Dict] = []
        if isinstance(raw_ovs, list):
            for ov in raw_ovs:
                if isinstance(ov, dict) and (ov.get("type") or "").lower() in valid_ov:
                    overlays.append({k: v for k, v in ov.items()
                                     if k in ("type","intensity","block_size","strength")})

        # Waveform overlay (requires audio)
        use_waveform = bool(data.get("use_waveform", False))
        waveform_title = str(data.get("waveform_title") or "").strip()[:80]
        waveform_height = max(60, min(300, int(_safe_float(data.get("waveform_height"), 100))))

        # Audio (optional). If present, audio segment drives the duration.
        use_audio = bool(data.get("use_audio", False))
        audio_path = _socials_find_audio(session_id) if use_audio else None
        if use_audio and not audio_path:
            job["status"] = "error"; job["error"] = "Audio enabled but no audio uploaded."
            _write_job(job_id, job); return

        # Waveform also requires audio
        if use_waveform and not audio_path:
            use_waveform = False

        if use_audio:
            seg_len = max(0.5, _safe_float(data.get("segment_length_sec"), 15.0))
            seg_start = max(0.0, _safe_float(data.get("segment_start_sec"), 0.0))
            duration = seg_len
        else:
            seg_len = 0.0; seg_start = 0.0
            duration = max(0.5, _safe_float(data.get("duration"), 8.0))

        # Pre-render waveform image if requested
        waveform_img_path: Optional[Path] = None
        total_audio_dur = 0.0
        if use_waveform and audio_path is not None:
            total_audio_dur = audio_duration(audio_path)
            waveform_img_path = render_out / "waveform.png" if (render_out := RENDER_DIR / session_id / "SOCIALS") else None
            render_out = RENDER_DIR / session_id / "SOCIALS"
            render_out.mkdir(parents=True, exist_ok=True)
            waveform_img_path = render_out / "waveform.png"
            _w = SOCIALS_W - 60  # 1020px, 30px padding each side
            _h = waveform_height
            # Generate waveform from the SEGMENT only (not full song).
            # Render at 4× then downscale with lanczos for crisp antialiased bars.
            wr = subprocess.run([
                FFMPEG, "-y",
                "-ss", str(seg_start), "-t", str(seg_len),
                "-i", str(audio_path),
                "-filter_complex",
                (f"showwavespic=s={_w*4}x{_h*4}:"
                 f"colors=0xffffff:split_channels=0:scale=cbrt:draw=full:filter=average,"
                 f"scale={_w}:{_h}:flags=lanczos"),
                str(waveform_img_path)
            ], capture_output=True, text=True)
            if wr.returncode != 0:
                waveform_img_path = None  # fail gracefully, skip waveform

        render_out = RENDER_DIR / session_id / "SOCIALS"
        render_out.mkdir(parents=True, exist_ok=True)

        # Plan outputs
        targets: List[str] = []
        if output_format in ("mp4", "both"): targets.append("mp4")
        if output_format in ("gif", "both"): targets.append("gif")

        job["total"] = len(targets)
        job["videos"] = []
        _write_job(job_id, job)
        created = 0

        # Build filter graph once — same for all targets
        fc = _socials_build_filter(
            bg_treatment, animation, strobe_op, pan_dur, duration,
            strobe_zoom=strobe_zoom,
            strobe_offset_x=strobe_off_x,
            strobe_offset_y=strobe_off_y,
            overlays=overlays,
        )

        base_name = image_path.stem.replace("image_", "")
        if base_name.endswith("_2k"):
            base_name = base_name[:-3]

        for fmt in targets:
            if STOP_EVENT.is_set(): break
            out_name = f"{base_name}_socials.{fmt}"
            out_path = _next_available(render_out / out_name)
            job["videos"].append({"name": out_path.name, "status": "rendering", "url": None})
            _write_job(job_id, job)
            v_idx = len(job["videos"]) - 1

            # Build inputs: [0]=image, [1]=audio (optional), [N]=waveform_img (optional)
            cmd: List[str] = [FFMPEG, "-y", "-loop", "1", "-i", str(image_path)]
            audio_input_idx = None
            wave_input_idx = None

            if use_audio and audio_path is not None:
                cmd += ["-ss", f"{seg_start}", "-t", f"{seg_len}", "-i", str(audio_path)]
                audio_input_idx = 1

            if waveform_img_path is not None and waveform_img_path.exists():
                cmd += ["-loop", "1", "-i", str(waveform_img_path)]
                wave_input_idx = 2 if audio_input_idx == 1 else 1

            # Extend filter graph with waveform overlay if needed
            if wave_input_idx is not None and total_audio_dur > 0:
                fc_final = _socials_append_waveform(
                    fc, wave_input_idx, "outv", "outv_wave",
                    seg_start=seg_start, seg_len=seg_len,
                    total_dur=total_audio_dur,
                    strip_h=waveform_height, title=waveform_title,
                    duration=duration,
                )
                out_label = "outv_wave"
            else:
                fc_final = fc
                out_label = "outv"

            cmd += ["-filter_complex", fc_final, "-map", f"[{out_label}]"]

            if fmt == "mp4":
                if use_audio and audio_input_idx is not None:
                    cmd += ["-map", f"{audio_input_idx}:a:0", "-c:a", "aac", "-b:a", "320k"]
                cmd += [
                    "-t", f"{duration}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-r", "30", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(out_path),
                ]
            else:  # gif — no audio
                cmd += [
                    "-t", f"{duration}",
                    "-r", "20",
                    "-loop", "0",
                    str(out_path),
                ]

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            set_current_proc(p)
            _, err = p.communicate()
            set_current_proc(None)

            if p.returncode == 0 and out_path.exists():
                created += 1
                job["videos"][v_idx]["status"] = "done"
                job["videos"][v_idx]["url"] = f"/download/{session_id}/SOCIALS/{out_path.name}"
            else:
                job["videos"][v_idx]["status"] = "error"
                job["videos"][v_idx]["error"] = (err or "")[-500:]
            job["done"] = created
            _write_job(job_id, job)

        job["status"] = "done" if created > 0 else "error"
        if created == 0:
            job["error"] = job.get("error") or "Render failed. See per-output errors."
        else:
            job["message"] = f"Done. Created {created} output(s)."
        _write_job(job_id, job)

    except Exception as e:
        job = _read_job(job_id) or job
        job["status"] = "error"; job["error"] = str(e)
        _write_job(job_id, job)


@app.route("/render_socials", methods=["POST"])
def render_socials():
    STOP_EVENT.clear()
    data = request.get_json(force=True) or {}
    job_id = str(uuid.uuid4())
    _write_job(job_id, {
        "status": "running", "total": 0, "done": 0,
        "videos": [], "message": "", "error": "", "zip_url": None,
    })
    threading.Thread(target=_run_socials_render_job, args=(job_id, data), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


# ------------------------------------------------------------
# Download
# ------------------------------------------------------------
@app.route("/download/<session_id>/<path:filename>")
def download(session_id: str, filename: str):
    # Security: only allow downloads from render dir for this session
    base = RENDER_DIR / session_id
    file_path = (base / filename).resolve()
    if not str(file_path).startswith(str(base.resolve())):
        return Response("Forbidden", status=403)
    if not file_path.exists():
        return Response("File not found", status=404)
    return send_file(str(file_path), as_attachment=True, download_name=file_path.name)

# ------------------------------------------------------------
# Cleanup session
# ------------------------------------------------------------
@app.route("/cleanup", methods=["POST"])
def cleanup():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "")
    if session_id:
        cleanup_session(session_id)
    return jsonify({"ok": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)