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
        save_path = audio_dir / Path(f.filename).name
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
def ensure_1080p_visual(visual_path: Path, sess_dir: Path) -> Path:
    out = sess_dir / f"{visual_path.stem}_1080p{visual_path.suffix}"
    if out.exists(): return out
    cmd = [FFMPEG, "-y", "-i", str(visual_path),
           "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
           "-an", str(out)]
    subprocess.run(cmd, capture_output=True)
    return out

@app.route("/render", methods=["POST"])
def render():
    STOP_EVENT.clear()
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "")
    audio_only = bool(data.get("audio_only", False))
    export_mp3 = bool(data.get("export_mp3", False))

    sess_dir = get_session_dir(session_id)
    audio_dir = sess_dir / "audio"
    render_out = RENDER_DIR / session_id
    render_out.mkdir(parents=True, exist_ok=True)

    audio_files = sorted([
        f for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    ]) if audio_dir.exists() else []

    if not audio_files:
        return Response("No audio files found.", status=400)

    # ---- AUDIO ONLY MODE ----
    if audio_only:
        clips = data.get("clips") or []
        clips_dir = render_out / "CLIPS"
        clips_dir.mkdir(exist_ok=True)
        created = 0
        tmp_paths = []

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
            set_current_proc(p)
            p.communicate()
            set_current_proc(None)
            if p.returncode == 0: created += 1

        # Zip clips for download
        if created > 0:
            zip_path = render_out / "clips.zip"
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(clips_dir))
            return jsonify({"ok": True, "message": f"Created {created} clip(s).",
                           "download_url": f"/download/{session_id}/clips.zip"})
        return Response("No clips were created.", status=500)

    # ---- VIDEO MODE ----
    visuals = [f for f in sess_dir.iterdir()
               if f.name.startswith("visual_") and f.suffix.lower() in VISUAL_EXTS]
    if not visuals:
        return Response("No visual found.", status=400)

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

    if visual_is_169:
        visual_path = ensure_1080p_visual(visual_path, sess_dir)

    separate_169 = (visual_is_169 and sixteen_nine_mode == "separate")
    if (not visual_is_169) or separate_169:
        match = [a for a in audio_files if a.name == selected_audio_name]
        if not match: return Response("Please choose a valid audio file.", status=400)
        render_audio_files = match
    else:
        render_audio_files = audio_files
        use_segment = False
        fade = False

    created = 0
    download_urls = []

    for audio in render_audio_files:
        if STOP_EVENT.is_set(): break
        output_video = render_out / f"{audio.stem}.mp4"
        if output_video.exists(): continue

        if visual_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            input_v = ["-loop", "1", "-i", str(visual_path)]
        else:
            input_v = ["-stream_loop", "-1", "-i", str(visual_path)]

        cmd = [FFMPEG, "-y", *input_v]
        if use_segment:
            cmd += ["-ss", str(max(seg_start, 0.0))]
            if seg_len > 0: cmd += ["-t", str(seg_len)]

        cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0"]

        if fade:
            dur = seg_len if (use_segment and seg_len > 0) else audio_duration(audio)
            f_in = _safe_float(data.get("fade_in"), 0.0)
            f_out = _safe_float(data.get("fade_out"), 0.0)
            cmd += ["-af", f"afade=t=in:st=0:d={f_in}:curve=tri,afade=t=out:st={max(dur-f_out,0.0)}:d={f_out}:curve=tri"]

        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-r", "24",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.0",
            "-c:a", "aac",
            "-b:a", "320k",
            "-movflags", "+faststart",
            "-shortest",
            str(output_video)
        ]

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        set_current_proc(p)
        _, err = p.communicate()
        set_current_proc(None)

        if p.returncode == 0:
            created += 1
            download_urls.append({
                "filename": output_video.name,
                "url": f"/download/{session_id}/{output_video.name}"
            })
        else:
            return Response(f"Render failed: {err}", status=500)

    if created == 0:
        return Response("No videos were created (all may already exist).", status=400)

    # If multiple videos, also offer a zip
    if len(download_urls) > 1:
        zip_path = render_out / "videos.zip"
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(render_out),
                           base_dir=None)
        download_urls.append({"filename": "videos.zip", "url": f"/download/{session_id}/videos.zip"})

    return jsonify({
        "ok": True,
        "message": f"Done. Created {created} video(s).",
        "downloads": download_urls,
    })

# ------------------------------------------------------------
# Merge clips
# ------------------------------------------------------------
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
