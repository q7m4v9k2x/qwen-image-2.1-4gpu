"""Qwen Image 2.1 gateway.

The gateway deliberately serialises work sent to ComfyUI. The Qwen tensor
parallel extension uses all four V100s for one request and is not safe to
re-enter, while a small bounded admission queue keeps bursts of HTTP traffic
from growing ComfyUI's queue without limit. Job progress is read from
ComfyUI's websocket when available and is persisted in the in-memory job
record so a browser refresh can continue polling the same job.
"""
import argparse
import base64
import json
import os
import queue
import secrets
import socket
import ssl
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


COMFY = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
ROOT = Path(__file__).parent
PUBLIC_BASE_PATH = os.getenv("QWEN_PUBLIC_BASE_PATH", "/qwen-image-ui").rstrip("/") or ""


def _positive_int(name, default, minimum, maximum):
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


# A TP task owns all four GPUs. Keep one worker by default; users can opt into
# another worker only when running a non-TP ComfyUI instance.
MAX_PENDING = _positive_int("QWEN_GATEWAY_MAX_PENDING", 8, 1, 128)
WORKERS = _positive_int("QWEN_GATEWAY_WORKERS", 1, 1, 8)
MAX_PIXELS = _positive_int("QWEN_MAX_PIXELS", 4 * 1024 * 1024, 256 * 256, 16 * 1024 * 1024)
MAX_DIMENSION = _positive_int("QWEN_MAX_DIMENSION", 2048, 256, 8192)
MAX_STEPS = _positive_int("QWEN_MAX_STEPS", 50, 1, 200)
JOB_TIMEOUT = _positive_int("QWEN_JOB_TIMEOUT_SECONDS", 2 * 60 * 60, 60, 24 * 60 * 60)
JOB_TTL = _positive_int("QWEN_JOB_TTL_SECONDS", 24 * 60 * 60, 300, 7 * 24 * 60 * 60)


# Public state. JOB_PROGRESS remains for compatibility with the original
# implementation and with small diagnostic scripts that import this module.
JOB_PROGRESS = {}
JOBS = {}
JOB_LOCK = threading.RLock()
JOB_QUEUE = queue.Queue(maxsize=MAX_PENDING)
WORKER_THREADS = []
WORKERS_STARTED = False
WORKERS_START_LOCK = threading.Lock()


class QueueLimitError(Exception):
    """Raised when the bounded gateway queue has no admission slot."""


def _now():
    return time.time()


def _prune_jobs_locked():
    cutoff = _now() - JOB_TTL
    stale = [
        job_id for job_id, state in JOBS.items()
        if state.get("status") in {"completed", "failed"}
        and state.get("updated", state.get("created", 0)) < cutoff
    ]
    for job_id in stale:
        JOBS.pop(job_id, None)
        JOB_PROGRESS.pop(job_id, None)


def _active_count_locked():
    return sum(
        1 for state in JOBS.values()
        if state.get("status") in {"queued", "submitting", "running", "saving"}
    )


def _set_job(job_id, **changes):
    with JOB_LOCK:
        state = JOBS.get(job_id)
        if state is None:
            return None
        state.update(changes)
        state["updated"] = _now()
        progress = state.get("progress")
        if isinstance(progress, dict):
            JOB_PROGRESS[job_id] = dict(progress)
        return dict(state)


def _job_snapshot(job_id):
    with JOB_LOCK:
        state = JOBS.get(job_id)
        if state is None:
            return None
        result = dict(state)
        result["progress"] = dict(state.get("progress") or {})
        result["images"] = list(state.get("images") or [])
        return result


def progress_for(job_id):
    with JOB_LOCK:
        return dict(JOB_PROGRESS.get(job_id, {}))


def _as_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        integer = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是整数") from None
    if isinstance(value, float) and value != integer:
        raise ValueError(f"{name} 必须是整数")
    return integer


def _parse_size(value):
    if not isinstance(value, str):
        raise ValueError("size 必须为 WIDTHxHEIGHT，例如 1024x1024")
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("size 必须为 WIDTHxHEIGHT，例如 1024x1024") from None
    if width < 256 or height < 256 or width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ValueError(f"单边分辨率必须在 256–{MAX_DIMENSION} 之间")
    if width % 8 or height % 8:
        raise ValueError("width 和 height 必须是 8 的倍数")
    pixels = width * height
    if pixels > MAX_PIXELS:
        max_mp = MAX_PIXELS / 1_000_000
        raise ValueError(f"图片像素不能超过 {MAX_PIXELS:,}（约 {max_mp:.1f}MP），1536² 请先确认显存")
    return width, height, f"{width}x{height}"


def workflow(body):
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 6000:
        raise ValueError("请输入 1–6000 字的提示词")
    if body.get("model", "qwen-image-2.1") != "qwen-image-2.1":
        raise ValueError("model 必须为 qwen-image-2.1")
    width, height, _ = _parse_size(body.get("size", "1024x1024"))
    steps = _as_int(body.get("steps", 25), "steps")
    seed = _as_int(body.get("seed", -1), "seed")
    if not 1 <= steps <= MAX_STEPS or not -1 <= seed <= 2**53 - 1:
        raise ValueError(f"steps 应为 1–{MAX_STEPS}，seed 应为 -1 或非负安全整数")
    if seed == -1:
        seed = secrets.randbelow(2**53)

    def n(cls, **inputs):
        return {"class_type": cls, "inputs": inputs}

    return {
        "1": n("UNETLoader", unet_name="qwen_image_2.1_int8_convrot.safetensors", weight_dtype="default"),
        "2": n("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default"),
        "3": n("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors"),
        "4": n("TextEncodeQwenImage21", clip=["2", 0], prompt=prompt.strip(), negative_prompt="", resolution=1024),
        "5": n("EmptyLatentImage", width=width, height=height, batch_size=1),
        "6": n("KSampler", model=["1", 0], positive=["4", 0], negative=["4", 1], latent_image=["5", 0],
               seed=seed, steps=steps, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0),
        "7": n("VAEDecode", samples=["6", 0], vae=["3", 0]),
        "8": n("SaveImage", images=["7", 0], filename_prefix="QwenStudio"),
    }, seed


def upstream(path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        COMFY + path, data,
        {"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw)


def _ws_read_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("ComfyUI websocket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _ws_send(sock, opcode, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    mask = secrets.token_bytes(4)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    length = len(masked)
    if length < 126:
        header = bytes([0x80 | opcode, 0x80 | length])
    elif length < 65536:
        header = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack("!H", length)
    else:
        header = bytes([0x80 | opcode, 0x80 | 127]) + struct.pack("!Q", length)
    sock.sendall(header + mask + masked)


def _ws_recv(sock):
    first, second = _ws_read_exact(sock, 2)
    fin = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _ws_read_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _ws_read_exact(sock, 8))[0]
    if length > 16 * 1024 * 1024:
        raise ValueError("ComfyUI websocket frame too large")
    mask = _ws_read_exact(sock, 4) if masked else None
    payload = _ws_read_exact(sock, length) if length else b""
    if mask:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    if not fin:
        fragments = [payload]
        while True:
            continuation = _ws_recv(sock)
            if continuation is None:
                return None
            fragments.append(continuation)
            break
        payload = b"".join(fragments)
    if opcode == 8:
        return None
    if opcode == 9:
        _ws_send(sock, 10, payload)
        return b""
    if opcode in (1, 2, 0):
        return payload
    return b""


def _ws_connect(client_id):
    parsed = urllib.parse.urlsplit(COMFY)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("COMFY_URL 必须是 http(s) URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    sock = socket.create_connection((parsed.hostname, port), timeout=5)
    if parsed.scheme == "https":
        context = ssl.create_default_context()
        sock = context.wrap_socket(sock, server_hostname=parsed.hostname)
    sock.settimeout(2)
    key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
    path = (parsed.path.rstrip("/") or "") + "/ws?clientId=" + urllib.parse.quote(client_id)
    host = parsed.hostname if parsed.port is None else f"{parsed.hostname}:{parsed.port}"
    request = (
        f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
        f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    ).encode("ascii")
    sock.sendall(request)
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("ComfyUI websocket handshake closed")
        response += chunk
        if len(response) > 65536:
            raise ConnectionError("ComfyUI websocket handshake too large")
    header = response.split(b"\r\n", 1)[0]
    if not header.startswith(b"HTTP/1.1 101"):
        raise ConnectionError("ComfyUI websocket handshake failed: " + header.decode("latin1", "replace"))
    return sock


def _terminal(job_id):
    with JOB_LOCK:
        state = JOBS.get(job_id)
        return state is None or state.get("status") in {"completed", "failed"}


def watch_progress(job_id, prompt_id, client_id):
    """Copy ComfyUI step events into /jobs/{job_id} using stdlib only."""
    try:
        sock = _ws_connect(client_id)
    except Exception:
        return
    try:
        while not _terminal(job_id):
            try:
                raw = _ws_recv(sock)
            except socket.timeout:
                continue
            if raw is None:
                return
            if not raw:
                continue
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            data = event.get("data") or {}
            event_prompt = data.get("prompt_id")
            if event_prompt and event_prompt != prompt_id:
                continue
            kind = event.get("type")
            if kind == "progress":
                current = max(0, _as_int(data.get("value", 0), "progress"))
                total = max(0, _as_int(data.get("max", 0), "progress"))
                with JOB_LOCK:
                    state = JOBS.get(job_id)
                    if state and state.get("status") not in {"completed", "failed"}:
                        previous = state.get("progress", {})
                        total = total or int(previous.get("total", 0))
                        state["progress"] = {"current": current, "total": total, "node": data.get("node")}
                        state["status"] = "running"
                        state["updated"] = _now()
                        JOB_PROGRESS[job_id] = dict(state["progress"])
            elif kind == "executing":
                with JOB_LOCK:
                    state = JOBS.get(job_id)
                    if state and state.get("status") not in {"completed", "failed"}:
                        progress = dict(state.get("progress") or {})
                        progress.setdefault("current", 0)
                        progress.setdefault("total", state.get("steps", 0))
                        progress["node"] = data.get("node")
                        state["progress"] = progress
                        state["status"] = "running"
                        state["updated"] = _now()
                        JOB_PROGRESS[job_id] = progress
    finally:
        try:
            _ws_send(sock, 8, struct.pack("!H", 1000))
        except Exception:
            pass
        sock.close()


def _record_error(record):
    status = record.get("status", {}) if isinstance(record, dict) else {}
    if status.get("status_str") != "error":
        return None
    messages = status.get("messages", [])
    for item in messages:
        if isinstance(item, (list, tuple)) and len(item) > 1 and item[0] == "execution_error":
            info = item[1] if isinstance(item[1], dict) else {}
            return str(info.get("exception_message") or info.get("exception_type") or "生成失败")
    return "生成失败"


def _images_from_record(record):
    return [
        image for output in (record.get("outputs", {}) or {}).values()
        for image in (output.get("images", []) or [])
        if isinstance(image, dict) and image.get("filename")
    ]


def _run_job(job_id):
    state = _job_snapshot(job_id)
    if not state:
        return
    try:
        _set_job(job_id, status="submitting")
        client_id = str(uuid.uuid4())
        result = upstream("/prompt", {"prompt": state["graph"], "client_id": client_id}, timeout=60)
        prompt_id = str(result["prompt_id"])
        _set_job(job_id, prompt_id=prompt_id, client_id=client_id, status="running",
                 progress={"current": 0, "total": state["steps"], "node": None})
        threading.Thread(target=watch_progress, args=(job_id, prompt_id, client_id), daemon=True).start()
        deadline = time.monotonic() + JOB_TIMEOUT
        while time.monotonic() < deadline:
            if _terminal(job_id):
                return
            try:
                history = upstream("/history/" + urllib.parse.quote(prompt_id, safe=""), timeout=20)
                record = history.get(prompt_id) if isinstance(history, dict) else None
                if record:
                    error = _record_error(record)
                    if error:
                        _set_job(job_id, status="failed", error=error[:1000])
                        return
                    images = _images_from_record(record)
                    status = record.get("status", {})
                    if images or status.get("completed"):
                        if not images:
                            _set_job(job_id, status="failed", error="ComfyUI 完成但没有返回图片")
                        else:
                            _set_job(job_id, status="completed", images=images,
                                     progress={"current": state["steps"], "total": state["steps"], "node": None})
                        return
                queue_state = upstream("/queue", timeout=10)
                running = any(item[1] == prompt_id for item in queue_state.get("queue_running", []))
                pending = any(item[1] == prompt_id for item in queue_state.get("queue_pending", []))
                if pending and not running:
                    _set_job(job_id, status="queued")
                elif running:
                    _set_job(job_id, status="running")
            except Exception as exc:
                # A transient poll failure must not discard the ComfyUI job.
                _set_job(job_id, last_poll_error=str(exc)[:300])
            time.sleep(0.5)
        _set_job(job_id, status="failed", error=f"任务超过 {JOB_TIMEOUT // 60} 分钟仍未完成")
    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc)[:1000])


def _worker_loop():
    while True:
        job_id = JOB_QUEUE.get()
        try:
            _run_job(job_id)
        finally:
            JOB_QUEUE.task_done()


def ensure_workers():
    global WORKERS_STARTED
    with WORKERS_START_LOCK:
        if WORKERS_STARTED:
            return
        for _ in range(WORKERS):
            thread = threading.Thread(target=_worker_loop, name="qwen-gateway-worker", daemon=True)
            thread.start()
            WORKER_THREADS.append(thread)
        WORKERS_STARTED = True


def _new_job(graph, seed, steps):
    ensure_workers()
    job_id = str(uuid.uuid4())
    state = {
        "id": job_id, "graph": graph, "seed": seed, "steps": steps,
        "status": "queued", "progress": {"current": 0, "total": steps, "node": None},
        "images": [], "created": _now(), "updated": _now(),
    }
    with JOB_LOCK:
        _prune_jobs_locked()
        if _active_count_locked() >= MAX_PENDING:
            raise QueueLimitError(f"生成队列已满（最多 {MAX_PENDING} 个任务），请稍后重试")
        JOBS[job_id] = state
        JOB_PROGRESS[job_id] = dict(state["progress"])
        try:
            JOB_QUEUE.put_nowait(job_id)
        except queue.Full:
            JOBS.pop(job_id, None)
            JOB_PROGRESS.pop(job_id, None)
            raise QueueLimitError(f"生成队列已满（最多 {MAX_PENDING} 个任务），请稍后重试") from None
    return state


def _public_job(state):
    response = {"status": state.get("status", "unknown"), "progress": dict(state.get("progress") or {})}
    if state.get("status") == "completed":
        response["data"] = [{"url": "image?" + urllib.parse.urlencode(image)} for image in state.get("images", [])]
    if state.get("status") == "failed":
        response["error"] = state.get("error", "生成失败")
    if state.get("prompt_id"):
        response["prompt_id"] = state["prompt_id"]
    if state.get("status") in {"queued", "submitting"}:
        with JOB_LOCK:
            queued = [item for item in JOBS.values() if item.get("status") in {"queued", "submitting"}]
        response["queue_position"] = max(1, next((index for index, item in enumerate(queued, 1) if item.get("id") == state.get("id")), 1))
    return response


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, status, data, mime="application/json; charset=utf-8", headers=None):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8") if isinstance(data, dict) else data
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, str(value))
        self.end_headers()
        self.wfile.write(raw)

    @staticmethod
    def _valid_job_id(value):
        try:
            return str(uuid.UUID(value)) == value.lower()
        except (ValueError, AttributeError, TypeError):
            return False

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self.reply(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/health":
                upstream("/system_stats", timeout=10)
                with JOB_LOCK:
                    active = _active_count_locked()
                return self.reply(200, {"ok": True, "model": "qwen-image-2.1",
                                       "queue": {"active": active, "limit": MAX_PENDING}})
            if path.startswith("/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                if not self._valid_job_id(job_id):
                    return self.reply(400, {"error": "无效任务 ID"})
                job_id = job_id.lower()
                state = _job_snapshot(job_id)
                if state:
                    return self.reply(200, _public_job(state))
                # Compatibility for jobs submitted directly to ComfyUI before
                # this gateway process was restarted.
                record = upstream("/history/" + urllib.parse.quote(job_id, safe=""), timeout=15).get(job_id)
                if record:
                    error = _record_error(record)
                    if error:
                        return self.reply(200, {"status": "failed", "error": error[:1000]})
                    images = _images_from_record(record)
                    if images:
                        return self.reply(200, {"status": "completed", "progress": {"current": 1, "total": 1},
                                               "data": [{"url": "image?" + urllib.parse.urlencode(image)} for image in images]})
                return self.reply(404, {"error": "任务不存在或网关已重启"})
            if path == "/image":
                args = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                filename = args.get("filename", [""])[0]
                if not filename.startswith("QwenStudio_") or "/" in filename or "\\" in filename or ".." in filename:
                    return self.reply(400, {"error": "无效图片"})
                query = urllib.parse.urlencode({"filename": filename, "type": "output", "subfolder": ""})
                with urllib.request.urlopen(COMFY + "/view?" + query, timeout=30) as response:
                    return self.reply(200, response.read(), response.headers.get("Content-Type", "image/png"))
            return self.reply(404, {"error": "not found"})
        except urllib.error.HTTPError as exc:
            return self.reply(502, {"error": f"ComfyUI HTTP {exc.code}"})
        except Exception as exc:
            return self.reply(502, {"error": str(exc)[:1000]})

    def do_POST(self):
        if urllib.parse.urlsplit(self.path).path != "/v1/images/generations":
            return self.reply(404, {"error": "not found"})
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= 32768:
                return self.reply(413, {"error": "请求大小必须小于 32 KB"})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("请求必须为 JSON 对象")
            graph, seed = workflow(body)
            state = _new_job(graph, seed, _as_int(body.get("steps", 25), "steps"))
            self.reply(202, {
                "created": int(state["created"]), "id": state["id"], "job_id": state["id"],
                "seed": seed, "status": "queued", "status_url": PUBLIC_BASE_PATH + "/jobs/" + state["id"],
                "progress": state["progress"], "queue_limit": MAX_PENDING,
            })
        except QueueLimitError as exc:
            return self.reply(429, {"error": str(exc), "retry_after": 5}, headers={"Retry-After": "5"})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self.reply(400, {"error": str(exc)})
        except Exception as exc:
            return self.reply(502, {"error": str(exc)[:1000]})

    def log_message(self, format, *args):
        return super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("GATEWAY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GATEWAY_PORT", "8190")))
    args = parser.parse_args()
    ensure_workers()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()

