"""Submit one Qwen Image job to ComfyUI and save its timings, history and images.

Only the Python standard library is required. The workflow comes from the sibling
gateway.py, so this benchmark uses the same model and sampler as the web service.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def request(url, path, *, body=None, timeout=30):
    """Return bytes and headers; never retry a POST that may have been accepted."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {path}: {detail}") from exc


def request_json(url, path, **kwargs):
    raw, _ = request(url, path, **kwargs)
    return json.loads(raw)


def write_result(path, result):
    """Keep a useful record even if generation times out or the process stops."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def history_timings(record):
    messages = record.get("status", {}).get("messages", [])
    starts = [data.get("timestamp") for kind, data in messages if kind == "execution_start"]
    ends = [data.get("timestamp") for kind, data in messages if kind == "execution_success"]
    cached = [node for kind, data in messages if kind == "execution_cached" for node in data.get("nodes", [])]
    duration = None
    if starts and ends and isinstance(starts[0], (int, float)) and isinstance(ends[-1], (int, float)):
        duration = (ends[-1] - starts[0]) / 1000
    return {"server_execution_seconds": duration, "cached_node_ids": cached,
            "sampler_cached": "6" in cached}


def run_benchmark(args):
    """Return (record_path, record); timeouts leave the remote job running."""
    from gateway import workflow

    body = {"prompt": args.prompt, "size": args.size, "steps": args.steps, "seed": args.seed}
    graph, seed = workflow(body)
    run_id = f"{args.prefix}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    graph["8"]["inputs"]["filename_prefix"] = run_id
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / f"{run_id}.json"
    result = {
        "run_id": run_id, "url": args.url, "status": "preparing",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parameters": {**body, "seed": seed}, "workflow": graph, "images": [],
        "timeout_seconds": args.timeout, "poll_interval_seconds": args.poll_interval,
        "timing_note": "Client generation time includes queueing and history polling; server execution time includes graph execution, not only sampling.",
    }
    start = time.monotonic()
    deadline = start + args.timeout

    def remaining_timeout():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Benchmark exceeded {args.timeout:g} seconds; remote job was not cancelled.")
        return min(args.request_timeout, remaining)

    write_result(record_path, result)
    try:
        result["system_stats"] = request_json(args.url, "/system_stats", timeout=remaining_timeout())
        submitted_at = time.monotonic()
        response = request_json(args.url, "/prompt", body={"prompt": graph, "client_id": run_id}, timeout=remaining_timeout())
        result["submission"] = response
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {response}")
        result.update({"prompt_id": prompt_id, "status": "submitted"})
        result["submission_seconds"] = time.monotonic() - submitted_at
        write_result(record_path, result)
        print(f"Submitted {prompt_id}; seed={seed}; result={record_path}", flush=True)
        last_notice = time.monotonic()
        while True:
            history = request_json(args.url, "/history/" + urllib.parse.quote(prompt_id, safe=""), timeout=remaining_timeout())
            record = history.get(prompt_id)
            if record:
                result["history"] = record
                result.update(history_timings(record))
                status = record.get("status", {})
                if status.get("status_str") == "error":
                    errors = [data for kind, data in status.get("messages", []) if kind in {"execution_error", "execution_interrupted"}]
                    raise RuntimeError(json.dumps(errors or status, ensure_ascii=False))
                images = [im for output in record.get("outputs", {}).values() for im in output.get("images", [])]
                if images or status.get("completed"):
                    result["generation_seconds"] = time.monotonic() - submitted_at
                    if not images:
                        raise RuntimeError("ComfyUI completed the workflow without any image output.")
                    result["status"] = "downloading"
                    write_result(record_path, result)
                    for index, image in enumerate(images, start=1):
                        query = urllib.parse.urlencode({key: image.get(key, default) for key, default in (("filename", ""), ("subfolder", ""), ("type", "output"))})
                        content, headers = request(args.url, "/view?" + query, timeout=remaining_timeout())
                        if not content:
                            raise RuntimeError("ComfyUI returned an empty image.")
                        suffix = Path(image.get("filename", "image.png")).suffix.lower()
                        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                            suffix = ".img"
                        image_path = out_dir / f"{run_id}_{index:02d}{suffix}"
                        image_path.write_bytes(content)
                        result["images"].append({"path": str(image_path), "bytes": len(content),
                                                 "sha256": hashlib.sha256(content).hexdigest(),
                                                 "source": image})
                        write_result(record_path, result)
                    result["status"] = "completed"
                    break
            now = time.monotonic()
            if now - last_notice >= 15:
                print(f"Waiting for {prompt_id}: {now - submitted_at:.1f}s", flush=True)
                last_notice = now
            pause = min(args.poll_interval, deadline - time.monotonic())
            if pause > 0:
                time.sleep(pause)
    except KeyboardInterrupt:
        result.update({"status": "interrupted", "error": "Local polling interrupted; the remote job was not cancelled."})
    except Exception as exc:
        result.update({"status": "timeout" if isinstance(exc, TimeoutError) else "failed", "error": str(exc)})
    finally:
        result["total_seconds"] = time.monotonic() - start
        result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_result(record_path, result)
    return record_path, result


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI base URL (not the 8190 gateway)")
    parser.add_argument("--size", default="512x512", help="WIDTHxHEIGHT; the gateway workflow validates 8-pixel alignment and limits")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="-1 chooses a random seed; repeated seeds may reuse ComfyUI's sampler cache")
    parser.add_argument("--prompt", default="a ceramic teapot on a wooden table, soft natural light, detailed photograph")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--prefix", default="qwen_bench", help="Safe filename prefix; every run appends UTC time and a unique suffix")
    parser.add_argument("--timeout", type=positive_number, default=1800, help="Overall timeout in seconds, including image downloads; does not cancel remote job")
    parser.add_argument("--request-timeout", type=positive_number, default=30, help="Per HTTP request timeout in seconds")
    parser.add_argument("--poll-interval", type=positive_number, default=2)
    args = parser.parse_args(argv)
    parsed = urllib.parse.urlsplit(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment or parsed.username or parsed.password:
        parser.error("--url must be an HTTP(S) URL without credentials, query or fragment")
    if not 1 <= len(args.prefix) <= 80 or not all(char.isascii() and (char.isalnum() or char in "_-") for char in args.prefix):
        parser.error("--prefix must contain 1-80 ASCII letters, digits, underscores or hyphens")
    try:
        path, result = run_benchmark(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if result["status"] != "completed":
        print(f"{result['status'].upper()}: {result.get('error')}; result={path}", file=sys.stderr)
        return 1
    print(f"COMPLETED: generation={result['generation_seconds']:.2f}s; total={result['total_seconds']:.2f}s; result={path}")
    for image in result["images"]:
        print(f"Image: {image['path']}")
    if result.get("sampler_cached"):
        print("Sampler output was cached. Choose a new --seed for a fresh performance measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
