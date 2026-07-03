#!/usr/bin/env python3
"""
agentloop.py — multi-turn tool-calling agent-loop eval for a local llama-server.

Drives an OpenAI-compatible /v1/chat/completions endpoint (the froggeric Qwen
tool-call template, --jinja) through a scripted task that REQUIRES several tool
calls to finish — mirroring how OpenClaude / Hermes Agent actually use the model
(read -> edit -> run -> observe -> repeat). Objective completion is decided by the
scenario's own `check` command, not a judge.

Measures the axes a single-shot quality pass misses:
  - tool-call format reliability (structured tool_calls vs raw <tool_call> drift)
  - multi-step task completion (objective: scenario check exits 0)
  - loop truncation (hit max_turns without finishing)
  - think-token overhead (reasoning burned before acting) — the chat/agent latency tax
  - per-request latency (median request wall-time = a TTFT proxy at growing context)

SECURITY: run_bash executes model-emitted commands with shell=True inside a
per-run temp sandbox (cwd + 30s timeout + path confinement for file tools). This
is NOT a real jail. Run only on a box you control, against trusted scenarios.

Server lifecycle (boot/stop, template/moe/ngl flags) is owned by run-agentloop.sh;
this script assumes the server is already up on --port.
"""
import argparse, json, os, re, shutil, subprocess, time, urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the working directory and return its contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Overwrite (or create) a text file in the working directory with the given content.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Run a shell command in the working directory; returns combined stdout+stderr and the exit code.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
KNOWN = {"read_file", "write_file", "run_bash"}
RAW_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _safe(workdir, path):
    """Confine file access to the sandbox; raise if a path tries to escape."""
    root = os.path.realpath(workdir)
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError(f"path escapes sandbox: {path}")
    return full


def exec_tool(name, args, workdir):
    try:
        if name == "read_file":
            with open(_safe(workdir, args["path"]), "r", encoding="utf-8", errors="replace") as f:
                return f.read()[:20000]
        if name == "write_file":
            p = _safe(workdir, args["path"])
            os.makedirs(os.path.dirname(p) or workdir, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(args["content"])
            return f"wrote {len(args['content'])} bytes to {args['path']}"
        if name == "run_bash":
            r = subprocess.run(args["command"], shell=True, cwd=workdir,
                               capture_output=True, text=True, timeout=30)
            return f"exit={r.returncode}\n{(r.stdout + r.stderr)[-8000:]}"
        return f"error: unknown tool {name}"
    except subprocess.TimeoutExpired:
        return "error: command timed out after 30s"
    except Exception as e:  # noqa: BLE001 — surface any tool failure back to the model
        return f"error: {e}"


def chat(port, messages, gen, temp):
    body = json.dumps({
        "messages": messages, "tools": TOOLS, "tool_choice": "auto",
        "temperature": temp, "top_p": 0.95, "top_k": 20, "max_tokens": gen,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp), (time.time() - t0) * 1000.0


def parse_raw_tool_calls(content):
    """Recover <tool_call>{...}</tool_call> JSON the server did NOT structure — format drift."""
    out = []
    for m in RAW_TC.finditer(content or ""):
        try:
            out.append(json.loads(m.group(1)))
        except Exception:  # noqa: BLE001
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sys", default="")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--gen", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()

    spec = json.load(open(os.path.join(a.scenario, "task.json")))
    max_turns = spec.get("max_turns", 12)

    if os.path.exists(a.workdir):
        shutil.rmtree(a.workdir)
    shutil.copytree(os.path.join(a.scenario, "seed"), a.workdir)

    messages = []
    if a.sys:
        messages.append({"role": "system", "content": a.sys})
    messages.append({"role": "user", "content": spec["user"]})

    m = {"turns": 0, "tool_calls_total": 0, "tool_calls_valid": 0, "format_drift": 0,
         "think_chars": 0, "req_ms": [], "truncated_loop": False,
         "task_completed": False, "error": None}
    try:
        for turn in range(max_turns):
            m["turns"] = turn + 1
            data, ms = chat(a.port, messages, a.gen, a.temp)
            m["req_ms"].append(round(ms))
            msg = data["choices"][0]["message"]
            m["think_chars"] += len(msg.get("reasoning_content") or "")

            tcs = msg.get("tool_calls") or []
            if not tcs:  # server didn't structure it — try raw <tool_call> text (drift)
                raw = parse_raw_tool_calls(msg.get("content"))
                if raw:
                    m["format_drift"] += len(raw)
                    tcs = [{"id": f"raw{i}", "function": {
                        "name": c.get("name"), "arguments": json.dumps(c.get("arguments", {}))}}
                        for i, c in enumerate(raw)]
            if not tcs:  # answered with no tool call -> model considers itself done
                messages.append({"role": "assistant", "content": msg.get("content") or ""})
                break

            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": [{"id": tc.get("id", f"c{i}"), "type": "function",
                                             "function": tc["function"]}
                                            for i, tc in enumerate(tcs)]})
            for i, tc in enumerate(tcs):
                m["tool_calls_total"] += 1
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if fn.get("name") in KNOWN:
                        m["tool_calls_valid"] += 1
                    result = exec_tool(fn.get("name"), args, a.workdir)
                except Exception as e:  # noqa: BLE001
                    result = f"error: bad tool call: {e}"
                messages.append({"role": "tool", "tool_call_id": tc.get("id", f"c{i}"),
                                 "content": str(result)[:8000]})
        else:
            m["truncated_loop"] = True  # exhausted max_turns without a natural stop

        chk = spec.get("check")
        if chk:
            r = subprocess.run(chk, shell=True, cwd=a.workdir,
                               capture_output=True, text=True, timeout=60)
            m["task_completed"] = (r.returncode == 0)
            m["check_output"] = (r.stdout + r.stderr)[-2000:]
    except Exception as e:  # noqa: BLE001
        m["error"] = str(e)

    m["req_ms_median"] = sorted(m["req_ms"])[len(m["req_ms"]) // 2] if m["req_ms"] else None
    json.dump(m, open(a.out, "w"), indent=2)
    json.dump(messages, open(a.out.replace(".json", "_transcript.json"), "w"), indent=2)
    print(json.dumps({k: m[k] for k in ("turns", "tool_calls_valid", "tool_calls_total",
          "format_drift", "think_chars", "req_ms_median", "truncated_loop", "task_completed")}))


if __name__ == "__main__":
    main()
