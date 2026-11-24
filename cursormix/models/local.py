"""Offline Qwen3-4B text inference through a loopback Ollama server."""

import json
import re
from http.client import HTTPConnection, HTTPException

from .errors import OutputLimitReached

MODEL = "qwen3:4b"


def answer(prompt, system, limit, *, json_mode=False):
    # This Qwen checkpoint's default chat template ends inside an open thinking block.
    content = prompt.replace("<|im_", "‹|im_")
    if isinstance(json_mode, dict):
        system += "\nUse this JSON schema: " + json.dumps(json_mode, separators=(",", ":"))
    chat = (
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    payload = {
        "model": MODEL,
        "prompt": chat,
        "raw": True,
        # Keep the task and window title alongside a full Accessibility inventory.
        "options": {"num_ctx": 8192, "num_predict": limit, "temperature": 0 if json_mode else 0.1},
        "stream": True,
    }
    payload["format"] = json_mode if isinstance(json_mode, dict) else "json"
    if not json_mode:
        payload["format"] = {
            "type": "object",
            "properties": {"reply": {"type": "string"}},
            "required": ["reply"],
            "additionalProperties": False,
        }
    buffer = ""
    sent = 0
    connection = HTTPConnection("127.0.0.1", 11434, timeout=90)
    try:
        connection.request(
            "POST", "/api/generate", json.dumps(payload).encode(), {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        if response.status == 404:
            raise RuntimeError(f"Install the local model with ollama pull {MODEL}.")
        if response.status != 200:
            raise RuntimeError(f"Local Ollama returned HTTP {response.status}.")
        for line in response:
            if not line.strip():
                continue
            packet = json.loads(line)
            if packet.get("error"):
                raise RuntimeError(f"Ollama could not finish the answer: {packet['error']}")
            text = packet.get("response")
            if text:
                if json_mode:
                    yield text
                else:
                    # Stream only decoded contents of the constrained reply field.
                    buffer += text
                    start = re.match(r'\s*\{\s*"reply"\s*:\s*"', buffer)
                    if start:
                        value = buffer[start.end() :]
                        escaped = re.match(r'(?:[^"\\]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*', value).group()
                        decoded = json.loads('"' + escaped + '"')
                        if len(decoded) > sent:
                            yield decoded[sent:]
                            sent = len(decoded)
            if packet.get("done"):
                if json_mode and packet.get("done_reason") == "length":
                    raise OutputLimitReached("Ollama reached the output limit during a structured response.")
                return
        raise RuntimeError("Ollama closed the stream before the reply finished.")
    except (OSError, HTTPException) as error:
        raise RuntimeError("Could not reach local Ollama. Start Ollama and try again.") from error
    finally:
        connection.close()
