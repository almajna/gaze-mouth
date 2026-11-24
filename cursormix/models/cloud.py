"""Text deltas from the selected provider. Keys stay in the process environment."""

import atexit
import base64
import json
import os
import ssl
from functools import cache

import httpx

from .errors import OutputLimitReached


@cache
def streaming_client():
    client = httpx.Client(
        verify=ssl.create_default_context(),
        timeout=25,
        limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=60),
    )
    atexit.register(client.close)
    return client


def output_schema(schema):
    # Both providers require closed objects in structured output. Validate
    # bounded list lengths again in the feature consuming each response.
    if isinstance(schema, list):
        return [output_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {
        key: output_schema(value) for key, value in schema.items() if key not in ("minItems", "maxItems")
    }
    if result.get("type") == "object":
        result["additionalProperties"] = False
    if "enum" in result and "type" not in result:
        result["type"] = "string"
    return result


def answer(prompt, provider, system, *, tier, image, limit, schema=None, require_complete=False):
    key_name = {"Anthropic": "ANTHROPIC_API_KEY", "OpenAI": "OPENAI_API_KEY"}[provider]
    key = os.getenv(key_name)
    if not key:
        raise RuntimeError(f"Set {key_name} in the environment used to start CursorMix.")
    payload = {"messages": [{"role": "user", "content": prompt}], "stream": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if provider == "Anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        payload.update(model=os.getenv("CLICKY_ANTHROPIC_MODEL", "claude-haiku-5-5"), max_tokens=1024)
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {key}"
        payload.update(model=os.getenv("CLICKY_OPENAI_MODEL", "gpt-4.1-mini"))
    message = payload["messages"][0]
    if provider == "Anthropic":
        payload["system"] = system
        payload["max_tokens"] = limit
        if tier != "fast":
            payload["model"] = os.getenv(f"CLICKY_ANTHROPIC_{tier.upper()}_MODEL", "claude-sonnet-5-5")
        if payload["model"] == "claude-sonnet-5-5":
            payload["thinking"] = {"type": "between_tools"}
        elif payload["model"] == "claude-haiku-5-5":
            # Adaptive thinking can consume a short reply budget before emitting text.
            payload["thinking"] = {"type": "disabled"}
        if schema:
            payload["output_config"] = {"format": {"type": "json_schema", "schema": output_schema(schema)}}
        if image:
            encoded = base64.b64encode(image).decode("ascii")
            message["content"] = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}},
                {"type": "text", "text": prompt},
            ]
    else:
        payload["messages"].insert(0, {"role": "system", "content": system})
        payload["max_completion_tokens"] = limit
        if tier != "fast":
            payload["model"] = os.getenv(f"CLICKY_OPENAI_{tier.upper()}_MODEL", "gpt-4.1")
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "clicky_response", "strict": True, "schema": output_schema(schema)},
            }
        if image:
            url_data = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
            message["content"] = [
                {"type": "image_url", "image_url": {"url": url_data}},
                {"type": "text", "text": prompt},
            ]
    ended = False
    truncated = False
    try:
        with streaming_client().stream("POST", url, json=payload, headers=headers) as response:
            if not response.is_success:
                raise RuntimeError(
                    f"{provider} returned HTTP {response.status_code}. Check your model, key, and quota."
                )
            for line in response.iter_lines():
                if ended or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    ended = True
                    continue
                packet = json.loads(data)
                if "error" in packet:
                    raise RuntimeError(f"{provider} interrupted the response.")
                if provider == "Anthropic":
                    if packet.get("type") == "message_stop":
                        ended = True
                        continue
                    delta = packet.get("delta", {})
                    if delta.get("stop_reason") == "max_tokens":
                        truncated = True
                    token = delta.get("text")
                else:
                    choices = packet.get("choices", [])
                    choice = choices[0] if choices else {}
                    if choice.get("finish_reason") == "length":
                        truncated = True
                    token = choice.get("delta", {}).get("content")
                if token:
                    yield token
        if not ended:
            raise RuntimeError(f"{provider} closed the stream early.")
        if require_complete and truncated:
            raise OutputLimitReached(f"{provider} reached the output limit before completing its response.")
    except httpx.RequestError as error:
        if not ended:
            raise RuntimeError(f"Could not reach {provider}. Check the connection and retry.") from error
