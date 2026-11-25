"""Exercise a running experimental audio server with an actual speech fixture.

No compiler or third-party Python packages are required. Requests are serial and
reset model session state. Do not run while someone is using the demo.
"""
import argparse
import base64
import http.client
import json
from pathlib import Path
import struct
import time
from urllib.parse import urlsplit
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("input_f32", type=Path, help="Mono float32 PCM at 16 kHz, within the compiled input limit")
    parser.add_argument("--output", type=Path, required=True, help="Validation report directory, outside the repository")
    parser.add_argument("--expected-text")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if url.scheme != "http" or url.path not in ("", "/"):
        parser.error("Use the server's http://host:port base URL")
    args.output.mkdir(parents=True, exist_ok=True)

    def open_request(path, body=None, content_type=None, headers=None):
        connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=60)
        request_headers = dict(headers or {})
        if content_type: request_headers["Content-Type"] = content_type
        connection.request("GET" if body is None else "POST", path, body, request_headers)
        return connection, connection.getresponse()

    def simple(path, body=None, content_type=None, expected=200, headers=None):
        connection, response = open_request(path, body, content_type, headers)
        data = response.read()
        connection.close()
        assert response.status == expected, (path, response.status, data[:200])
        return json.loads(data) if data else None

    health = simple("/api/health")
    assert not health["busy"], "Wait for the active request before testing"
    pcm = args.input_f32.read_bytes()
    assert pcm and len(pcm) % 4 == 0 and len(pcm) // 4 <= health["max_input_samples"]
    simple("/api/speech?id=empty", b"", "application/octet-stream", 400)
    simple("/api/speech?id=nan", struct.pack("<f", float("nan")), "application/octet-stream", 400)
    simple("/api/speech?id=oversized", b"\0" * (4 * (health["max_input_samples"] + 1)), "application/octet-stream", 413)
    simple("/api/text?id=origin", b'{"text":"hi"}', "application/json", 403, {"Origin": "http://unrelated.invalid"})
    simple("/api/text?id=bad-limit", b'{"text":"hi","max_steps":0}', "application/json", 400)
    simple("/api/text?id=fractional-limit", b'{"text":"hi","max_steps":1.5}', "application/json", 400)
    simple("/api/text?id=wrong-type", b'{"text":"hi"}', "text/plain", 415)

    # Read the start event before checking contention and cancellation.
    connection, response = open_request("/api/text?id=cancel-check",
        b'{"text":"Count from one to ten aloud.","max_steps":512}', "application/json")
    assert response.status == 200
    assert json.loads(response.readline())["type"] == "start"
    assert simple("/api/health")["busy"]
    simple("/api/text?id=concurrent", b'{"text":"hi"}', "application/json", 409)
    assert not simple("/api/stop?id=wrong-id", b"")["stopping"]
    stopped = time.monotonic()
    assert simple("/api/stop?id=cancel-check", b"")["stopping"]
    events = [json.loads(line) for line in response if line.strip()]
    connection.close()
    assert events[-1]["type"] == "done" and events[-1]["cancelled"], events
    cancellation_ms = (time.monotonic() - stopped) * 1000
    assert cancellation_ms < 5000, cancellation_ms
    assert not simple("/api/health")["busy"]

    connection, response = open_request("/api/text?id=disconnect-check",
        b'{"text":"Count from one to ten aloud.","max_steps":512}', "application/json")
    assert response.status == 200
    assert json.loads(response.readline())["type"] == "start"
    disconnected = time.monotonic()
    response.close(); connection.close()
    while simple("/api/health")["busy"]:
        assert time.monotonic() - disconnected < 5, "Disconnected request retained the session"
        time.sleep(.05)
    disconnect_ms = (time.monotonic() - disconnected) * 1000

    results = []
    for trial in range(2):
        started = time.monotonic()
        connection, response = open_request(f"/api/speech?id=speech-check-{trial}", pcm, "application/octet-stream")
        assert response.status == 200
        events, output = [], bytearray()
        first_pcm = None
        for line in response:
            event = json.loads(line)
            events.append(event)
            assert event["type"] != "error", event
            if event["type"] == "audio":
                if first_pcm is None: first_pcm = (time.monotonic() - started) * 1000
                assert event["sample_rate"] == 24000
                output.extend(base64.b64decode(event["pcm_s16le"], validate=True))
        connection.close()
        final = events[-1]
        assert final["type"] == "done" and not final["cancelled"]
        assert output and len(output) == final["samples"] * 2
        assert final["first_pcm_ms"] < final["request_ms"]
        if args.expected_text: assert final["text"] == args.expected_text, final["text"]
        final["client_first_pcm_ms"] = first_pcm
        final["client_complete_ms"] = (time.monotonic() - started) * 1000
        with wave.open(str(args.output / f"speech-{trial}.wav"), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000); wav.writeframes(output)
        (args.output / f"speech-{trial}.ndjson").write_text("".join(json.dumps(e) + "\n" for e in events))
        results.append(final)
    assert results[0]["text"] == results[1]["text"], "Independent greedy requests must reset state"
    assert (args.output / "speech-0.wav").read_bytes() == (args.output / "speech-1.wav").read_bytes()
    report = {"health": health, "protocol_checks": "passed", "cancellation_ms": cancellation_ms,
              "disconnect_cleanup_ms": disconnect_ms, "speech_requests": results}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"protocol_checks": "passed", "cancellation_ms": cancellation_ms, "disconnect_cleanup_ms": disconnect_ms,
                      "speech_requests": results}, indent=2))


if __name__ == "__main__":
    main()
