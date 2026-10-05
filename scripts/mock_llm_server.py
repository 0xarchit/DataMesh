import json
from http.server import BaseHTTPRequestHandler, HTTPServer

COUNTER = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"data": [{"id": "mock", "object": "model"}]})
        else:
            self._json({"ok": True})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        COUNTER["n"] += 1
        payload = {
            "vendor": "ACME",
            "receipt_no": "R-{:04d}".format(COUNTER["n"]),
            "date": "2025-03-15",
            "amount": 19900,
        }
        body = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(payload)}}],
        }
        self._json(body)

    def _json(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8099), Handler).serve_forever()
