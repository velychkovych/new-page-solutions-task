"""One request to the mock, run inside the mock container:
`python -c <this file> METHOD PATH BODY [IDEMPOTENCY_KEY]`.

The runner passes this file's source to `docker compose exec -T mock python -c`. The tokens
stay in the container's environment: this prints only the status and the response body,
and the host process never sees a token. Standard library only, since it runs in the image.
"""

import json
import os
import sys
import urllib.error
import urllib.request

MOCK_URL = "http://127.0.0.1:8000"


def request(method: str, path: str, body: str, idempotency_key: str = "") -> dict[str, object]:
    headers = {"Authorization": "Bearer " + os.environ["MOCK_API_TOKEN"]}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    if path.startswith("/admin/"):
        headers["X-Admin-Token"] = os.environ["MOCK_ADMIN_TOKEN"]
    if body:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(  # noqa: S310 - a fixed loopback http URL
        MOCK_URL + path, data=body.encode() or None, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    return {"status": status, "body": json.loads(raw or b"null")}


if __name__ == "__main__":
    sys.stdout.write(json.dumps(request(*sys.argv[1:5])) + "\n")
