"""Python client for a deployed KumoFlow pipeline (deploy.py) — stdlib only,
so this single file can be copied into any project, no installs.

    from client import Pipeline

    client = Pipeline("localhost:8001")
    result = client.run(images={"image": "path/to/cat.jpg"})
    print(result)

`images` / `videos` map an input node id to a file path (or a list of paths
for a batch). A workflow with a single input node accepts any key — "image"
above just reads well. Pipeline.info() lists the deployed input node ids.

`result` is {target node id: result} exactly as the editor shows it: preview
URLs under "images", export zips under "download", classifications/counts as
plain JSON. Pipeline.fetch(url) downloads what a result points at.
"""

import json
import mimetypes
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class Pipeline:
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.url = (api_url if "://" in api_url else f"http://{api_url}").rstrip("/")

    def info(self) -> dict:
        """The deployed workflow's name, input node ids, and targets."""
        return self._request("/info")

    def run(self, images: dict | None = None, videos: dict | None = None) -> dict:
        """Run the pipeline -> {target node id: result}. No files = the deployed inputs."""
        files = []
        for key, paths in {**(images or {}), **(videos or {})}.items():
            for p in [paths] if isinstance(paths, (str, Path)) else paths:
                p = Path(p)
                ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                files.append((key, p.name, ctype, p.read_bytes()))
        body, headers = _multipart(files) if files else (None, {})
        return self._request("/run", method="POST", body=body, headers=headers)["results"]

    def fetch(self, media_url: str) -> bytes:
        """Download a /media/... URL a result points at (preview image, export zip)."""
        with urlopen(self.url + media_url) as r:
            return r.read()

    def _request(self, path, method="GET", body=None, headers=None):
        try:
            with urlopen(Request(self.url + path, data=body, headers=headers or {}, method=method)) as r:
                return json.load(r)
        except HTTPError as e:
            raise RuntimeError(f"pipeline API returned {e.code}: {e.read().decode(errors='replace')}") from None


def _multipart(files):
    """[(field, filename, content_type, bytes)] -> (body, headers)."""
    b = uuid.uuid4().hex
    body = b"".join(
        f'--{b}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n"
        for field, name, ctype, data in files
    ) + f"--{b}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={b}"}
