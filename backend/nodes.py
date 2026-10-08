"""KumoFlow nodes.

To add a node: subclass InputNode, TransformNode, or OutputNode below (or Node
directly for multi-port nodes), set `kind` (unique) and `label`, declare config
as pydantic fields (each with a default), list `inputs`/`outputs` port names if
they differ from the base, and implement `run`. The class registers itself and
shows up in the frontend palette automatically.

Dataflow — batches are first-class citizens. A batch is a length-B list (a
sequence), so per-image sizes may differ and a single image is just B=1:
  * image   : list length B of torch.Tensor (C, H, W), RGB, uint8 — H,W per image
  * masks   : list length B of torch.Tensor (N_i, H, W) uint8  — per-image instance masks
  * boxes   : list length B of torch.Tensor (N_i, 4) int32     — x1,y1,x2,y2 inclusive AABBs
  * embedding: list length B of torch.Tensor (D,) float32      — one vector per image
  * labels  : list length B of str ("" = unlabeled)             — per-image class labels
              SAM3 instead emits one name per *instance*: each batch item is a
              list aligned 1:1 with its boxes/masks (see _per_instance)
  * video   : list length B of torch.Tensor (T,3,H,W) uint8, RGB — clips; T,H,W per clip
  * ids     : per-frame track ids (see below), one int32 per box/mask; -1 = flicker
From a video segmentation the same `masks`/`boxes` ports carry per-frame
values: each batch item is a list length T of those per-image tensors (aligned
1:1 with its clip; identity across frames is what TrackObjects' `ids` add).
There is no separate video_masks port type — consumers tell the two forms
apart by structure (_per_frame).
Nodes whose op is per-frame take either `image` or `video` (exactly one wired)
and answer on the matching output port, keeping clip structure intact.
  * prompts : dict; drawn prompts are per-image (`per_image`, one set per image),
              a text concept is a single string SAM3 broadcasts across the batch
A length-1 image list broadcasts against length-B (see WarpOverlay). A stacked
(B,C,H,W) tensor is only ever a node-internal detail — HF processors take the
list as-is and batch/pad inside — never the edge contract. Only the final image
batch is serialized (to PNGs); tensors/lists flow between nodes for free.
Video keeps the same rule one level up: the batch unit is the clip (`video`,
one (T,3,H,W) tensor per clip — a folder of clips is a video dataset, one
video is B=1), and SampleFrames turns clips into a plain image batch so every
per-image node works on video after that one explicit step.

Each node declares `modalities` — which input kinds its output makes sense
for. Input nodes name the modality they produce (["image"] or ["video"]);
processing nodes default to both, and a future video-only node (e.g. a
tracker) sets ["video"]. The frontend only offers processing nodes whose
modalities overlap the input node(s) on the canvas.
"""

import base64
import os
import tempfile
import warnings
from contextvars import ContextVar
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

REGISTRY: dict[str, type["Node"]] = {}

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v", ".ts", ".flv", ".wmv", ".3gp", ".mpg", ".mpeg")
# directory filtering only — _decode_rgb sniffs content, so an odd extension on
# a directly-given file still decodes
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".ppm", ".pgm", ".tga", ".ico", ".dds", ".jp2", ".avif", ".heic", ".heif")

# Server-side files the frontend fetches by URL (GET /media/{name} in main.py):
# encoded previews, playback clips, export zips. Content-addressed names, under
# the OS tmpdir (cleaned by the OS, no eviction of our own). ponytail: grows
# until the OS clears tmp — add eviction if a long-lived deployment needs it.
MEDIA = Path(tempfile.gettempdir()) / "kumoflow_media"

# FLOW_SPILL_DIR override matters where the OS tmpdir is tmpfs ("disk" would
# actually be RAM+swap there) — point it at a real filesystem for long clips.
SPILL_DIR = Path(os.environ.get("FLOW_SPILL_DIR") or tempfile.gettempdir()) / "kumoflow_spill"


def _spill_free():
    """Free bytes on the spill filesystem."""
    import shutil

    SPILL_DIR.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(SPILL_DIR).free


def _spill_bytes_guard(nbytes, remedy):
    """Refuse allocations that don't fit on the spill filesystem: needs 1.25x
    headroom over `nbytes` (the mmap file plus in-flight writes)."""
    free = _spill_free()
    need = 1.25 * nbytes
    if need > free:
        raise ValueError(f"needs ~{need / 1e9:.1f} GB (~{free / 1e9:.1f} GB free disk space) — {remedy}")


class _SpillWriter:
    """Append-only writer into one temp file under SPILL_DIR; finish() maps it
    and returns tensors that view the mmap. The file is unlinked immediately
    after mapping — the mapping keeps the inode alive and the disk space frees
    itself when the last tensor dies (cache eviction, run end, crash). No
    tmp-cleaner races, no growth after process death."""

    def __init__(self):
        SPILL_DIR.mkdir(parents=True, exist_ok=True)
        self._fh = tempfile.NamedTemporaryFile(dir=SPILL_DIR, delete=False)
        self._path = Path(self._fh.name)
        self._records = []  # [(offset, shape), ...] in add() order
        self._offset = 0
        self._done = False

    def add(self, x):
        """Write one tensor/ndarray's raw uint8 bytes, record its shape."""
        import numpy as np
        import torch

        arr = x.detach().cpu().contiguous().numpy() if torch.is_tensor(x) else np.ascontiguousarray(x)
        assert arr.dtype == np.uint8
        self._fh.write(arr.tobytes())
        self._records.append((self._offset, arr.shape))
        self._offset += arr.nbytes

    def _map(self):
        import torch

        self._fh.close()
        self._done = True
        total = self._offset
        mapped = torch.from_file(str(self._path), shared=True, size=total, dtype=torch.uint8) if total else None
        self._path.unlink(missing_ok=True)
        return mapped

    def __del__(self):
        # a guard raising mid-decode abandons the writer before finish() runs
        # — without this the partial file leaks on disk forever (it's already
        # unlinked from any directory listing's perspective once finished, but
        # an abandoned one never got unlinked at all)
        if getattr(self, "_done", True):
            return
        try:
            self._fh.close()
        except Exception:
            pass
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    def finish(self):
        """-> list[torch.Tensor], one view per add() call, in order."""
        import torch

        mapped = self._map()
        if mapped is None:  # every record was zero-byte (or there were none)
            return [torch.zeros(shape, dtype=torch.uint8) for _, shape in self._records]
        out = []
        for offset, shape in self._records:
            n = 1
            for d in shape:
                n *= d
            out.append(mapped[offset : offset + n].view(shape))
        return out

    def finish_stacked(self, shape):
        """-> one tensor viewing the whole file (every add() must have had the
        same shape, i.e. this is `shape[0]` identical records)."""
        import torch

        mapped = self._map()
        return mapped.view(shape) if mapped is not None else torch.zeros(shape, dtype=torch.uint8)


# --- batch conversion helpers (the only place we bridge torch <-> numpy/opencv) ---


def _batch_from_hwc(imgs):
    """list of (H,W,3) RGB uint8 ndarrays -> list of (3,H,W) uint8 tensors."""
    import torch

    return [torch.from_numpy(im).permute(2, 0, 1).contiguous() for im in imgs]


def _batch_to_hwc(image):
    """list of (3,H,W) RGB uint8 tensors -> list of (H,W,3) RGB uint8 ndarrays."""
    return [x.permute(1, 2, 0).contiguous().cpu().numpy() for x in image]


def _image_or_video(image, video):
    """Dual-port nodes take `image` or `video` — never both, never neither.
    Returns the wired batch and its port name so results leave the same port."""
    if (image is None) == (video is None):
        raise ValueError("connect either an image or a video input (exactly one)")
    return (image, "image") if image is not None else (video, "video")


def _mem_available():
    """Available RAM in bytes (Linux MemAvailable); None where unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# --- progress + stop --------------------------------------------------------
# Nodes that take a while (models, tracking, video decode) say how much work
# they have and tick it off, so the UI draws a real bar and Stop takes effect
# between work units. The engine installs the run's slot here; nodes only ever
# call _total() once and _step() as they go — everything routed through
# _chunked() gets its ticks for free.
#
# A ContextVar, not a plain global: FastAPI runs each /run in its own
# threadpool context copy, so concurrent runs can't see each other's slot.


class Cancelled(RuntimeError):
    """The run was stopped from the UI."""


PROGRESS: ContextVar[dict | None] = ContextVar("progress", default=None)


def _total(n):
    """Declare the running node's work as `n` batch items — images, frames or
    clips, counted the way the canvas shows them."""
    if (p := PROGRESS.get()) is not None:
        p["nodes"][p["node"]] = [0, n]


def _step(k=1):
    """k items done — and the point where a stopped run actually stops:
    canvas-wide via the cancel flag, or mid-node when a per-target stop
    dropped the running node out of the run's live set (see main.RUNS)."""
    if (p := PROGRESS.get()) is not None:
        live = p.get("live")
        if p["cancel"] or (live is not None and p.get("node") not in live):
            raise Cancelled("stopped")
        p["nodes"].setdefault(p["node"], [0, 0])[0] += k


def _per_frame(vals):
    """masks/boxes come in two structural forms on the same port: per-image
    (each batch item a tensor) or per-frame from a video segmentation (each
    batch item a list of per-frame tensors). The value tells which."""
    import torch

    return bool(vals) and not torch.is_tensor(vals[0])


def _check_per_frame(video, vals, what):
    """Annotations wired alongside video must be per-frame and shaped like the
    clips they came from (same B, same T per clip) — catches per-image values
    and a different video than the one the model saw."""
    if not _per_frame(vals):
        raise ValueError(f"these {what} are per-image — wire SAM3's video output here")
    if len(vals) != len(video) or any(len(v) != len(c) for v, c in zip(vals, video)):
        raise ValueError(f"{what} don't line up with the clips — wire the same video into SAM3 and here")


def _frame_counts(vals):
    """[[instances per frame] per clip] of a per-frame boxes/labels/ids value.
    Two ports that came from the same node line up exactly here; a mismatch
    means two different SAM3s got wired in and the answer would be nonsense."""
    return [[len(x) for x in clip] for clip in vals]


def _check_per_image(vals, what):
    """The inverse: nodes that work per image reject per-frame annotations."""
    if vals is not None and _per_frame(vals):
        raise ValueError(f"these {what} are per-frame (video) — put a Sample Frames node in between")


def _per_instance(labels):
    """`labels` has two structural forms on the same port, like masks/boxes: one
    class label per batch item (a str, from Load Image / Load Video) or SAM3's
    per-instance concept names (a list, one name per box/mask)."""
    return bool(labels) and not isinstance(labels[0], str)


def _check_class_labels(labels):
    """Nodes that want one label per batch item reject SAM3's per-instance names."""
    if labels is not None and _per_instance(labels):
        raise ValueError(
            "these labels are SAM3's per-instance concept names — wire per-item labels "
            "(e.g. Load Image's folder labels) here instead"
        )


# Canonical port order, most-used first. spec() sorts every node's ports by
# it, so the same port sits at the same relative height on every node and
# edges between nodes don't cross (SAM3's boxes meets View Image's boxes).
PORT_ORDER = ["image", "video", "overlay", "prompts", "embedding", "classification", "boxes", "masks", "labels", "ids"]


def _port_sorted(names):
    return sorted(names, key=lambda p: PORT_ORDER.index(p) if p in PORT_ORDER else len(PORT_ORDER))


PREVIEW_SIDE = 512  # longest edge of a canvas preview — a node renders a few hundred px wide


def batch_to_previews(image, max_side=PREVIEW_SIDE):
    """list of (3,H,W) RGB uint8 tensors -> list of JPEG byte strings, one per
    image, downscaled so the long edge is at most `max_side`.

    Display size and lossy, not source size and lossless: a 1284x1168 photo
    costs 185 ms and 1 MB as a full-res PNG, so a 1100-image batch was ~3.5
    minutes of encoding and a 1.5 GB response — for pixels that never reach a
    screen. At 512 px JPEG it's 1 ms and 30 KB.

    Whatever gets drawn on a preview (Crop's box, a prompt node's boxes and
    points) is in *source* pixels, so the frontend can't measure the preview to
    map a click back — /run sends the true size alongside as `sizes`."""
    from torchvision.io import encode_jpeg
    from torchvision.transforms.v2.functional import resize

    out = []
    for frame in image:
        frame = frame.cpu()
        h, w = frame.shape[-2:]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            frame = resize(frame, [max(1, round(h * s)), max(1, round(w * s))])
        out.append(encode_jpeg(frame.contiguous(), quality=85).numpy().tobytes())
    return out


def _instances_to_boxes_masks(masks, h, w, names=()):
    """iterable of instance masks (H,W or 1,H,W) -> (Tensor(N,4) int32 AABBs,
    Tensor(N,H,W) uint8, list[str] of N names).

    Keeps boxes and masks aligned 1:1 and drops empty/degenerate masks. Accepts
    torch tensors or ndarrays. `names` (a concept name per input mask) is filtered
    by the same drops, so it stays aligned too; missing entries become "".
    """
    import numpy as np
    import torch

    bxs, ms, kept = [], [], []
    for j, m in enumerate(masks):
        arr = m.detach().cpu() if torch.is_tensor(m) else torch.as_tensor(np.asarray(m))
        b = arr > 0
        if b.ndim == 3:
            b = b[0]
        ys, xs = torch.nonzero(b, as_tuple=True)
        if xs.numel() == 0:
            continue
        ms.append(b.to(torch.uint8))
        bxs.append(torch.tensor([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], dtype=torch.int32))
        kept.append(names[j] if j < len(names) else "")
    boxes = torch.stack(bxs) if bxs else torch.zeros((0, 4), dtype=torch.int32)
    stacked = torch.stack(ms) if ms else torch.zeros((0, h, w), dtype=torch.uint8)
    return boxes, stacked, kept


def _track_windows(T, W):
    """[(start, end), ...] covering [0, T) in chunks of at most W frames, each
    overlapping the next by one frame (window k+1 starts at window k's last
    frame) so a tracked object can be carried across the seam."""
    wins, s = [], 0
    while True:
        e = min(s + W, T)
        wins.append((s, e))
        if e >= T:
            return wins
        s = e - 1


class Node(BaseModel):
    """Base for all nodes. Don't subclass directly unless you need custom ports.

    Results may be cached across runs and are shared between consumers — treat
    input values and returned values as immutable (no in-place edits)."""

    kind: ClassVar[str] = ""  # unique id; setting it registers the class
    # cross-run result cache (see evaluate() in main.py). False = this node's
    # output depends on more than kind+config+inputs (nondeterminism): it is
    # never cached and neither are any of its descendants.
    cacheable: ClassVar[bool] = True
    label: ClassVar[str] = ""
    # palette grouping (and its order, CAT_ORDER in App.svelte): set per base
    # class, overridden where a node's role differs from its ports — Count
    # Objects computes rather than displays ("Analyze"), Filter by Class is a
    # batch op that merely reads a classification ("Transform")
    category: ClassVar[str] = "Other"
    color: ClassVar[str] = "#2563eb"
    inputs: ClassVar[list[str]] = []  # input port names (param names of run)
    outputs: ClassVar[list[str]] = []  # output port names (keys of run's dict)
    required: ClassVar[list[str] | None] = None  # required input ports; None -> just inputs[:1]
    # input kinds this node is relevant for: input nodes name what they produce,
    # processing nodes what they can consume (default: anything)
    modalities: ClassVar[list[str]] = ["image", "video"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.kind:
            REGISTRY[cls.kind] = cls

    @model_validator(mode="before")
    @classmethod
    def _blank_means_default(cls, values):
        """The settings panel sends raw strings, so an emptied numeric field
        arrives as "" — treat that as "use the default" (Load Video: fps /
        max_frames default 0 = decode everything) instead of a type error."""
        if isinstance(values, dict):
            for k, v in list(values.items()):
                f = cls.model_fields.get(k)
                if v == "" and f is not None and isinstance(f.default, (int, float)) and not isinstance(f.default, bool):
                    del values[k]
        return values

    @classmethod
    def required_ports(cls):
        """Input ports that must be connected. Defaults to the first input only."""
        return cls.required if cls.required is not None else cls.inputs[:1]

    @classmethod
    def spec(cls):
        """What the frontend needs to render this node in the palette."""
        return {
            "kind": cls.kind,
            "label": cls.label,
            "doc": cls.__doc__,  # node semantics for the chat catalog (chat.py); frontend ignores it
            "category": cls.category,
            "color": cls.color,
            # sorted for display only — run()/required_ports keep the class order
            "inputs": _port_sorted(cls.inputs),
            "outputs": _port_sorted(cls.outputs),
            "required_inputs": cls.required_ports(),  # rest are optional (see evaluate)
            "modalities": cls.modalities,
            "config": {name: f.default for name, f in cls.model_fields.items()},
            # hyperparameter help (Field description) — the settings panel shows an info icon
            "config_info": {name: f.description for name, f in cls.model_fields.items() if f.description},
            # dropdown presets (Field json_schema_extra "options": plain strings or
            # {value, label} dicts) — the settings panel renders a select; dict
            # options also get a Custom… free-text escape hatch
            "config_options": {
                name: f.json_schema_extra["options"]
                for name, f in cls.model_fields.items()
                if isinstance(f.json_schema_extra, dict) and "options" in f.json_schema_extra
            },
            # HF Hub pipeline tags (Field json_schema_extra "hf_search") — the
            # settings panel's Custom… input live-searches the Hub filtered to
            # these, so any compatible Hub model is one click away
            "config_hf": {
                name: f.json_schema_extra["hf_search"]
                for name, f in cls.model_fields.items()
                if isinstance(f.json_schema_extra, dict) and "hf_search" in f.json_schema_extra
            },
        }

    def run(self, **inputs):
        raise NotImplementedError

    def cache_extra(self) -> str:
        """Extra cache-key material for results that depend on state outside
        config+inputs (e.g. file mtimes — see LoadVideo). Must be cheap (stat
        calls, not reads) and must not raise on missing files — return
        something that changes instead."""
        return ""


class InputNode(Node):
    """Produces an image batch from its config. Implement `run(self) -> list[Tensor(C,H,W)]`."""

    category: ClassVar[str] = "Input"
    color: ClassVar[str] = "#1db5b0"
    inputs: ClassVar[list[str]] = []
    outputs: ClassVar[list[str]] = ["image"]


class TransformNode(Node):
    """Turns an image batch into another. Implement `run(self, image) -> list[Tensor(C,H,W)]`."""

    category: ClassVar[str] = "Transform"
    color: ClassVar[str] = "#2563eb"
    inputs: ClassVar[list[str]] = ["image"]
    outputs: ClassVar[list[str]] = ["image"]


class OutputNode(Node):
    """Consumes an image batch and returns the batch the frontend shows."""

    category: ClassVar[str] = "Output"
    color: ClassVar[str] = "#9333ea"
    inputs: ClassVar[list[str]] = ["image"]
    outputs: ClassVar[list[str]] = []

    def run(self, image):
        return image


# --- concrete nodes ---------------------------------------------------------


def _decode_rgb(raw):
    """Any image bytes -> (3,H,W) RGB uint8. torchvision decodes the common web
    formats (JPEG/PNG/GIF/WEBP) without a PIL->tensor copy; everything else
    (BMP, TIFF, PPM/PGM, TGA, ICO, DDS, JPEG2000, AVIF, HEIC, ...) falls back
    to Pillow. Both paths apply EXIF orientation — phone photos arrive rotated."""
    import io

    import numpy as np
    import torch
    from torchvision.io import ImageReadMode, decode_image

    try:
        return decode_image(torch.frombuffer(bytearray(raw), dtype=torch.uint8), mode=ImageReadMode.RGB, apply_exif_orientation=True)
    except Exception:
        pass
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener

    register_heif_opener()  # idempotent; PIL alone can't identify HEIC
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    except Exception:
        raise ValueError("could not decode an image — unsupported or corrupt format") from None
    if img.mode in ("I", "F") or img.mode.startswith("I;16"):
        # 16-bit/float images: PIL's convert("RGB") clips to white instead of scaling
        arr = np.asarray(img, dtype=np.float32)
        img = Image.fromarray((arr / max(float(arr.max()), 1e-6) * 255).astype(np.uint8))
    return torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).contiguous()


class LoadImage(InputNode):
    """Load one image, a multi-select, or a folder as a batch. A
    class-per-subfolder folder also fills per-image `labels` (wire into Train
    Classifier)."""

    kind = "load"
    label = "Load Image"
    outputs = ["image", "labels"]
    modalities = ["image"]

    # Browser upload: one "data:image/...;base64,..." URL or a list of them (a
    # folder / multi-select). A single file stays B=1.
    # Alternatively a filesystem path (file or directory) on the backend host,
    # same as Load Video — agents driving the MCP server have local files, not
    # base64.
    data: str | list[str] = ""
    # Per-image class labels, aligned with `data`. The frontend fills this from
    # a class-per-subfolder dataset layout (PetImages/Cat/1.jpg -> "Cat") when a
    # folder is picked; "" = unlabeled. Wire the `labels` port into e.g. LogReg.
    labels: list[str] = []

    def cache_extra(self):
        """Filesystem-path mode: editing an image on disk must invalidate the
        cache, so the sig covers each file's (path, mtime_ns, size). Data-URL
        mode returns "" — the content is already in config."""
        if not isinstance(self.data, str) or not self.data or self.data.startswith("data:"):
            return ""
        return _stat_sig(self._paths())

    def _paths(self):
        """The file list a filesystem-path `data` resolves to (dir -> sorted images)."""
        from pathlib import Path

        root = Path(self.data).expanduser()
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if root.is_dir() else [root]

    def run(self):
        from pathlib import Path

        if not self.data:
            raise ValueError("choose a file or folder, or type a path, first")
        if isinstance(self.data, str) and not self.data.startswith("data:"):
            root = Path(self.data).expanduser()
            paths = self._paths()
            if not paths or not paths[0].is_file():
                raise ValueError(f"no images found at {self.data}")
            # class-per-subfolder labels, same layout as the browser folder pick
            labels = [p.parent.name if root.is_dir() and p.parent != root else "" for p in paths]
            return {"image": [_decode_rgb(p.read_bytes()) for p in paths], "labels": labels}

        def _from_data_url(u):
            if not u.startswith("data:"):  # list items can be paths too
                return _decode_rgb(Path(u).expanduser().read_bytes())
            return _decode_rgb(base64.b64decode(u.split(",", 1)[-1]))

        urls = self.data if isinstance(self.data, list) else [self.data]
        labels = [str(s) for s in self.labels[: len(urls)]]
        labels += [""] * (len(urls) - len(labels))  # missing tail = unlabeled
        return {"image": [_from_data_url(u) for u in urls], "labels": labels}


def _stat_sig(paths):
    """cache_extra material for filesystem-path data: (path, mtime_ns, size)
    per file — cheap stat calls, missing files change the sig instead of raising."""
    stats = []
    for p in paths:
        try:
            s = p.stat()
            stats.append((str(p), s.st_mtime_ns, s.st_size))
        except OSError:
            stats.append((str(p), "missing"))
    return repr(stats)


class LoadVideo(InputNode):
    """Load video files; `video` is a length-B list of (T,3,H,W) RGB uint8
    clips (T, H, W may differ per clip) — one file is B=1, a folder of clips
    is a video dataset. Wire a Sample Frames node after it to turn clips into
    an image batch for the per-frame nodes.

    Clips are stored on disk (mmap, see _SpillWriter), so ~30-60 min videos
    fit — but model inference downstream still visits every decoded frame
    (a 60-min clip at 5 fps is 18k SAM3 forwards): `fps` and `max_frames`
    are the knobs that bound that work."""

    kind = "load_video"
    label = "Load Video"
    outputs = ["video", "labels"]
    modalities = ["video"]

    # Browser upload: one "data:video/...;base64,..." URL or a list of them (a
    # folder / multi-select). A class-per-subfolder layout (UCF101-style,
    # actions/Walk/1.mp4 -> "Walk") fills `labels` with one class per clip,
    # exactly like Load Image's folder labels; "" = unlabeled.
    # Alternatively a filesystem path (file or directory) on the backend host —
    # browser upload ships videos as base64 JSON, which breaks past ~400MB
    # (V8's max string length), so big local datasets go by path instead.
    data: str | list[str] = ""
    labels: list[str] = []
    fps: float = Field(0, description="frames decoded per second of video; empty or 0 = every frame at the native rate")
    # the disk guard in _capture catches decodes that can't fit
    max_frames: int = Field(0, description="per-clip frame cap, spread across the whole clip; empty or 0 = the whole clip")
    # Decoded 4K is ~27MB/frame and every consumer downscales anyway (SAM3 to
    # ~1024, SigLIP/DINOv2 to 224, webm previews to 512), so carrying the
    # original is pure RAM waste.
    max_side: int = Field(1024, description="longest frame side after decode; 0 = original resolution")

    def cache_extra(self):
        """Filesystem-path mode: editing a video on disk must invalidate the
        cache, so the sig covers each file's (path, mtime_ns, size). Data-URL
        mode returns "" — the content is already in config."""
        if not isinstance(self.data, str) or not self.data or self.data.startswith("data:"):
            return ""
        return _stat_sig(self._paths())

    def _paths(self):
        """The file list a filesystem-path `data` resolves to (dir -> sorted videos)."""
        from pathlib import Path

        root = Path(self.data).expanduser()
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS) if root.is_dir() else [root]

    def _decode(self, url):
        """One data URL or file path -> a (T,3,H,W) uint8 clip tensor, sampled at self.fps."""
        import tempfile

        if not url.startswith("data:"):
            return self._capture(url)
        raw = base64.b64decode(url.split(",", 1)[-1])
        # cv2 can't decode from memory; ffmpeg sniffs the container from the
        # bytes, so the suffix doesn't need to match the actual format
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(raw)
            f.flush()
            return self._capture(f.name)

    def _decode_all(self, urls):
        """Decode one clip per url, reporting progress — a folder of videos is
        minutes of decoding, and the only work this node does."""
        _total(len(urls))
        clips = []
        for u in urls:
            clips.append(self._decode(u))
            _step()
        return clips

    @staticmethod
    def _grab_count(path):
        """True frame count by demuxing without decoding (grab-only)."""
        import cv2

        cap = cv2.VideoCapture(path)
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        return n

    def _capture_av(self, path):
        """Fallback when cv2 can't open a file: PyAV bundles a fuller ffmpeg
        (AV1, exotic containers). Same sampling semantics as _capture, except
        max_frames caps from the head — no reliable total without a second pass."""
        import av
        import cv2

        try:
            container = av.open(path)
        except Exception:
            raise ValueError("could not decode a video — unsupported or corrupt format") from None
        remedy = "lower fps, max_frames, or max_side on Load Video"
        writer, next_t, T, h, w = _SpillWriter(), 0.0, 0, 0, 0
        with container:
            for frame in container.decode(video=0):
                t = frame.time or 0.0
                if self.fps > 0 and t < next_t:
                    continue
                next_t = t + (1 / self.fps if self.fps > 0 else 0)
                rgb = frame.to_ndarray(format="rgb24")
                if self.max_side and max(rgb.shape[:2]) > self.max_side:
                    s = self.max_side / max(rgb.shape[:2])
                    rgb = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
                h, w = rgb.shape[:2]
                writer.add(rgb.transpose(2, 0, 1))
                T += 1
                _spill_bytes_guard(T * rgb.nbytes, remedy)  # disk guard, same idea as _capture
                if self.max_frames and T >= self.max_frames:
                    break
        if T == 0:
            raise ValueError("no frames decoded from a video")
        return writer.finish_stacked((T, 3, h, w))

    def _capture(self, path):
        import cv2
        import torch

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return self._capture_av(path)  # cv2's ffmpeg lacks e.g. AV1 — PyAV's is fuller
        native = cap.get(cv2.CAP_PROP_FPS)
        native = native if 0 < native < 1000 else 30.0  # matroska headers often lack or garble it
        step = max(1, round(native / self.fps)) if self.fps > 0 else 1
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with open(path, "rb") as f:
            is_matroska = f.read(4) == b"\x1a\x45\xdf\xa3"  # EBML magic — sniffed, not suffix: base64 uploads get a .mp4 temp name
        if is_matroska:
            # matroska frame counts are notoriously wrong (missing, or duration
            # in timebase units) — a bogus huge count would falsely trip the
            # RAM guard below. Recount by demuxing (grab-only, cheap), and only
            # when max_frames needs the total to spread its samples; otherwise
            # stream-decode with the incremental guard.
            count = self._grab_count(path) if self.max_frames else 0
        idx = range(0, count, step)
        if self.max_frames and len(idx) > self.max_frames:
            # the cap spreads across the whole clip (like Sample Frames) instead
            # of truncating the head — the event you care about may be at the end
            idx = [idx[j] for j in torch.linspace(0, len(idx) - 1, self.max_frames).round().long().tolist()]

        # Whole-video decode at native rate is the default, so refuse decodes
        # that can't fit on disk instead of filling the spill filesystem to death.
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = self.max_side / max(w, h, 1) if self.max_side and max(w, h) > self.max_side else 1
        frame_bytes = round(w * scale) * round(h * scale) * 3  # what a stored (downscaled) frame costs
        remedy = "lower fps, max_frames, or max_side on Load Video"

        def check_fit(n):
            _spill_bytes_guard(n * frame_bytes, remedy)

        check_fit(len(idx))
        # ponytail: a header without a frame count (rare) falls back to head-first
        # sampling capped at max_frames — better than a counting decode pass
        wanted = set(idx) or None
        writer, T, hh, ww, i = _SpillWriter(), 0, 0, 0, 0
        while (i <= idx[-1]) if wanted else not (self.max_frames and T >= self.max_frames):
            if not cap.grab():  # grab-only on skipped frames: no color convert/copy
                break
            if (i in wanted) if wanted else (i % step == 0):
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if self.max_side and max(frame.shape[:2]) > self.max_side:
                    # downscale per frame, from the frame's own shape (headers lie);
                    # peak RAM stays one full-res frame, not the whole clip
                    s = self.max_side / max(frame.shape[:2])
                    size = (round(frame.shape[1] * s), round(frame.shape[0] * s))
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hh, ww = rgb.shape[:2]
                writer.add(rgb.transpose(2, 0, 1))
                T += 1
                check_fit(T)  # the unknown-count path grows unbounded without this
            i += 1
        cap.release()
        if T == 0:
            raise ValueError("no frames decoded from a video")
        return writer.finish_stacked((T, 3, hh, ww))

    def run(self):
        from pathlib import Path

        if not self.data:
            raise ValueError("choose a video file or folder, or type a path, first")
        if isinstance(self.data, str) and not self.data.startswith("data:"):
            root = Path(self.data).expanduser()
            paths = self._paths()
            if not paths or not paths[0].is_file():
                raise ValueError(f"no videos found at {self.data}")
            # class-per-subfolder labels, same layout as the browser folder pick
            labels = [p.parent.name if root.is_dir() and p.parent != root else "" for p in paths]
            return {"video": self._decode_all([str(p) for p in paths]), "labels": labels}
        urls = self.data if isinstance(self.data, list) else [self.data]
        labels = [str(s) for s in self.labels[: len(urls)]]
        labels += [""] * (len(urls) - len(labels))  # missing tail = unlabeled
        return {"video": self._decode_all(urls), "labels": labels}


def _to_uint8(arr):
    """Min-max scale any numeric array to uint8 (flat input -> zeros)."""
    import numpy as np

    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros(arr.shape, np.uint8)


def _slice_pos(ds):
    """Sort key for slices in a DICOM series: position along the slice normal
    (never filename order); InstanceNumber when position tags are missing."""
    import numpy as np

    try:
        iop = np.array(ds.ImageOrientationPatient, float)
        return float(np.dot(np.cross(iop[:3], iop[3:]), np.array(ds.ImagePositionPatient, float)))
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0) or 0)


def _is_nifti(path):
    """Content-sniffed (magic at offset 344, behind gzip if present) — uploads
    arrive uuid-renamed, so extensions prove nothing here."""
    import gzip

    try:
        with open(path, "rb") as f:
            gz = f.read(2) == b"\x1f\x8b"
        with (gzip.open if gz else open)(path, "rb") as f:
            return f.read(348)[344:348] in (b"n+1\x00", b"ni1\x00")
    except OSError:
        return False


class LoadVolume(InputNode):
    """Load medical images — DICOM (.dcm/.ima/extensionless, single files or
    whole series) and NIfTI (.nii/.nii.gz) — as clips on the `video` port: one
    (T,3,H,W) uint8 clip per volume with slices as frames, so Sample Frames
    and the whole image toolchain work downstream. A folder groups DICOM files
    into one clip per series (slices sorted along the scan axis); a multi-frame
    DICOM or a NIfTI file is a clip on its own, and a single 2D DICOM is a
    1-frame clip. `labels` carries the DICOM series description (or a NIfTI
    dataset's class subfolder), "" when absent."""

    kind = "load_volume"
    label = "Load Volume"
    outputs = ["video", "labels"]
    modalities = ["video"]

    # Server paths from the browser upload (multipart, like Load Video) or a
    # filesystem path (file or directory) typed on the backend host.
    data: str | list[str] = ""
    labels: list[str] = []  # per-file folder labels from the browser pick, used for NIfTI files
    window_center: float = Field(0, description="intensity window center in HU (CT presets: brain 40, lung -600, bone 400, soft tissue 50); width 0 = auto")
    window_width: float = Field(0, description="intensity window width in HU (CT presets: brain 80, lung 1500, bone 1800, soft tissue 350); 0 = auto (DICOM window tags, else 0.5–99.5 percentiles)")

    def cache_extra(self):
        """Typed-path mode: editing files on disk must invalidate the cache.
        Uploads (immutable uuid paths / data URLs already in config) need nothing."""
        if not isinstance(self.data, str) or not self.data or self.data.startswith("data:"):
            return ""
        return _stat_sig(self._paths())

    @staticmethod
    def _spill(url):
        """data URL -> a content-addressed temp file, so the same sniffing and
        loading path serves deploy.py's /run uploads (which arrive as data URLs)."""
        import hashlib

        raw = base64.b64decode(url.split(",", 1)[-1])
        up = Path(tempfile.gettempdir()) / "kumoflow_uploads"
        up.mkdir(exist_ok=True)
        p = up / f"vol_{hashlib.sha1(raw).hexdigest()[:20]}"
        if not p.exists():
            p.write_bytes(raw)
        return p

    def _paths(self):
        if isinstance(self.data, list):
            return [self._spill(d) if d.startswith("data:") else Path(d) for d in self.data]
        if self.data.startswith("data:"):
            return [self._spill(self.data)]
        root = Path(self.data).expanduser()
        return sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]

    def _intensity(self, arr, ds=None):
        """Windowed float array -> what _to_uint8 scales: explicit window >
        DICOM window tags > robust percentiles (CT air/bone tails wash out min-max)."""
        import numpy as np

        if self.window_width > 0:
            lo = self.window_center - self.window_width / 2
            return np.clip(arr, lo, lo + self.window_width)
        if ds is not None and ("WindowCenter" in ds or "VOILUTSequence" in ds):
            from pydicom.pixels import apply_voi_lut

            return apply_voi_lut(arr, ds, index=0)
        return np.clip(arr, *np.percentile(arr, (0.5, 99.5)))

    def _series_clip(self, dss):
        """One sorted DICOM series -> (T,H,W,3) uint8. Windowing and scaling
        run on the whole volume, not per slice — per-slice scaling would give
        every slice its own brightness."""
        import numpy as np
        from pydicom.pixels import apply_modality_lut

        def frames(ds):  # (H,W...) single-slice or (N,H,W...) multi-frame -> (N,H,W...)
            arr = ds.pixel_array  # color data arrives RGB already (pydicom converts YBR)
            return arr[None] if arr.ndim == 2 + (int(getattr(ds, "SamplesPerPixel", 1)) == 3) else arr

        first = dss[0]
        if int(getattr(first, "SamplesPerPixel", 1)) == 3:
            return np.concatenate([frames(ds) for ds in dss]).astype(np.uint8)
        vol = np.concatenate([apply_modality_lut(frames(ds), ds) for ds in dss])
        u8 = _to_uint8(self._intensity(vol, first))
        if getattr(first, "PhotometricInterpretation", "") == "MONOCHROME1":
            u8 = 255 - u8
        return np.stack([u8] * 3, axis=-1)

    def _nii_clip(self, path):
        """One NIfTI file -> (T,H,W,3) uint8, axial slices in canonical (RAS) order."""
        import gzip

        import nibabel as nib
        import numpy as np

        raw = path.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        # from_bytes, not nib.load: uploads are uuid-renamed and nib.load trusts extensions
        img = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(raw))  # so "axial" means the same for every file
        vol = img.get_fdata(dtype=np.float32)  # applies scl_slope/inter
        if vol.ndim == 4:
            vol = vol[..., 0]  # ponytail: fMRI time series -> first timepoint; expose t if it matters
        vol = np.atleast_3d(vol)
        u8 = _to_uint8(self._intensity(vol))
        # RAS axis 2 = inferior->superior; rot90 turns each slice upright
        return np.stack([np.stack([np.rot90(u8[:, :, k])] * 3, axis=-1) for k in range(u8.shape[2])])

    def run(self):
        import numpy as np
        import torch
        from pydicom import dcmread
        from pydicom.misc import is_dicom

        if not self.data:
            raise ValueError("choose DICOM/NIfTI files or a folder, or type a path, first")
        paths = [p for p in self._paths() if p.is_file()]
        # both sniffed by content — extensionless DICOM is common, uploads are uuid-renamed
        sniffed = [(p, _is_nifti(p)) for p in paths]
        nii = [(i, p) for i, (p, isn) in enumerate(sniffed) if isn]
        dcm = [p for p, isn in sniffed if not isn and is_dicom(p)]
        if not nii and not dcm:
            raise ValueError("no DICOM or NIfTI files found")
        _total(len(dcm) + len(nii))

        def as_clip(nhwc):
            return torch.from_numpy(np.ascontiguousarray(nhwc)).permute(0, 3, 1, 2).contiguous()

        series = {}  # one clip per DICOM series, whatever the file layout
        for p in dcm:
            ds = dcmread(p)
            if "PixelData" in ds:  # skips DICOMDIR, reports, presentation states
                series.setdefault(str(getattr(ds, "SeriesInstanceUID", p)), []).append(ds)
            _step()
        clips, labels = [], []
        for dss in series.values():
            dss.sort(key=_slice_pos)
            clips.append(as_clip(self._series_clip(dss)))
            labels.append(str(getattr(dss[0], "SeriesDescription", "") or ""))
        root = Path(self.data).expanduser() if isinstance(self.data, str) else None
        for i, p in nii:
            clips.append(as_clip(self._nii_clip(p)))
            # class label: subfolder in typed-path mode, the browser's folder labels for uploads
            own = p.parent.name if root is not None and root.is_dir() and p.parent != root else ""
            labels.append(own or (self.labels[i] if i < len(self.labels) else ""))
            _step()
        return {"video": clips, "labels": labels}


class LoadStream(InputNode):
    """Grab a window of frames from a live source — a webcam on the backend
    host (device index, "0" = /dev/video0) or a network stream URL (rtsp://,
    http://, incl. MJPEG) — and emit it as one clip (B=1) on the `video` port.
    Each run captures `duration` seconds at `fps`; re-run for a fresh window,
    so the graph stays a batch pipeline."""

    kind = "load_stream"
    label = "Load Stream"
    outputs = ["video"]
    modalities = ["video"]
    cacheable = False  # a live source: every run is a new capture

    source: str = Field("0", description="camera device index (0 = /dev/video0) or a stream URL (rtsp://, http://)")
    duration: float = Field(3, description="seconds to capture per run")
    fps: float = Field(0, description="frames kept per second; empty or 0 = every frame the source delivers")
    max_side: int = Field(1024, description="longest frame side; 0 = original resolution")

    def run(self):
        import os
        import time

        import cv2

        if self.duration <= 0:
            raise ValueError("set a capture duration (seconds)")
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")  # UDP loss = gray smear
        src = self.source.strip()
        if src.isdigit():
            cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # deliver current frames, not a backlog
        else:
            # timeouts are open-only props (constructor params, .set() is too late) —
            # without them a dead RTSP host blocks for ~30s+
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        if not cap.isOpened():
            raise ValueError(f"could not open stream source {self.source!r} — device index or rtsp/http URL?")
        remedy = "lower duration, fps, or max_side on Load Stream"
        writer, misses, T, h, w = _SpillWriter(), 0, 0, 0, 0
        interval = 1 / self.fps if self.fps > 0 else 0.0
        deadline = time.monotonic() + self.duration
        next_keep = 0.0
        try:
            while time.monotonic() < deadline and misses < 5:
                if not cap.grab():  # grab-only keeps draining the buffer between kept frames
                    misses += 1
                    continue
                misses = 0
                now = time.monotonic()
                if now < next_keep:
                    continue  # ahead of the target fps — frame dropped undecoded
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                if self.max_side and max(frame.shape[:2]) > self.max_side:
                    s = self.max_side / max(frame.shape[:2])
                    frame = cv2.resize(frame, (round(frame.shape[1] * s), round(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                writer.add(rgb.transpose(2, 0, 1))
                T += 1
                _spill_bytes_guard(T * rgb.nbytes, remedy)
                next_keep = now + interval
        finally:
            cap.release()
        if T == 0:
            raise ValueError(f"no frames captured from {self.source!r} — is the camera/stream live?")
        return {"video": [writer.finish_stacked((T, 3, h, w))]}


class SampleFrames(Node):
    """Turn a clip batch (`video`) into an image batch: `per_clip` evenly
    spaced frames from every clip (0 = all frames), concatenated in clip
    order — a single clip with per_clip=0 makes the whole video the batch.
    Optionally wire Load Video's `labels` through: each clip's label repeats
    once per emitted frame, so per-frame nodes downstream stay aligned.
    Optionally wire SAM3's per-frame `masks` / `boxes` in alongside: the same
    sampled frames come out as per-image `masks` / `boxes`, so the whole image
    toolchain (View, Refine Masks, Blur, Warp Overlay, Crop) works on video
    frames."""

    kind = "sample_frames"
    label = "Sample Frames"
    category = "Transform"
    color = "#2563eb"
    modalities = ["video"]
    inputs = ["video", "labels", "masks", "boxes"]  # all but video optional
    outputs = ["image", "labels", "masks", "boxes"]

    per_clip: int = 0  # frames taken per clip; 0 = every frame

    def run(self, video, labels=None, masks=None, boxes=None):
        import torch

        _check_class_labels(labels)
        for what, vals in (("masks", masks), ("boxes", boxes)):
            if vals is not None:
                _check_per_frame(video, vals, what)
        images, out_labels, out_masks, out_boxes = [], [], [], []
        for i, clip in enumerate(video):
            n = min(self.per_clip, len(clip)) if self.per_clip > 0 else len(clip)
            idx = torch.linspace(0, len(clip) - 1, n).round().long().tolist()
            images += [clip[j] for j in idx]
            out_labels += [labels[i] if labels is not None and i < len(labels) else ""] * n
            if masks is not None:
                out_masks += [masks[i][j] for j in idx]
            if boxes is not None:
                out_boxes += [boxes[i][j] for j in idx]
        out = {"image": images, "labels": out_labels}
        # only claim the masks/boxes outputs when they were wired in
        if masks is not None:
            out["masks"] = out_masks
        if boxes is not None:
            out["boxes"] = out_boxes
        return out


class Blur(TransformNode):
    """Blur the batch — whole frames, or only *inside* wired `masks`
    (anonymize what a segmenter found). To blur the background instead,
    invert the masks first (Refine Masks)."""

    kind = "blur"
    label = "Blur"
    inputs = ["image", "video", "masks"]  # masks optional; per-image or per-frame, matching the input kind
    required = []  # image or video, exactly one, checked in run
    outputs = ["image", "video"]

    blur: int = 25  # kernel size; fixed (no random variation), odd and >= 3

    def run(self, image=None, video=None, masks=None):
        import cv2

        batch, port = _image_or_video(image, video)
        if port == "image":
            _check_per_image(masks, "masks")

        def blur_one(im, mask_set):
            """HWC frame + optional (N,H,W) masks: blur all, or only inside the union."""
            blurred = cv2.blur(im, (self.blur, self.blur))
            if mask_set is None:
                return blurred
            union = (mask_set > 0).any(dim=0).cpu().numpy()  # (H,W); empty -> all False -> no blur
            res = im.copy()
            res[union] = blurred[union]
            return res

        if port == "video":  # e.g. anonymization: blur what SAM3 found, frame by frame
            if masks is not None:
                _check_per_frame(batch, masks, "masks")
            out = []
            for ci, clip in enumerate(batch):
                fm = masks[ci] if masks is not None else None
                writer, h, w = _SpillWriter(), clip.shape[-2], clip.shape[-1]
                T = 0
                for t, f in enumerate(_batch_to_hwc(clip)):
                    writer.add(blur_one(f, fm[t] if fm is not None else None).transpose(2, 0, 1))
                    T += 1
                out.append(writer.finish_stacked((T, 3, h, w)))
            return {"video": out}
        out = [blur_one(im, masks[i] if masks is not None else None) for i, im in enumerate(_batch_to_hwc(batch))]
        return {"image": _batch_from_hwc(out)}


class Flip(TransformNode):
    """Mirror images or clips horizontally or vertically."""

    kind = "flip"
    label = "Flip"
    inputs = ["image", "video"]
    required = []  # exactly one of the two, checked in run
    outputs = ["image", "video"]

    direction: str = Field("horizontal", json_schema_extra={"options": ["horizontal", "vertical"]})

    def run(self, image=None, video=None):
        import torch

        batch, port = _image_or_video(image, video)
        dim = -2 if self.direction == "vertical" else -1  # (...,H,W): same op for images and clips
        if port == "image":
            return {"image": [torch.flip(x, dims=[dim]) for x in batch]}
        out = []
        for clip in batch:
            T, h, w = clip.shape[0], clip.shape[-2], clip.shape[-1]
            writer = _SpillWriter()
            for i in range(0, T, 32):  # chunked: peak RAM is one chunk, not the whole clip
                writer.add(torch.flip(clip[i : i + 32], dims=[dim]))
            out.append(writer.finish_stacked((T, 3, h, w)))
        return {"video": out}


class Reorient(TransformNode):
    """Reslice volume clips onto another anatomical plane: the output's frames
    are exactly the slices View Volume shows for that axis (axial = identity).
    Prediction plane is a graph decision — put this before a model so a 2D
    model segments coronal/sagittal cross-sections; its per-frame masks then
    line up with the reoriented clip. Masks made upstream of a Reorient don't
    carry through it."""

    kind = "reorient"
    label = "Reorient"
    inputs = ["video"]
    outputs = ["video"]
    modalities = ["video"]

    plane: str = Field("axial", json_schema_extra={"options": ["axial", "coronal", "sagittal"]})

    def run(self, video):
        if self.plane == "axial":
            return {"video": video}
        # (T,3,H,W): new frame axis = old H (coronal) / old W (sagittal); rows of
        # each new frame run along old T, flipped so superior stays on top —
        # matches VolumeView's reslice, so reorient(p) viewed axially == viewer's p
        dims = (2, 1, 0, 3) if self.plane == "coronal" else (3, 1, 0, 2)
        out = []
        for clip in video:
            view = clip.permute(*dims).flip(2)  # no copy yet — each output frame is a strided read of the input
            T, h, w = view.shape[0], view.shape[-2], view.shape[-1]
            writer = _SpillWriter()
            for i in range(0, T, 32):  # chunked materialize: peak RAM is one chunk
                writer.add(view[i : i + 32].contiguous())
            out.append(writer.finish_stacked((T, 3, h, w)))
        return {"video": out}


class Crop(TransformNode):
    """Crop each image (or every frame of each clip) to a bounding box. Takes
    the boxes from the optional `boxes` input (e.g. from SAM3; per-image, so
    image input only) or, when nothing is wired in, the box drawn on the node in
    the frontend (stored as "x1,y1,x2,y2" pixels) applied to every item. Coords
    are inclusive, matching SAM3's boxes and View's overlay.

    Every box becomes its own crop, so the batch that comes out is one image per
    detection (SAM3 finding 4 objects in 2 images gives 8 crops, in image then
    box order) — that is what makes "identify then classify" classify all of
    them, not just the first. Wired boxes decide alone: an image the segmenter
    found nothing in contributes no crops, whatever the drawn box says.

    Optional `labels` follow the crops so a labeled dataset survives the crop:
    a per-item class label (Load Image's folder label) is repeated once per crop
    from that image, and SAM3's per-instance concept names — already 1:1 with
    the boxes — become one label per crop. Crop is the only node that turns
    per-instance names into per-item labels; after the crop one instance *is*
    one batch item, so the crops are directly trainable."""

    kind = "crop"
    label = "Crop"
    inputs = ["image", "video", "boxes", "labels"]  # boxes optional, per-image only; labels optional
    required = []  # image or video, exactly one, checked in run
    outputs = ["image", "video", "labels"]

    box: str = ""  # manual "x1,y1,x2,y2"; used only when `boxes` is unconnected

    def run(self, image=None, video=None, boxes=None, labels=None):
        batch, port = _image_or_video(image, video)
        if boxes is not None and port == "video":
            raise ValueError("cropping clips by boxes isn't supported — use the drawn/typed box, or Sample Frames first")
        _check_per_image(boxes, "boxes")
        manual = self.box.replace(" ", "").split(",") if self.box.strip() else None
        if manual is not None and len(manual) != 4:
            raise ValueError('box must be "x1,y1,x2,y2"')
        if boxes is None and manual is None:
            raise ValueError("connect a boxes input or set the box field")
        if boxes is not None and len(boxes) != len(batch):
            raise ValueError(f"{len(boxes)} box sets for {len(batch)} images — wire the same batch into the segmenter")
        # strict, unlike SampleFrames' "missing tail = unlabeled": mid-graph a mismatch means the wrong batch got wired
        if labels is not None and len(labels) != len(batch):
            raise ValueError(f"{len(labels)} labels for {len(batch)} items — wire the same batch into both")
        per_instance = labels is not None and _per_instance(labels)
        if per_instance:
            if boxes is None:  # drawn/typed box or video: nothing to align concept names to
                _check_class_labels(labels)
            if any(len(ls) != len(b) for ls, b in zip(labels, boxes)):
                raise ValueError("labels don't line up with the boxes — wire the same SAM3 into both")
        out, out_labels = [], []
        for i, im in enumerate(batch):
            h, w = im.shape[-2:]
            for j, coords in enumerate(boxes[i].tolist() if boxes is not None else [manual]):
                x1, x2 = sorted((int(float(coords[0])), int(float(coords[2]))))
                y1, y2 = sorted((int(float(coords[1])), int(float(coords[3]))))
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w - 1), min(y2, h - 1)
                if x2 < x1 or y2 < y1:
                    raise ValueError(f"empty crop box {coords} for {w}x{h} image")
                out.append(im[..., y1 : y2 + 1, x1 : x2 + 1])
                if labels is not None:  # zero-box images drop out of both outputs together
                    out_labels.append(labels[i][j] if per_instance else labels[i])
        if not out:  # every image's box set was empty -> an empty batch downstream is a worse error
            raise ValueError("no boxes to crop — the segmenter found nothing (try a lower threshold or another prompt)")
        # only claim the labels output when it was wired in (SampleFrames' rule)
        return {port: out} | ({"labels": out_labels} if labels is not None else {})


def _order_corners(pts):
    """Order 4 corner points as [top-left, top-right, bottom-right, bottom-left]
    so a perspective warp maps an overlay's corners to the matching mask corners."""
    import numpy as np

    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]  # y - x: smallest at top-right, largest at bottom-left
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def _min_area_quad(b):
    """Corners (TL,TR,BR,BL) of the minimum-area rotated rectangle around the True
    pixels of boolean mask `b` (cv2.minAreaRect; degenerate masks come out as a
    zero-width/height quad, which the perspective warp handles)."""
    import cv2
    import numpy as np

    ys, xs = np.nonzero(b)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    return _order_corners(cv2.boxPoints(cv2.minAreaRect(pts)))


# WarpOverlay's `color` config shadows Node's palette-color ClassVar on purpose:
# spec() reads the ClassVar off the class, run() reads the field off the
# instance — disjoint, so only pydantic's warning needs silencing.
warnings.filterwarnings("ignore", message='Field name "color" in "WarpOverlay" shadows')


class WarpOverlay(TransformNode):
    """Warp an overlay image onto each masked region of the base image.

    Fits the overlay into every mask's minimum-area (rotated) bounding box via a
    perspective transform, so e.g. a logo lands on a tilted license plate at the
    right angle, then composites it over the base clipped to the mask. Wire the
    scene into `image`, the logo into `overlay`, and a segmenter's `masks` in.
    The overlay is stretched to fill each region (aspect ratio not preserved).
    A single-image overlay batch is broadcast across the scene batch. With
    nothing wired into `overlay`, fills each masked region with `color` instead;
    combined with Refine Masks' invert this puts detected subjects on a plain
    canvas."""

    kind = "warp_overlay"
    label = "Warp Overlay"
    inputs = ["image", "overlay", "masks"]
    required = ["image", "masks"]
    outputs = ["image"]

    color: str = Field(
        "#ffffff",
        pattern=r"^#[0-9a-fA-F]{6}$",
        description='fill for the masked regions when no overlay is wired, "#rrggbb"',
    )

    def run(self, image, masks, overlay=None):
        import torch
        from torchvision.transforms.v2 import functional as F

        _check_per_image(masks, "masks")
        fill = torch.tensor(_hex_rgb(self.color), dtype=torch.uint8).unsqueeze(1)  # (3,1) broadcasts over (3,K)
        out = []
        for i, im in enumerate(image):
            _, h, w = im.shape
            if overlay is not None:
                start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                ov = overlay[i] if len(overlay) > 1 else overlay[0]  # broadcast a single overlay
                ov_full = F.resize(ov, [h, w])  # fill the scene frame; corners map to the mask quad
            res = im.clone()
            for m in masks[i]:
                b = m.bool()
                if not b.any():
                    continue
                if overlay is None:  # solid fill needs no perspective warp
                    res[:, b] = fill
                    continue
                end = [[float(x), float(y)] for x, y in _min_area_quad(b.cpu().numpy())]
                warped = F.perspective(ov_full, start, end, fill=0)  # (C,H,W)
                res[:, b] = warped[:, b]  # rotated rect encloses the mask, so it covers every b pixel
            out.append(res)
        return out


class VisualPrompt(Node):
    """Interactive prompt source: page through the upstream batch in the frontend
    and draw positive/negative points and boxes on each image independently. The
    image input is both the drawing reference and how we learn the batch size —
    the node outputs one prompt set per image (`per_image`), which SAM3 applies
    image-for-image (frames you didn't draw on get no mask).

    Wire a `video` in instead to prompt clips: the preview pages through every
    frame of every clip (flattened clip-major), one prompt set per frame — draw
    on whichever frames pin the object(s) down, and SAM3's video path seeds its
    tracker at each drawn frame and follows the object(s) through the clip."""

    kind = "visual_prompt"
    label = "Visual Prompt"
    category = "Prompt"
    color = "#0d9488"
    inputs = ["image", "video"]
    required = []  # image or video, exactly one, checked in run
    outputs = ["prompts"]

    # Filled by the frontend drawing UI: one dict per drawing surface — per
    # image, or per frame of each clip (flattened clip-major, matching the
    # preview) — each with {points, point_labels, boxes, box_labels} in
    # full-resolution image pixels. May be shorter than the batch (trailing
    # items left undrawn -> empty).
    frames: list = []

    def run(self, image=None, video=None):
        batch, port = _image_or_video(image, video)
        if port == "video":  # one prompt set per frame, across all clips
            batch = [f for clip in batch for f in clip]

        def frame(i):
            f = self.frames[i] if i < len(self.frames) and isinstance(self.frames[i], dict) else {}
            return {
                "points": f.get("points") or [],
                "point_labels": f.get("point_labels") or [],
                "boxes": f.get("boxes") or [],
                "box_labels": f.get("box_labels") or [],
            }

        frames = [frame(i) for i in range(len(batch))]
        if not any(f["points"] or f["boxes"] for f in frames):
            raise ValueError("Visual Prompt is empty. Draw a point or box in the visual prompt node.")
        return {"prompts": {"per_image": frames}}


class TextPrompt(Node):
    """Text concept prompt for SAM3's open-vocabulary detector, e.g. "person, truck".
    A pure source: no image needed, just wire its `prompts` port into SAM3."""

    kind = "text_prompt"
    label = "Text Prompt"
    category = "Prompt"
    color = "#0d9488"
    inputs = []
    outputs = ["prompts"]

    text: str = ""

    def run(self):
        if not self.text.strip():
            raise ValueError('Text Prompt is empty — type what to segment, e.g. "person"')
        return {"prompts": {"text": self.text}}


# SAM3 models are heavy — load each once, lazily, and reuse across runs.
_MODELS: dict = {}


def free_models():
    """Drop the cached models and hand their VRAM back — /chat calls this so
    the LLM fits on the 12 GB card (chat.unload_model() is the mirror move in
    /run). A model a concurrently running node still references stays alive
    until that run's local reference dies; nothing breaks, it just frees late."""
    if not _MODELS:
        return
    import gc

    import torch

    _MODELS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _chunked_iter(items, fn, chunk=8):
    """Run `fn(sublist) -> list` over `items` in GPU-sized bites, yielding each
    chunk's results as they're produced (peak RAM = one chunk, for callers
    that finalize per-item work immediately instead of holding the batch).

    Activation memory scales with the frames/images per forward pass, so a
    whole clip in one batch OOMs real GPUs. Chunks cap the peak, and a CUDA
    OOM halves the chunk and retries — self-tuning to whatever card is
    present instead of hardcoding a per-GPU batch size.

    Also where model nodes get their progress bar: one forward pass = one tick,
    so a caller that declared _total() reports without another line of code."""
    import torch

    i = 0
    while i < len(items):
        n = min(chunk, len(items) - i)
        try:
            res = fn(items[i : i + n])
            i += n
            _step(n)
            yield from res
        except torch.cuda.OutOfMemoryError:
            if n == 1:
                raise  # can't go smaller — the model itself doesn't fit
            chunk = max(1, n // 2)
            torch.cuda.empty_cache()


def _chunked(items, fn, chunk=8):
    return list(_chunked_iter(items, fn, chunk))


def _mask_budget(budget, spent, masks, n_total, i, remedy, unit="RAM"):
    """Running guard for masks as they come off `_chunked`/`_chunked_iter`:
    `spent` bytes so far (masks are ~1 byte/px, bool or uint8) plus this
    pass's `masks`, projected from the `i` items done so far to the full
    `n_total`-item batch or clip — 2x for the uint8 stack copy
    `_instances_to_boxes_masks` makes. Raises naming `remedy` (the knobs to
    fix it) if that outgrows `budget`; `budget` falsy (unknown) means no
    guard. Image callers project against RAM (`_mem_available`); video
    callers pass `unit="disk"` with `budget=_spill_free()` — masks spill.
    Returns the updated `spent`."""
    import numpy as np

    spent += sum(int(np.asarray(m).size) for m in masks)
    if budget and 2 * spent * n_total > budget * i:
        raise ValueError(
            f"needs ~{2 * spent * n_total / i / 1e9:.1f} GB of {unit} for masks "
            f"(~{budget / 1e9:.1f} GB available) — {remedy}"
        )
    return spent


def _slice_grid(h, w, side, overlap):
    """[(x1,y1,x2,y2), ...] tiles covering (h,w) with `overlap` fraction shared
    between neighbors, last tile clamped flush to the edge, PLUS the full
    frame as the final entry (SAHI's full-image pass — catches objects bigger
    than a tile). Coordinates are exclusive on the right (crop bounds, not
    port-contract boxes). Returns [] when the image already fits one tile
    (max(h,w) <= side) — caller runs the untiled path, identical to today."""
    if side <= 0 or max(h, w) <= side:
        return []
    stride = max(1, round(side * (1 - overlap)))  # guard degenerate overlap >= 1

    def starts(size):
        s = list(range(0, max(size - side, 0) + 1, stride))
        if s[-1] != max(size - side, 0):  # flush the last tile to the edge, no gap
            s.append(max(size - side, 0))
        return s

    seen, tiles = set(), []
    for y in starts(h):
        for x in starts(w):
            t = (x, y, min(x + side, w), min(y + side, h))
            if t not in seen:  # starts() can repeat a coordinate when the image barely exceeds `side`
                seen.add(t)
                tiles.append(t)
    tiles.append((0, 0, w, h))
    return tiles


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _phrases(text):
    """SAM3 is prompted with one noun phrase at a time, so a comma-separated Text
    Prompt ("person, truck") is several concepts — one detection pass each (same
    convention as SigLIP 2's labels), which is also what makes per-concept counts
    possible. Empty text -> a single unnamed pass (box exemplars only)."""
    return [p.strip() for p in text.split(",") if p.strip()] or [""]


def _concept_model():
    """facebook/sam3 open-vocabulary detector+segmenter (text + box exemplars)."""
    if "concept" not in _MODELS:
        from transformers import Sam3Model, Sam3Processor

        proc = Sam3Processor.from_pretrained("facebook/sam3")
        model = Sam3Model.from_pretrained("facebook/sam3").to(_device()).eval()
        _MODELS["concept"] = (model, proc)
    return _MODELS["concept"]


def _tracker_model():
    """facebook/sam3 video tracker: single-frame for image point/box prompting,
    propagated over the whole clip for drawn prompts on video."""
    if "tracker" not in _MODELS:
        import torch
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        proc = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
        # bf16 throughout: halves the weights AND the video session, whose
        # per-frame storage (see _track_drawn) is the app's biggest allocation
        model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(_device()).eval()
        _MODELS["tracker"] = (model, proc)
    return _MODELS["tracker"]


def _siglip_model():
    """google/siglip2 zero-shot image classifier (sigmoid over image-text logits)."""
    if "siglip" not in _MODELS:
        from transformers import AutoModel, AutoProcessor

        name = "google/siglip2-base-patch16-224"
        proc = AutoProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(_device()).eval()
        _MODELS["siglip"] = (model, proc)
    return _MODELS["siglip"]


# --- generic Hugging Face model loading ---------------------------------------
# Model nodes load any checkpoint straight from its Hub id, via transformers'
# pipeline() for detect/classify/segment/depth/caption (its output
# normalization across model families is the whole point) or a plain
# AutoModel encoder for Embed. One _MODELS cache keyed by (task, model_id)
# so different models/tasks coexist and free_models() evicts them the same
# way it does SAM3/SigLIP.


def _hf_load(build, model_id):
    """Load an HF model/pipeline: one retry after free_models() on a CUDA OOM,
    and any other load error (bad id, gated repo, task mismatch) turned into a
    one-line message instead of a raw traceback."""
    import torch

    try:
        return build()
    except torch.cuda.OutOfMemoryError:
        free_models()
        try:
            return build()
        except torch.cuda.OutOfMemoryError:
            raise ValueError(
                f"HF model {model_id!r} doesn't fit in VRAM even after freeing other cached models — try a smaller model"
            ) from None
    except Exception as e:
        msg = str(e)
        hint = " — set HF_TOKEN in the environment for gated/private models" if any(
            t in msg for t in ("401", "403", "gated", "Gated")
        ) else ""
        raise ValueError(f"couldn't load HF model {model_id!r}: {(msg.splitlines() or [repr(e)])[0]}{hint}") from e


def _hf_pipeline(task, model_id, trust_remote_code):
    key = ("hf", task, model_id)
    if key not in _MODELS:
        from transformers import pipeline

        _MODELS[key] = _hf_load(
            lambda: pipeline(task, model=model_id, device=_device(), trust_remote_code=trust_remote_code), model_id
        )
    return _MODELS[key]


def _hf_zero_shot(model_id, mapping, trust_remote_code):
    """True if model_id's architecture is registered for this zero-shot task
    family (OWL-ViT/OWLv2 for detection, CLIP/SigLIP for classification) — the
    same lookup transformers' pipeline() uses to route a model to a task, so a
    new zero-shot arch doesn't need a hardcoded name check."""
    from transformers import AutoConfig

    try:
        model_type = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code).model_type
    except Exception:
        return False
    return model_type in mapping


def _hf_encoder(model_id, trust_remote_code):
    """Any Hugging Face vision encoder, loaded as AutoModel + AutoImageProcessor."""
    key = ("hf", "embed", model_id)
    if key not in _MODELS:
        from transformers import AutoImageProcessor, AutoModel

        def build():
            proc = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code).to(_device()).eval()
            return model, proc

        _MODELS[key] = _hf_load(build, model_id)
    return _MODELS[key]


class HfNode(Node):
    """Base for the model nodes — holds the model id and the remote-code trust
    boundary every one of them takes."""

    model_id: str
    trust_remote_code: bool = Field(
        False, description="Run custom modeling code shipped in the model repo. Only enable for models you trust."
    )


def _seg_concepts(model, proc, hwc, phrases, threshold, with_scores=False, **kw):
    """Segment every concept in `phrases` over one chunk of RGB HxWx3 images.

    The scoring head mean-pools the prompt into a single vector, so SAM3 takes one
    noun phrase per pass — but the vision backbone is ~90% of that pass, and
    `vision_embeds=` accepts its output in place of `pixel_values`. So the chunk is
    encoded once and each concept only pays for the head, bit-identical to a full
    forward per phrase. `kw` carries the box exemplars, if any.

    Returns, per image, a tuple of its instance masks for each phrase (in order).
    `with_scores=True` (SAHI tile merging needs to rank instances) instead
    tuples each phrase's masks with its per-instance score tensor (or None)."""
    import torch

    embeds, by_phrase = None, []
    for phrase in phrases:
        inputs = proc(
            images=hwc,
            text=[phrase] * len(hwc) if phrase else None,
            return_tensors="pt",
            **kw,
        ).to(_device())
        pixel_values = inputs.pop("pixel_values")  # replaced by vision_embeds= below
        with torch.inference_mode():
            if embeds is None:
                embeds = model.vision_encoder(pixel_values)
            outputs = model(vision_embeds=embeds, **inputs)
        seg = proc.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5,
            target_sizes=[tuple(im.shape[:2]) for im in hwc],
        )
        # bool, not the post-processor's int64: these binary masks accumulate
        # per instance per frame — at 8 bytes/px a flock of birds on a clip
        # OOM-killed the process
        masks = [s["masks"].to("cpu", torch.bool) if torch.is_tensor(s["masks"]) else s["masks"] for s in seg]
        if with_scores:
            scores = [s.get("scores").cpu() if torch.is_tensor(s.get("scores")) else s.get("scores") for s in seg]
            by_phrase.append(list(zip(masks, scores)))
        else:
            by_phrase.append(masks)
    return list(zip(*by_phrase))


class Segment(HfNode):
    """Segment objects in images or video; the settings dropdown picks the model.

    SAM3 (facebook/sam3, the default) is promptable and NEEDS a prompt wired
    into `prompts`. The image arrives as a list of (C,H,W) tensors (sizes may
    differ); the HF processor takes that list natively and batches/pads
    internally, so there is no manual wrapping here. Prompting routes to the
    right SAM3 model:
      * drawn points present -> Sam3TrackerVideoModel (single object, points + box),
                                 run per batch image with that image's own prompt
      * drawn boxes only     -> Sam3Model concept model (positive/negative box
                                 exemplars), per image, returning every matching
                                 instance
      * text only            -> Sam3Model concept model, one text prompt broadcast
                                 across the batch
    Drawn prompts (Visual Prompt) arrive per image via `prompts["per_image"]`;
    text (Text Prompt) is broadcast to all, one concept per comma-separated
    phrase — SAM3 takes a single noun phrase per pass, so "person, truck" runs
    two passes whose instances are concatenated (both share one vision pass, so
    the second concept is nearly free — see _seg_concepts). Mixes that would
    silently drop a prompt raise instead: points + text, a box-only image while
    other images have points, and text + boxes when some images have no box.
    `threshold` applies to SAM3's concept paths only. Outputs per-image `boxes`
    and `masks` plus `labels` — the concept name behind each instance, aligned
    1:1 with them (see Count Objects); "" for drawn prompts.

    Wire `video` + a Text Prompt instead for per-frame concept segmentation of
    clips: the same `masks`/`boxes` ports then carry per-frame values (per
    clip, per frame, aligned 1:1). Those carry no identity across frames —
    put a Track Objects node after them for that. Wire `video` + a Visual
    Prompt to track specific objects instead: prompts drawn on any frame(s) of
    a clip seed the tracker there and propagate through the clip with memory
    attention (points = one object, re-anchored at each drawn frame; boxes
    alone = one object per box, seeded at its frame) — see _track_drawn.
    Frames where an object is hidden contribute no instance; instance order
    per frame stays stable, so Track Objects still ids them.

    Any other model id (the Mask2Former instance / SegFormer semantic presets,
    or a custom Hugging Face image-segmentation checkpoint) is NOT promptable —
    wiring a prompt raises. It segments every image (or every frame of each
    clip, same per-frame port shapes) as-is via the HF pipeline: semantic
    checkpoints give one mask per class present in the image; `labels` carries
    the class name per instance, and `threshold` is ignored.

    `slice_size` (SAHI tiled inference) works with SAM3's Text Prompt path
    only — drawn prompts and non-SAM3 models raise. Known limitation: an
    object split across a tile border with < 0.5 IoU between its fragments
    survives as two instances (NMM-style mask merging is the upgrade path).
    """

    kind = "segment"
    label = "Segment"
    category = "Segment"
    color = "#db2777"
    inputs = ["image", "video", "prompts"]
    required = []  # prompts only for SAM3; image or video either-or — both checked in run
    outputs = ["boxes", "masks", "labels"]

    model_id: str = Field(
        "facebook/sam3",
        json_schema_extra={"options": [
            {"value": "facebook/sam3", "label": "SAM3 (promptable)"},
            {"value": "facebook/mask2former-swin-tiny-coco-instance", "label": "Mask2Former (instance)"},
            {"value": "nvidia/segformer-b0-finetuned-ade-512-512", "label": "SegFormer (semantic)"},
        ], "hf_search": ["image-segmentation"]},
    )
    threshold: float = Field(0.5, description="Confidence cutoff for SAM3's text-concept model — detections scoring below it are dropped. Ignored for drawn prompts (the tracker path has no score) and for non-SAM3 models.")
    slice_size: int = Field(0, description="Tile side in px for sliced inference (SAHI) — the image is cut into overlapping tiles, each inferred separately, detections merged. 0 = off. Use when small objects on large images get missed.")
    slice_overlap: float = Field(0.2, description="Fraction of tile side that adjacent tiles overlap; only used when slice_size > 0.")

    def run(self, image=None, video=None, prompts=None):
        import torch

        if self.slice_size > 0 and self.model_id != "facebook/sam3":
            raise ValueError("slicing needs SAM3 with a Text Prompt, or use Detect")

        if self.model_id != "facebook/sam3":
            if prompts is not None:
                raise ValueError(
                    f"{self.model_id!r} isn't promptable — remove the prompt, or pick SAM3 in settings"
                )
            return self._run_pipeline(image, video)

        if prompts is None:
            raise ValueError("SAM3 needs a prompt — wire a Text/Visual Prompt, or pick another model in settings")
        batch, port = _image_or_video(image, video)
        text = (prompts.get("text") or "").strip()
        phrases = _phrases(text)  # one concept detection pass per phrase
        # normalize drawn prompts once: all four keys present
        # (graphs are user-POSTed JSON, so don't assume Visual Prompt's shape)
        empty = {"points": [], "point_labels": [], "boxes": [], "box_labels": []}
        per = prompts.get("per_image") or []

        if port == "video":
            # Visual Prompt on video is per-frame: one prompt set per frame of
            # each clip, flattened clip-major (matching its preview paging) —
            # regroup into per-clip lists so each clip carries its own T sets
            total = sum(len(clip) for clip in batch)
            flat = [(empty | per[i]) if i < len(per) and isinstance(per[i], dict) else empty for i in range(total)]
            frames, i = [], 0
            for clip in batch:
                frames.append(flat[i : i + len(clip)])
                i += len(clip)
            drawn = any(f["points"] or f["boxes"] for fs in frames for f in fs)
            if self.slice_size > 0 and drawn:
                raise ValueError(
                    "sliced inference needs a Text Prompt only — remove the drawn prompts or set slice_size to 0"
                )
            if drawn:
                if text:
                    raise ValueError(
                        "SAM3 on video: drawn prompts can't be combined with a text prompt — remove one of the two"
                    )
                return self._track_drawn(batch, frames)
            if not text:
                raise ValueError("SAM3 on video needs a prompt: draw on the clip in a Visual Prompt or wire a Text Prompt")
            # before the model load: an empty bar beats no bar while weights come in
            if self.slice_size > 0:
                _total(sum(  # tiles (or 1, untiled) per frame, every frame of every clip
                    (len(_slice_grid(clip.shape[-2], clip.shape[-1], self.slice_size, self.slice_overlap)) or 1) * clip.shape[0]
                    for clip in batch
                ))
            else:
                _total(sum(len(clip) for clip in batch))  # a chunk of frames per forward, all concepts
            model, proc = _concept_model()

            def seg_frames(sub):  # a few frames per forward; masks leave the GPU per chunk
                return _seg_concepts(model, proc, sub, phrases, self.threshold)

            out_masks, out_boxes, out_labels = [], [], []
            remedy = "raise threshold, or lower fps, max_frames, or max_side on Load Video, or free disk space"
            for clip in batch:
                hwc = _batch_to_hwc(clip)  # the clip's T frames
                # masks cost instances × frames × H×W — a dense scene (a flock
                # of birds) adds up fast, so project the clip's total from the
                # frames done so far and refuse what can't fit instead of
                # getting OOM-killed near the end. Finalized per frame as it
                # arrives and spilled — peak RAM is one chunk, not the clip.
                budget, spent = _spill_free(), 0
                frame_results = (
                    (self._tiled_instances(model, proc, f, phrases) for f in hwc) if self.slice_size > 0
                    else _chunked_iter(hwc, seg_frames)
                )
                writer, boxes_c, labels_c = _SpillWriter(), [], []
                for t, by_phrase in enumerate(frame_results):
                    masks_t, names_t = [], []
                    for phrase, m in zip(phrases, by_phrase):  # each concept's instances append to the frame
                        masks_t += list(m)
                        names_t += [phrase] * len(m)
                    spent = _mask_budget(budget, spent, masks_t, len(hwc), t + 1, remedy, unit="disk")
                    h, w = hwc[t].shape[:2]
                    bx, ms, kept = _instances_to_boxes_masks(masks_t, h, w, names_t)
                    boxes_c.append(bx)
                    labels_c.append(kept)
                    writer.add(ms)
                out_boxes.append(boxes_c)
                out_masks.append(writer.finish())
                out_labels.append(labels_c)
            return {"masks": out_masks, "boxes": out_boxes, "labels": out_labels}

        imgs = _batch_to_hwc(batch)  # list of B RGB HxWx3 uint8 arrays
        sizes = [im.shape[:2] for im in imgs]  # [(h, w), ...]
        B = len(imgs)
        frames = [(empty | per[i]) if i < len(per) and isinstance(per[i], dict) else empty for i in range(B)]
        has_points = any(f["points"] for f in frames)
        has_boxes = any(f["boxes"] for f in frames)
        if self.slice_size > 0 and (has_points or has_boxes):
            raise ValueError(
                "sliced inference needs a Text Prompt only — remove the drawn prompts or set slice_size to 0"
            )
        # instances and their concept names, per image; unprompted images stay empty
        per_image = [[] for _ in range(B)]
        names = [[] for _ in range(B)]

        if has_points:  # interactive path: tracker seeds one object per image
            if text:
                raise ValueError("SAM3: drawn points can't be combined with a text prompt — remove one of the two")
            for i, f in enumerate(frames):
                if f["boxes"] and not f["points"]:
                    raise ValueError(
                        f"SAM3: image {i + 1} has only a box drawn while other images have points — "
                        "add a point to it or remove its box"
                    )
            _total(sum(1 for f in frames if f["points"]))  # one tracker pass per prompted image
            model, proc = _tracker_model()
            for i, (im, (h, w), f) in enumerate(zip(imgs, sizes, frames)):
                pts = f["points"]
                if not pts:  # nothing drawn on this image -> no masks
                    continue
                labels = f["point_labels"] or [1] * len(pts)
                session = proc.init_video_session(
                    video=[im],
                    inference_device=_device(),
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=torch.bfloat16,  # matches the model's dtype
                )
                kw = {
                    "input_points": [[[[float(x), float(y)] for x, y in pts]]],
                    "input_labels": [[[int(v) for v in labels]]],
                }
                if f["boxes"]:  # tracker seeds one object -> use this image's first box only
                    b = f["boxes"][0]
                    kw["input_boxes"] = [[[float(b[0]), float(b[1]), float(b[2]), float(b[3])]]]
                proc.add_inputs_to_inference_session(
                    inference_session=session, frame_idx=0, obj_ids=1, clear_old_inputs=True, **kw
                )
                with torch.inference_mode():
                    out = model(inference_session=session, frame_idx=0)
                # drawn prompts name no concept -> `labels` stays "" for these
                per_image[i] = proc.post_process_masks([out.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                _step()
        elif has_boxes or text:  # concept model: box exemplars per image and/or one text broadcast
            if has_boxes and text:
                for i, f in enumerate(frames):
                    if not f["boxes"]:
                        # ponytail: raise rather than text-only fallback on boxless
                        # images; add a second text-only concept pass if this bites
                        raise ValueError(
                            "SAM3: with box exemplars drawn, text applies only to boxed images; "
                            f"image {i + 1} has no box — box it too or remove the text prompt"
                        )
            if has_boxes:  # run only the images that have boxes
                idxs = [i for i, f in enumerate(frames) if f["boxes"]]
                kw = {
                    "input_boxes": [[[float(c) for c in b] for b in frames[i]["boxes"]] for i in idxs],
                    "input_boxes_labels": [
                        [int(v) for v in (frames[i]["box_labels"] or [1] * len(frames[i]["boxes"]))] for i in idxs
                    ],
                }
            else:  # text only: the whole batch
                idxs, kw = list(range(B)), {}
            pos = {i: j for j, i in enumerate(idxs)}  # image index -> row in kw's lists
            # before the model load: an empty bar beats no bar while weights come in
            if self.slice_size > 0:  # has_boxes is false here (guarded above) — idxs is the whole batch
                _total(sum(len(_slice_grid(*sizes[i], self.slice_size, self.slice_overlap)) or 1 for i in idxs))
            else:
                _total(len(idxs))  # a chunk of prompted images per forward, all concepts
            model, proc = _concept_model()

            def seg_images(sub):  # sub: image indices, a few per forward
                return _seg_concepts(
                    model, proc, [imgs[i] for i in sub], phrases, self.threshold,
                    **{k: [v[pos[i]] for i in sub] for k, v in kw.items()},
                )

            # same overrun risk as the video path above, keyed off images done so far
            budget, spent = _mem_available(), 0
            image_results = (
                ((i, self._tiled_instances(model, proc, imgs[i], phrases)) for i in idxs) if self.slice_size > 0
                else zip(idxs, _chunked(idxs, seg_images))
            )
            for k, (i, by_phrase) in enumerate(image_results):
                for phrase, m in zip(phrases, by_phrase):  # each concept's instances append to the image
                    per_image[i] += list(m)
                    names[i] += [phrase] * len(m)
                spent = _mask_budget(
                    budget, spent, [m for ms in by_phrase for m in ms], len(idxs), k + 1,
                    "raise threshold, or use smaller images",
                )
        else:
            raise ValueError("SAM3 needs a prompt: draw points/boxes in a Visual Prompt or wire a Text Prompt")

        boxes_out, masks_out, labels_out = [], [], []
        for m, (h, w), nm in zip(per_image, sizes, names):
            bx, ms, kept = _instances_to_boxes_masks(m, h, w, nm)
            boxes_out.append(bx)
            masks_out.append(ms)
            labels_out.append(kept)
        return {"boxes": boxes_out, "masks": masks_out, "labels": labels_out}

    def _tiled_instances(self, model, proc, frame, phrases):
        """SAHI for one HWC frame/image: tile it, run `_seg_concepts` per tile
        (chunked, so OOM-halving and progress still work), paste each tile-local
        mask into a full-frame canvas at its tile offset, then per concept keep
        the highest-scoring instances via greedy NMS at box-IoU 0.5 (falls back
        to mask area when a score is missing — bigger fragment wins). Returns
        the same per-phrase tuple of instance-mask stacks `_seg_concepts` does,
        so callers don't care whether slicing happened."""
        import torch
        from torchvision.ops import nms

        h, w = frame.shape[:2]
        grid = _slice_grid(h, w, self.slice_size, self.slice_overlap)
        if not grid:  # fits in one tile -> untiled path, identical to today
            return _chunked([frame], lambda sub: _seg_concepts(model, proc, sub, phrases, self.threshold))[0]
        tiles = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in grid]
        per_tile = _chunked(tiles, lambda sub: _seg_concepts(model, proc, sub, phrases, self.threshold, with_scores=True))
        out = []
        for p in range(len(phrases)):
            boxes, scores, masks = [], [], []
            for (x1, y1, x2, y2), tile_result in zip(grid, per_tile):
                tm, ts = tile_result[p]
                for k in range(tm.shape[0]):
                    canvas = torch.zeros((h, w), dtype=torch.bool)
                    canvas[y1:y2, x1:x2] = tm[k]
                    ys, xs = torch.nonzero(canvas, as_tuple=True)
                    if xs.numel() == 0:
                        continue
                    boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
                    scores.append(float(ts[k]) if ts is not None else float(canvas.sum()))
                    masks.append(canvas)
            if not masks:
                out.append(torch.zeros((0, h, w), dtype=torch.bool))
                continue
            keep = nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32), 0.5)
            out.append(torch.stack([masks[i] for i in keep.tolist()]))
        return tuple(out)

    def _track_drawn(self, batch, frames):
        """Drawn prompts on clips (`frames`: per clip, one prompt set per
        frame): seed the video tracker at every prompted frame and propagate
        through the clip (memory attention, not box geometry). Any points in a
        clip -> ONE object, re-anchored at each prompted frame with that
        frame's points plus its first box (the image path's rule, per frame) —
        covers drift correction and late appearance alike. Boxes alone -> each
        box on any frame is its own tracked object, seeded where it was drawn.
        Frames before the earliest prompt still get tracked within its window:
        memory attention conditions them on the later seed — but long clips
        are split into overlapping windows (memory only spans one window), so
        draw on an early frame; windows entirely before the first prompt come
        out empty, and an object can lose its identity at a window seam while
        it's hidden there. Clips with nothing drawn come out empty, like
        undrawn images. Instances name no concept (labels ""), and a frame
        where an object is hidden just has fewer instances."""
        import torch

        for fs in frames:  # validate everything before any weights load
            for f in fs:
                if f["boxes"] and not f["points"] and not all(f["box_labels"] or [1] * len(f["boxes"])):
                    raise ValueError("SAM3 on video: a negative box can't seed tracking — use negative points instead")
        # before the model load: an empty bar beats no bar while weights come in
        _total(sum(len(clip) for clip in batch))
        model, proc = _tracker_model()
        # the session stores every frame resized to the model's input size for
        # memory attention — ~6 MB/frame in bf16, by far the app's biggest
        # allocation (float32 was ~12 MB/frame: minute-long clips OOM-killed
        # the whole process). Clips too long for one session get split into
        # overlapping windows (_track_windows) instead of refused.
        size = proc.video_processor.size
        frame_bytes = 3 * size.height * size.width * 2  # bf16
        avail = _mem_available()

        clip_info = []  # (T, h, w, prompted frame indices, windows or None)
        for clip, fs in zip(batch, frames):
            T, h, w = len(clip), clip.shape[-2], clip.shape[-1]
            prompted = [t for t, f in enumerate(fs) if f["points"] or f["boxes"]]
            wins = None
            if prompted:
                # ponytail: half the old refusal's 2x headroom — leaves room
                # for this window's mask/box outputs alongside its session
                W = T if avail is None else min(T, max(32, avail // (4 * frame_bytes)))
                wins = _track_windows(T, W)
            clip_info.append((T, h, w, prompted, wins))
        # windows need the model loaded (for frame_bytes), so progress can only
        # reflect real work (seam frames counted twice) after this point —
        # second call just overwrites the provisional [0, n] from above
        _total(sum(T if wins is None else sum(e - s for s, e in wins) for T, h, w, prompted, wins in clip_info))

        out_masks, out_boxes, out_labels = [], [], []
        for clip, fs, (T, h, w, prompted, wins) in zip(batch, frames, clip_info):
            boxes_c = [torch.zeros((0, 4), dtype=torch.int32) for _ in range(T)]
            labels_c: list = [[] for _ in range(T)]
            if not prompted:
                masks_c = [torch.zeros((0, h, w), dtype=torch.uint8) for _ in range(T)]
                _step(T)  # nothing drawn on this clip -> it stays empty (cheap, stays in RAM)
                out_boxes.append(boxes_c)
                out_masks.append(masks_c)
                out_labels.append(labels_c)
                continue

            single = any(fs[t]["points"] for t in prompted)  # any points -> one object
            writer = _SpillWriter()
            k = 0  # multi-box object id counter — runs across the whole clip, not per window

            def add_prompt(session, local_idx, f, k):
                if single:  # obj 1: this frame's points + its first box re-anchor it
                    kw = {}
                    if f["points"]:
                        kw["input_points"] = [[[[float(x), float(y)] for x, y in f["points"]]]]
                        kw["input_labels"] = [[[int(v) for v in (f["point_labels"] or [1] * len(f["points"]))]]]
                    if f["boxes"]:
                        b = f["boxes"][0]
                        kw["input_boxes"] = [[[float(b[0]), float(b[1]), float(b[2]), float(b[3])]]]
                    proc.add_inputs_to_inference_session(
                        inference_session=session, frame_idx=local_idx, obj_ids=1, clear_old_inputs=True,
                        original_size=(h, w), **kw
                    )
                    return k
                for b in f["boxes"]:  # boxes only: each box is its own tracked object
                    k += 1
                    proc.add_inputs_to_inference_session(
                        inference_session=session, frame_idx=local_idx, obj_ids=k, clear_old_inputs=True,
                        original_size=(h, w), input_boxes=[[[float(c) for c in b]]],
                    )
                return k

            carry = {}  # obj_id -> seam mask, seeded at the next window's local frame 0
            for win_idx, (s, e) in enumerate(wins):
                Tw = e - s
                prompted_in_win = [t for t in prompted if s <= t < e]
                if not carry and not prompted_in_win:
                    # nothing to track here — before the first prompt, or every
                    # object was hidden at the seam and no later prompt re-seeds
                    # it: no session, cheap empty frames
                    for t in range(s, e):
                        if win_idx > 0 and t == s:
                            continue  # seam frame's output already came from the earlier window
                        writer.add(torch.zeros((0, h, w), dtype=torch.uint8))
                    _step(Tw)
                    continue

                session = proc.init_video_session(
                    inference_device=_device(),
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=torch.bfloat16,
                )
                # frames go in chunk by chunk: passing video= above would build
                # the whole window as float32 before the bf16 cast — a transient
                # 3x the stored size
                sub = clip[s:e]
                for i in range(0, Tw, 32):
                    px = proc.video_processor(videos=_batch_to_hwc(sub[i : i + 32]), device="cpu", return_tensors="pt")
                    for j, fr in enumerate(px.pixel_values_videos[0]):
                        session.add_new_frame(fr, i + j)

                frame0_prompted = bool(prompted_in_win) and prompted_in_win[0] == s
                if carry:  # seed objects carried over from the previous window's seam frame
                    for obj_id, mask in carry.items():
                        proc.add_inputs_to_inference_session(
                            inference_session=session, frame_idx=0, obj_ids=obj_id, input_masks=mask
                        )
                if frame0_prompted:  # a user re-anchor on the seam frame wins over the carried seed
                    k = add_prompt(session, 0, fs[s], k)
                    prompted_in_win = prompted_in_win[1:]
                if carry or frame0_prompted:
                    # consume this frame's inputs before adding the next frame's:
                    # the model treats any frame it visits while an object still
                    # has pending inputs elsewhere as an unprompted conditioning
                    # frame, which would store a garbage anchor
                    with torch.inference_mode():
                        model(inference_session=session, frame_idx=0)
                for t in prompted_in_win:
                    k = add_prompt(session, t - s, fs[t], k)
                    with torch.inference_mode():
                        model(inference_session=session, frame_idx=t - s)

                # propagate_in_video_iterator visits every frame of this window
                # 0..Tw-1, in order, exactly once — masks spill straight into a
                # writer as they come off the model instead of a preallocated
                # RAM list
                carry = {}
                with torch.inference_mode():
                    for out in model.propagate_in_video_iterator(session, start_frame_idx=0):
                        lt = int(out.frame_idx)
                        # a hidden object's empty mask drops out here, so instance
                        # counts vary per frame — the order of the rest stays stable
                        m = proc.post_process_masks([out.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                        if win_idx + 1 < len(wins) and lt == Tw - 1:
                            # this window's last frame seeds the next window's local
                            # frame 0 — raw per-object masks (before the empty-drop
                            # below) so object identity survives the seam
                            carry = {oid: mk for oid, mk in zip(out.object_ids, m) if bool((mk > 0).any())}
                        _step()
                        if win_idx > 0 and lt == 0:
                            continue  # the seam frame — its output already came from the earlier window
                        bx, ms, _ = _instances_to_boxes_masks(m, h, w)
                        t = s + lt
                        boxes_c[t], labels_c[t] = bx, [""] * len(bx)
                        writer.add(ms)
                del session  # frees before the next window's session allocates

            masks_c = writer.finish()
            out_boxes.append(boxes_c)
            out_masks.append(masks_c)
            out_labels.append(labels_c)
        return {"masks": out_masks, "boxes": out_boxes, "labels": out_labels}

    def _run_pipeline(self, image, video):
        """Non-SAM3 checkpoints through the HF image-segmentation pipeline —
        semantic, instance and panoptic models all come back as
        {label, score?, mask} dicts, normalized into aligned boxes/masks/labels."""
        import numpy as np
        from PIL import Image

        batch, port = _image_or_video(image, video)

        def to_outs(dets, h, w):  # -> (boxes, masks, labels) for one image/frame
            masks = [np.array(d["mask"]) for d in dets]
            names = [d["label"] for d in dets]
            return _instances_to_boxes_masks(masks, h, w, names)

        if port == "video":
            _total(sum(len(clip) for clip in batch))  # before the model load: an empty bar beats no bar
            pipe = _hf_pipeline("image-segmentation", self.model_id, self.trust_remote_code)
            remedy = "lower fps, max_frames, or max_side on Load Video, or free disk space"
            out_boxes, out_masks, out_labels = [], [], []
            for clip in batch:
                hwc = _batch_to_hwc(clip)
                pil = [Image.fromarray(f) for f in hwc]
                budget, spent = _spill_free(), 0
                writer, boxes_c, labels_c = _SpillWriter(), [], []
                for t, (d, im) in enumerate(zip(_chunked_iter(pil, pipe), hwc)):
                    bx, ms, kept = to_outs(d, *im.shape[:2])
                    boxes_c.append(bx)
                    labels_c.append(kept)
                    writer.add(ms)
                    spent = _mask_budget(budget, spent, [x["mask"] for x in d], len(hwc), t + 1, remedy, unit="disk")
                out_boxes.append(boxes_c)
                out_masks.append(writer.finish())
                out_labels.append(labels_c)
            return {"boxes": out_boxes, "masks": out_masks, "labels": out_labels}

        _total(len(batch))
        pipe = _hf_pipeline("image-segmentation", self.model_id, self.trust_remote_code)
        hwc = _batch_to_hwc(batch)
        pil = [Image.fromarray(f) for f in hwc]
        budget, spent, per_image = _mem_available(), 0, []
        for i, (d, im) in enumerate(zip(_chunked(pil, pipe), hwc)):
            per_image.append(to_outs(d, *im.shape[:2]))
            spent = _mask_budget(budget, spent, [x["mask"] for x in d], len(batch), i + 1, "use smaller images")
        return {
            "boxes": [x[0] for x in per_image],
            "masks": [x[1] for x in per_image],
            "labels": [x[2] for x in per_image],
        }


class MaskOps(Node):
    """Post-process masks from a segmenter (e.g. SAM3). Applies, in order:
    invert and fill_holes (fill interior background regions fully enclosed by
    foreground). Toggle each independently. Passes masks through unchanged if
    both are off. Per-image and per-frame (video) masks both work.

    invert means "the background of everything detected": the N instance masks
    are unioned first, then complemented, so out comes a single mask (N=1) —
    per-mask inversion would make each object part of every other object's
    background. Inverted masks therefore no longer align 1:1 with boxes/labels."""

    kind = "mask_ops"
    label = "Refine Masks"
    category = "Segment"
    color = "#db2777"
    inputs = ["masks"]
    outputs = ["masks"]

    invert: bool = False
    fill_holes: bool = False

    def run(self, masks):
        import numpy as np
        import torch
        from scipy.ndimage import binary_fill_holes

        def fix(per):  # Tensor(N,H,W) -> Tensor(N,H,W); invert unions to (1,H,W)
            if self.invert:  # background of all objects, not per-mask complement
                per = ~(per.cpu() > 0).any(dim=0, keepdim=True)
            res = []
            for m in per:
                b = m.cpu().numpy() > 0
                if self.fill_holes:
                    b = binary_fill_holes(b)
                res.append(torch.from_numpy(b.astype(np.uint8)))
            _step()
            return torch.stack(res) if res else per

        # fill_holes over a long clip's frames is slow enough to watch
        _total(sum(len(clip) for clip in masks) if _per_frame(masks) else len(masks))
        if _per_frame(masks):
            return {"masks": [[fix(f) for f in clip] for clip in masks]}
        return {"masks": [fix(per) for per in masks]}


class TrackObjects(Node):
    """Give SAM3's per-frame detections an identity: the same object keeps one
    track id for as long as it stays in the clip.

    Wire SAM3's per-frame `boxes` in (its video path), plus its `labels` so a
    "person" never gets matched to a "truck" when the Text Prompt named several
    concepts. `ids` comes out aligned 1:1 with those boxes — per clip, per
    frame, one int per box — so View Video colors and numbers each track, and
    Count Objects counts every object once instead of once per frame it shows up in.

    Association is by box overlap: each detection is matched to the track whose
    next box it overlaps most (optimal assignment per frame, not first-come),
    above `iou_threshold`. "Next" is the track's last box carried forward at the
    speed it has been moving, so an object that keeps going while it's hidden is
    still recognized when it reappears — a track survives `max_lost` frames of
    absence (occlusion, a missed detection) before it's closed. A track seen in
    fewer than `min_frames` frames is flicker rather than an object: those come
    out as id -1, drawn grey and never counted.
    # ponytail: constant velocity, no Kalman noise model, and the first two
    # frames of a track have no velocity yet — an object moving further than its
    # own width per frame (fast motion at a low Load Video fps) never gets
    # started and needs a lower iou_threshold. Upgrade path if that bites:
    # transformers' Sam3VideoModel, which tracks with memory attention over the
    # frames instead of box geometry.
    """

    kind = "track"
    label = "Track Objects"
    category = "Track"  # not Segment: it refines no masks, it adds identity to detections
    color = "#db2777"  # detection-pipeline pink, shared with the SAM3 nodes it hangs off
    modalities = ["video"]
    inputs = ["boxes", "labels"]  # labels optional: keeps association inside one concept
    outputs = ["ids"]

    iou_threshold: float = Field(0.3, description="Minimum box overlap (IoU) between frames to call it the same object. Raise it if different objects get merged, lower it for fast-moving objects.")
    max_lost: int = Field(5, description="Frames a track may go unseen before it is closed; a later match starts a new id.")
    min_frames: int = Field(2, description="Tracks seen in fewer frames than this count as flicker and get id -1.")

    def run(self, boxes, labels=None):
        import torch
        from scipy.optimize import linear_sum_assignment
        from torchvision.ops import box_iou

        if not _per_frame(boxes):
            raise ValueError("Track Objects needs per-frame boxes — wire a video into SAM3 (identity needs frames)")
        if labels is not None and not _per_instance(labels):
            raise ValueError("wire SAM3's labels output here (one name per object), not per-item class labels")
        if labels is not None and _frame_counts(labels) != _frame_counts(boxes):
            raise ValueError("labels don't line up with the boxes — wire both ports from the same SAM3 node")

        out = []
        _total(sum(len(clip) for clip in boxes))  # one assignment problem per frame
        for ci, clip in enumerate(boxes):
            names = labels[ci] if labels is not None else None
            tracks = []  # one dict per track: last box, concept, last frame seen, frames seen
            per_frame = []
            for t, bx in enumerate(clip):
                assign = [-1] * len(bx)  # detection j -> index in `tracks`
                # +1: a track seen in the previous frame has gone unseen for 0 frames
                live = [k for k, tr in enumerate(tracks) if t - tr["last"] <= self.max_lost + 1]
                if live and len(bx):
                    # where each track should be now: its last box carried forward
                    # at its own speed (0 for a brand-new one) over the frames since
                    pred = torch.stack([tracks[k]["box"] + tracks[k]["vel"] * (t - tracks[k]["last"]) for k in live])
                    # nan_to_num: two single-pixel boxes have zero area both sides,
                    # and 0/0 would blow up linear_sum_assignment
                    iou = box_iou(bx.float(), pred).nan_to_num()
                    if names is not None:  # zero out cross-concept pairs: never match person <-> truck
                        iou = iou * torch.tensor(
                            [[names[t][j] == tracks[k]["name"] for k in live] for j in range(len(bx))],
                            dtype=iou.dtype,
                        )
                    for j, k in zip(*linear_sum_assignment(-iou.numpy())):
                        # > 0 as well as the threshold: zero overlap is never the same
                        # object, and it's exactly what the cross-concept mask writes —
                        # so iou_threshold=0 must not wave both of those through
                        if iou[j, k] > 0 and iou[j, k] >= self.iou_threshold:
                            assign[j] = live[k]
                for j, b in enumerate(bx):
                    box = b.float()
                    if assign[j] < 0:  # nothing to match: a new track starts on this detection
                        name = names[t][j] if names is not None else ""
                        tracks.append({"box": box, "vel": torch.zeros(4), "name": name, "last": t, "n": 0})
                        assign[j] = len(tracks) - 1
                    tr = tracks[assign[j]]
                    # half-and-half so one jumpy frame can't poison the prediction
                    tr["vel"] = 0.5 * tr["vel"] + 0.5 * (box - tr["box"]) / max(t - tr["last"], 1)
                    tr["box"], tr["last"], tr["n"] = box, t, tr["n"] + 1
                per_frame.append(assign)
                _step()
            # tracks that survive min_frames get numbered 1..K in order of first
            # appearance (stable, readable ids); the flicker keeps -1
            keep = {}
            for k, tr in enumerate(tracks):
                if tr["n"] >= self.min_frames:
                    keep[k] = len(keep) + 1
            out.append([torch.tensor([keep.get(k, -1) for k in f], dtype=torch.int32) for f in per_frame])
        return {"ids": out}


class FilterRegion(Node):
    """Keep only the objects inside a region of interest — count pedestrians in
    one part of the street, ignore the rest of the frame. Detection still runs
    on the whole frame; this drops every box whose center falls outside the
    region drawn on the node in the frontend (stored as "x1,y1,x2,y2" source
    pixels, drawn on the first image/frame like Crop's box — one static region,
    so it fits a fixed camera). Center-inside, not overlap: an object straddling
    the border counts once it is mostly in.

    Wire a detector's `boxes` in — per-image, or SAM3's per-frame video path —
    and `masks`/`labels`/`ids` follow the kept objects when wired. Put it after
    Track Objects to count the objects that ever entered the region (ids stay
    stable while they're outside), or before it to track inside the region
    only. The optional image/video input is just the drawing reference, like
    Visual Prompt's.
    # ponytail: axis-aligned rectangle only; polygon zones (streets are rarely
    # rectangles) need a polygon drawing surface first.
    """

    kind = "filter_region"
    label = "Filter by Region"
    category = "Analyze"  # zone analytics: it feeds Count Objects, not the pixels
    color = "#db2777"  # detection-pipeline pink: it refines SAM3/Track output
    inputs = ["image", "video", "boxes", "masks", "labels", "ids"]
    required = ["boxes"]
    outputs = ["boxes", "masks", "labels", "ids"]

    box: str = ""  # the region, "x1,y1,x2,y2" in source pixels, drawn on the node

    def run(self, image=None, video=None, boxes=None, masks=None, labels=None, ids=None):
        import torch

        if not self.box.strip():
            raise ValueError("Filter by Region has no region — load the input on the node and drag a box")
        coords = self.box.replace(" ", "").split(",")
        if len(coords) != 4:
            raise ValueError('region must be "x1,y1,x2,y2"')
        x1, x2 = sorted((float(coords[0]), float(coords[2])))
        y1, y2 = sorted((float(coords[1]), float(coords[3])))
        if image is not None and video is not None:
            raise ValueError("connect either an image or a video input, not both")
        if labels is not None and not _per_instance(labels):
            raise ValueError("wire the detector's labels output here (one name per object), not per-item class labels")

        per_frame = _per_frame(boxes)
        if video is not None:
            _check_per_frame(video, boxes, "boxes")
        if image is not None:
            _check_per_image(boxes, "boxes")
        if ids is not None and not per_frame:
            raise ValueError("ids are per-frame — wire SAM3's video-path boxes in alongside them")
        # masks/labels/ids came from the same node as the boxes, so their
        # instance counts line up exactly; a mismatch means two different
        # detectors got wired in and the filtered result would be nonsense
        counts = _frame_counts(boxes) if per_frame else [len(b) for b in boxes]
        wired = [(w, v) for w, v in (("masks", masks), ("labels", labels), ("ids", ids)) if v is not None]
        for what, vals in wired:
            try:
                ok = (_frame_counts(vals) if per_frame else [len(v) for v in vals]) == counts
            except TypeError:  # per-image value where per-frame is expected (or vice versa)
                ok = False
            if not ok:
                raise ValueError(f"{what} don't line up with the boxes — wire both ports from the same node")

        def inside(bx):  # keep-mask for one (N,4) box tensor: center in the region
            cx, cy = (bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2
            return (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)

        def take(vals, keep):
            if torch.is_tensor(vals):  # boxes/masks/ids index by the bool mask
                return vals[keep]
            return [v for v, k in zip(vals, keep.tolist()) if k]  # per-instance labels

        outs = [("boxes", boxes)] + wired  # only claim the ports that were wired in
        if per_frame:
            keep = [[inside(bx) for bx in clip] for clip in boxes]
            return {w: [[take(f, k) for f, k in zip(clip, kc)] for clip, kc in zip(v, keep)] for w, v in outs}
        keep = [inside(bx) for bx in boxes]
        return {w: [take(x, k) for x, k in zip(v, keep)] for w, v in outs}


class Classify(HfNode):
    """Classify each image (or clip); the settings dropdown picks the model.

    SigLIP 2 (google/siglip2-base-patch16-224, the default) and other
    open-vocabulary checkpoints (CLIP, SigLIP, ...) are zero-shot: wire a Text
    Prompt in ("cat, dog, car") — or several Text Prompt nodes, which merge
    comma-joined — and each comma-separated phrase becomes one candidate
    label; without one they raise. SigLIP uses a sigmoid head, so its scores
    are independent per-label probabilities (they need not sum to 1). The one
    label set is broadcast across the whole batch; the HF processor takes the
    image list natively. Fixed-vocabulary checkpoints (the ViT preset's
    ImageNet classes, ResNet, ...) need no prompt and run as-is, ignoring any
    wired one. Wire `video` instead for one score set per clip — the mean of
    its frames' scores. Outputs per-item `classification` (list length B of
    {labels, scores})."""

    kind = "classify"
    label = "Classify"
    category = "Classify"
    color = "#ea580c"
    inputs = ["image", "video", "prompts"]
    required = []  # prompts only for zero-shot models; image or video either-or — both checked in run
    outputs = ["classification"]

    model_id: str = Field(
        "google/siglip2-base-patch16-224",
        json_schema_extra={"options": [
            {"value": "google/siglip2-base-patch16-224", "label": "SigLIP 2 (zero-shot)"},
            {"value": "google/vit-base-patch16-224", "label": "ViT (ImageNet classes)"},
        ], "hf_search": ["image-classification", "zero-shot-image-classification"]},
    )

    def run(self, image=None, video=None, prompts=None):
        import torch

        labels = [t.strip() for t in ((prompts or {}).get("text") or "").split(",") if t.strip()]
        batch, port = _image_or_video(image, video)
        clips = batch if port == "video" else [im.unsqueeze(0) for im in batch]  # an image is a 1-frame clip
        frames = [f for clip in clips for f in clip]

        if self.model_id == "google/siglip2-base-patch16-224":
            # dedicated body for the exact default id: it carries the model
            # card's padding="max_length" training requirement
            if not labels:
                raise ValueError("SigLIP 2 needs at least one text prompt (comma-separated labels)")
            _total(len(frames))  # before the model load: an empty bar beats no bar while weights come in
            model, proc = _siglip_model()

            def classify(sub):
                # padding="max_length": SigLIP 2 was trained with it (per the HF model card)
                inputs = proc(images=_batch_to_hwc(sub), text=labels, padding="max_length", return_tensors="pt").to(_device())
                with torch.inference_mode():
                    return list(model(**inputs).logits_per_image.sigmoid().cpu())

            probs = torch.stack(_chunked(frames, classify))  # (total frames, L)
            out, i = [], 0
            for clip in clips:  # mean over each clip's frames; a 1-frame clip is just its scores
                row = probs[i : i + len(clip)].mean(0)
                out.append({"labels": labels, "scores": [float(s) for s in row]})
                i += len(clip)
            return {"classification": out}

        from PIL import Image
        from transformers.models.auto.modeling_auto import MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES as ZS

        zero_shot = _hf_zero_shot(self.model_id, ZS, self.trust_remote_code)
        if zero_shot and not labels:
            raise ValueError(
                f"Classify: {self.model_id!r} is a zero-shot model — wire a Text Prompt (comma-separated labels)"
            )
        _total(len(frames))  # before the model load: an empty bar beats no bar
        pipe = _hf_pipeline("zero-shot-image-classification" if zero_shot else "image-classification", self.model_id, self.trust_remote_code)
        pil = [Image.fromarray(f) for f in _batch_to_hwc(frames)]

        def classify(sub):
            kw = {"candidate_labels": labels} if zero_shot else {}
            return pipe(sub, **kw)

        rows = _chunked(pil, classify)  # one [{label, score}, ...] list per frame
        out, i = [], 0
        for clip in clips:  # mean over each clip's frames; a 1-frame clip is just its scores
            per_frame, i = rows[i : i + len(clip)], i + len(clip)
            # ponytail: fixed-vocab results are top-k per frame, so label sets can
            # differ within a clip — sum scores by name (absent frames score 0)
            # and average over the clip, sorted best first like the pipeline's own
            acc: dict[str, float] = {}
            for r in per_frame:
                for det in r:
                    acc[det["label"]] = acc.get(det["label"], 0.0) + float(det["score"])
            names = sorted(acc, key=acc.__getitem__, reverse=True)
            out.append({"labels": names, "scores": [acc[n] / len(per_frame) for n in names]})
        return {"classification": out}


class Embed(HfNode):
    """Embed each image with a vision encoder: one feature vector per image.
    The settings dropdown picks the model — DINOv2 base (facebook/dinov2-base,
    768-d, the default), DINOv2 large, or any Hugging Face AutoModel encoder
    id; no prompt, no labels needed. Uses the checkpoint's pooler output where
    it has one; otherwise the CLS token, or the mean over tokens for encoders
    configured for mean pooling (e.g. BEiT-family models). The HF processor
    takes the image list natively and resizes/pads inside. Wire `video`
    instead for one vector per clip — the mean of its frame embeddings (the
    standard cheap video baseline), so Load Video's per-clip labels stay
    aligned for Train Classifier downstream. Outputs `embedding`, list length
    B of (D,) float32."""

    kind = "embed"
    label = "Embed"
    category = "Embed"
    color = "#0891b2"
    inputs = ["image", "video"]
    required = []  # exactly one of the two, checked in run
    outputs = ["embedding"]

    # no bare CLIP/SigLIP presets: AutoModel returns their dual encoder, which
    # fails on pixel-values-only input
    model_id: str = Field(
        "facebook/dinov2-base",
        json_schema_extra={"options": [
            {"value": "facebook/dinov2-base", "label": "DINOv2 base"},
            {"value": "facebook/dinov2-large", "label": "DINOv2 large"},
        ], "hf_search": ["image-feature-extraction"]},
    )

    def run(self, image=None, video=None):
        import torch

        batch, port = _image_or_video(image, video)
        clips = batch if port == "video" else [im.unsqueeze(0) for im in batch]  # an image is a 1-frame clip
        frames = [f for clip in clips for f in clip]
        _total(len(frames))  # before the model load, so the bar is up while weights come in
        model, proc = _hf_encoder(self.model_id, self.trust_remote_code)

        def embed(sub):
            inputs = proc(images=_batch_to_hwc(sub), return_tensors="pt").to(_device())
            with torch.inference_mode():
                out = model(**inputs)
            feat = getattr(out, "pooler_output", None)
            if feat is None:  # no pooler: CLS token, or token mean where configured
                lhs = out.last_hidden_state
                feat = lhs.mean(dim=1) if getattr(model.config, "use_mean_pooling", False) else lhs[:, 0]
            return list(feat.float().cpu())

        pooled = torch.stack(_chunked(frames, embed))  # (total frames, D)
        out, i = [], 0
        for clip in clips:  # mean over each clip's frames; a 1-frame clip is just its embedding
            out.append(pooled[i : i + len(clip)].mean(0))
            i += len(clip)
        return {"embedding": out}


def _umap_reduce(embedding, n_components):
    """Reduce an embedding batch to `n_components` dims with UMAP. Fixed seed
    so the layout doesn't jump between runs."""
    import torch
    import umap

    if len(embedding) <= n_components + 1:
        raise ValueError(
            f"UMAP with n_components={n_components} needs at least "
            f"{n_components + 2} images — load a bigger batch"
        )
    x = torch.stack([e.float().cpu() for e in embedding]).numpy()
    reducer = umap.UMAP(n_components=n_components, n_neighbors=min(15, len(x) - 1), random_state=42)
    return [torch.from_numpy(v) for v in reducer.fit_transform(x)]


class Umap(Node):
    """Reduce embeddings to `n_components` dims (default 2) with UMAP — wire
    Embed in and View Embeddings out for a 2-D scatter of the batch. Fixed
    seed so the layout doesn't jump between runs."""

    kind = "umap"
    label = "Reduce Dimensions"
    category = "Embed"
    color = "#0891b2"
    inputs = ["embedding"]
    outputs = ["embedding"]

    n_components: int = Field(2, description="Dimensions of the reduced embedding: 2 for the View Embeddings scatter; higher keeps more structure for downstream models but needs a bigger batch.")

    def run(self, embedding):
        return {"embedding": _umap_reduce(embedding, self.n_components)}


def _parse_labels(text):
    """Label list from LogReg's config string. Multi-line text (a pasted or
    uploaded .txt/.csv, no header) is one label per line — the last comma field
    of each line wins, so "filename,label" CSV rows work too. A single line is
    comma-separated: "cat,cat,dog"."""
    text = text.strip()
    if not text:
        return []
    if "\n" in text:
        return [ln.split(",")[-1].strip() for ln in text.splitlines()]
    return [s.strip() for s in text.split(",")]


class LogReg(Node):
    """Fit a logistic regression on the embeddings of the labeled images, then
    predict class probabilities for the whole batch. Labels are per image, in
    batch order; empty entries mean predict-only for that image. They come from
    the optional `labels` input port (e.g. Load Image's folder labels) or, when
    that is unwired or all-empty, from the `labels` config string — typed
    ("cat,cat,dog,dog" trains on the first four, predicts the rest) or loaded
    from a .txt/.csv file (see _parse_labels). Don't reorder or filter the batch
    between labeling and here, or labels misalign. Outputs the same
    `classification` shape as SigLIP 2, so View Classification and Filter by
    Class work downstream."""

    kind = "logreg"
    label = "Train Classifier"
    category = "Classify"
    color = "#ea580c"
    inputs = ["embedding", "labels"]  # labels optional (config string fallback)
    outputs = ["classification"]

    labels: str = ""  # per-image training labels; empty entry = predict-only

    def run(self, embedding, labels=None):
        import torch
        from sklearn.linear_model import LogisticRegression

        _check_class_labels(labels)
        x = torch.stack([e.float().cpu() for e in embedding]).numpy()
        wired = [str(s).strip() for s in labels] if labels is not None else []
        given = wired if any(wired) else _parse_labels(self.labels)
        while given and not given[-1]:
            given.pop()
        if len(given) > len(x):
            raise ValueError(f"{len(given)} labels for {len(x)} images — one comma-separated entry per image")
        given += [""] * (len(x) - len(given))
        train = [i for i, s in enumerate(given) if s]
        classes = sorted({given[i] for i in train})
        if not classes:
            raise ValueError(
                "no training labels — wire Load Image's labels output (pick a folder with one "
                "subfolder per class) or set labels in the node settings, typed or from a .txt/.csv file"
            )
        if len(classes) < 2:
            raise ValueError(f"all labels are {classes[0]!r} — need at least two classes to fit")
        clf = LogisticRegression(max_iter=1000).fit(x[train], [given[i] for i in train])
        probs = clf.predict_proba(x)  # predict every image, labeled ones included
        names = [str(c) for c in clf.classes_]
        return {"classification": [{"labels": names, "scores": [float(s) for s in row]} for row in probs]}


def _normal_indices(labels, n, normal_label):
    """Which batch items are the known-good reference set: the ones labeled
    `normal_label` (Load Image's folder labels, e.g. a good/ subfolder), or the
    whole batch when no label is named — then nothing is known to be good and
    the scores can only rank it. Under two references there is no spread to
    calibrate a threshold against, so that raises."""
    want = normal_label.strip()
    if not want:
        ref = list(range(n))
    else:
        have = [str(s).strip() for s in (labels or [])]
        ref = [i for i, s in enumerate(have) if s == want]
        if not ref:
            raise ValueError(
                f"no images labeled {want!r} — wire Load Image's labels (pick a folder with a "
                f"{want}/ subfolder); this batch has {sorted(set(have) - {''}) or 'no labels'}"
            )
    if len(ref) < 2:
        raise ValueError("need at least 2 known-good images to learn what normal looks like")
    return ref


def _anomaly_scores(raw, calib, tolerance):
    """Raw anomaly distances -> the `classification` shape, so defects reuse the
    Classify plumbing (View Classification, Filter by Class) instead of a port
    of their own.

    The threshold is the worst of the `calib` items — good ones scored without
    themselves counting as their own reference, so it reads as "as unusual as a
    good part gets" — times `tolerance`. Scores are raw/(raw+threshold):
    monotone in raw, so the ranking is the raw one, and exactly 0.5 at the
    threshold, so >= 0.5 means defect."""
    thr = max(max(float(raw[i]) for i in calib) * tolerance, 1e-9)
    return [
        {"labels": ["defect", "normal"], "scores": [v / (v + thr), thr / (v + thr)]}
        for v in (float(x) for x in raw)
    ]


class AnomalyScore(Node):
    """Score how unusual each item looks next to known-good ones — defect
    detection that trains on good parts only, no defect examples needed.

    Wire DINOv2's `embedding` in plus Load Image's `labels`, and name the class
    marking the good parts in `normal_label` (e.g. "good" for a good/
    subfolder). Each item scores the mean distance to its `k` nearest good ones
    (kNN, PatchCore's idea without the patch grid), never to itself. Leave
    `normal_label` empty and the whole batch is the reference: nothing is known
    to be good, so the scores only rank it ("find the odd ones out").

    Outputs the same `classification` shape as SigLIP 2 — labels "defect" and
    "normal", 0.5 at the calibrated threshold — so Filter by Class orders the
    batch most-unusual-first (and drops what passes) while View Classification
    shows the scores. Raise `tolerance` when good parts get flagged, lower it to
    catch subtler defects.
    # ponytail: one vector per image finds a wrong-looking part, not a small
    # scratch on an otherwise right-looking one. That needs patch-level scoring
    # (Find Defects (PatchCore)) or a Crop around the region of interest.
    """

    kind = "anomaly_score"
    label = "Score Anomalies"
    category = "Classify"
    color = "#ea580c"
    inputs = ["embedding", "labels"]  # labels optional: unlabeled = rank the batch
    outputs = ["classification"]

    normal_label: str = Field("", description="Label marking the known-good items (e.g. \"good\"). Empty = the whole batch is the reference, so scores only rank the odd ones out.")
    k: int = Field(1, description="Each item is scored by its mean distance to the k nearest good items. Raise it to smooth out the influence of any single reference image.")
    tolerance: float = Field(1.0, description="Scales the defect threshold (the worst good item's score). Raise it when good parts get flagged, lower it to catch subtler defects.")

    def run(self, embedding, labels=None):
        import torch

        _check_class_labels(labels)
        ref = _normal_indices(labels, len(embedding), self.normal_label)
        x = torch.stack([e.float().cpu() for e in embedding])
        d = torch.cdist(x, x[ref])
        # nothing is its own nearest neighbour: the good items are scored too —
        # that is what calibrates the threshold — and a self-distance of 0 would
        # make every one of them look perfect
        for j, i in enumerate(ref):
            d[i, j] = float("inf")
        raw = d.topk(min(max(self.k, 1), len(ref) - 1), largest=False).values.mean(1)
        return {"classification": _anomaly_scores(raw, ref, self.tolerance)}


_CORESET_RATIO = 0.1  # share of the good parts' patches kept in the memory bank (anomalib's default)


def _patchcore():
    """Anomalib's PatchCore — wide_resnet50_2 patch features and a coreset
    memory bank — with anomalib's own preprocessing (256x256, ImageNet
    normalization). Built per run, unlike the other models here: the memory
    bank is fitted to that run's good parts, so there is nothing to keep across
    runs but the backbone weights (timm caches those on disk)."""
    from anomalib.models.image.patchcore.lightning_model import Patchcore
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    model = PatchcoreModel(layers=["layer2", "layer3"]).to(_device())
    return model, Patchcore.configure_pre_processor().transform


class PatchCore(Node):
    """Find defects *and where they are*: PatchCore (Intel's Anomalib), trained
    on good parts only — the industrial-inspection standard.

    Wire the images in plus Load Image's `labels`, and name the class marking
    the good parts in `normal_label` (e.g. "good" for a good/ subfolder); with
    it empty the whole batch is the reference and the scores only rank it. The
    good parts' patch embeddings become a memory bank; every image is then
    scored by how far its worst patch sits from anything in there, which is why
    a scratch on an otherwise correct part registers where Score Anomalies'
    one-vector-per-image distance would miss it.

    `classification` comes out in SigLIP 2's shape ("defect" / "normal", 0.5 at
    the threshold) so Filter by Class orders the batch most-unusual-first and
    View Classification shows the scores, plus `masks`/`boxes` marking the
    defect region — wire those into View Image to see it. A part that scores
    clean has no region and gets neither.

    `reference_images` bounds how many good parts the bank is built from (they
    are spread evenly over the ones available, not the first N): each costs
    ~6 MB of VRAM and the coreset pass over them is quadratic, so the default
    200 — the regime PatchCore is tuned for — keeps a folder of 1000 good parts
    from asking 10 GB. Raise it on a bigger card if coverage looks thin.
    # ponytail: one region per image (the whole above-threshold area, not
    # separate blobs), and the coreset ratio is anomalib's default rather than a
    # knob. Split the region per connected component if that ever matters.
    """

    kind = "patchcore"
    label = "Find Defects"
    category = "Classify"
    color = "#ea580c"
    modalities = ["image"]
    inputs = ["image", "labels"]  # labels optional (but that leaves nothing known-good)
    outputs = ["classification", "masks", "boxes"]

    normal_label: str = Field("", description="Label marking the known-good items (e.g. \"good\"). Empty = the whole batch is the reference, so scores only rank the odd ones out.")
    tolerance: float = Field(1.0, description="Scales the defect threshold (the worst held-out good part's score). Raise it when good parts get flagged, lower it to catch subtler defects.")
    reference_images: int = Field(200, description="How many good parts the memory bank is built from (spread evenly over those available). Each costs ~6 MB VRAM and fitting is quadratic — raise on a big card if coverage looks thin.")

    def run(self, image, labels=None):
        import torch
        from torchvision.transforms.v2.functional import resize

        _check_class_labels(labels)
        ref = _normal_indices(labels, len(image), self.normal_label)
        # every fifth good part stays out of the memory bank: scored against the
        # rest, those show how unusual a good part gets, which is the threshold.
        # With their own patches in the bank they would score near 0 and put the
        # threshold so low that every good part comes out a defect.
        held = set(ref[::5])
        bank = [i for i in ref if i not in held]
        # A good part costs ~6 MB of VRAM in the bank (1024 patches x 1536 dims)
        # and the coreset pass over it is quadratic, so a folder of 1000 good
        # parts wants 10 GB — the store, then vstack's contiguous copy of it —
        # and minutes of greedy selection. Cap the count, spread evenly over the
        # set rather than taking the first N: folders tend to be ordered by
        # capture session, and the first 200 would then all be one session.
        cap = max(self.reference_images, 2)
        if len(bank) > cap:
            bank = [bank[i * len(bank) // cap] for i in range(cap)]
        # before the model load: an empty bar beats no bar while the backbone comes in
        _total(len(bank) + len(image))  # the memory bank pass, then every image scored
        model, tf = _patchcore()

        def prep(sub):  # batch item indices -> the (n,3,256,256) float tensor the model wants
            return torch.stack([tf(image[i].float() / 255) for i in sub]).to(_device())

        def collect(sub):
            model(prep(sub))  # training mode: the model keeps the patch embeddings itself
            return []

        def score(sub):
            out = model(prep(sub))
            return list(zip(out.pred_score.cpu(), out.anomaly_map.cpu()))

        # no_grad rather than inference_mode: the collected patch embeddings
        # outlive the block as the memory bank
        with torch.no_grad():
            model.train()  # PatchCore "training" is just collecting the good parts' patches
            _chunked(bank, collect)
            model.subsample_embedding(_CORESET_RATIO)
            model.eval()
            scored = _chunked(list(range(len(image))), score)

        maps = [m for _, m in scored]
        scores = _anomaly_scores([s for s, _ in scored], held, self.tolerance)
        # the level a good part's worst patch reaches — where "unusual" starts
        good_level = max(max(float(maps[i].max()) for i in held) * self.tolerance, 1e-9)
        boxes, masks = [], []
        for im, m, c in zip(image, maps, scores):
            h, w = im.shape[-2:]
            m = resize(m, [h, w])  # the map is 256x256 (what the model saw) — back to this image's size
            # only a part that scored as a defect gets a region, and it's the half
            # of the map's climb from that level to this part's own worst point:
            # the whole map sits well above zero, so a plain fraction of the peak
            # would flood the image, and the bare level marks a diffuse blob
            hot = [m >= (good_level + float(m.max())) / 2] if c["scores"][0] >= 0.5 else []
            bx, ms, _ = _instances_to_boxes_masks(hot, h, w)
            boxes.append(bx)
            masks.append(ms)
        return {"classification": scores, "masks": masks, "boxes": boxes}


class ShowEmbeddings(OutputNode):
    """Scatterplot of the batch's embeddings, one point per item, in batch
    order. Input over 2-D is reduced to 2-D internally (UMAP, fixed seed) —
    wire Reduce Dimensions in front only to control that reduction or reuse
    its output downstream. Optionally wire the embedded image batch into
    `image` — or the embedded clip batch into `video` (first frame per clip)
    — to get a small per-point thumbnail shown when hovering a point."""

    kind = "show_embeddings"
    label = "View Embeddings"
    inputs = ["embedding", "image", "video"]  # image/video optional: hover thumbnails

    def run(self, embedding, image=None, video=None):
        if image is not None and video is not None:
            raise ValueError("wire image or video into the thumbnails, not both")
        if video is not None:
            image = [clip[0] for clip in video]  # a clip's thumbnail is its first frame
        if any(e.numel() != 2 for e in embedding):
            embedding = _umap_reduce(embedding, 2)
        out = {"points": [[float(v) for v in e.flatten()] for e in embedding]}
        if image is not None:
            from torchvision.transforms.v2 import functional as F

            if len(image) != len(embedding):
                raise ValueError(
                    f"{len(image)} images for {len(embedding)} points — wire the same batch that was embedded"
                )
            # inline, unlike the batch previews: 64 px is ~2 KB, and a hover
            # that had to fetch would lag behind the pointer
            thumbs = [F.resize(im, 64, max_size=96) for im in image]
            out["thumbnails"] = [
                "data:image/jpeg;base64," + base64.b64encode(p).decode() for p in batch_to_previews(thumbs)
            ]
        return out


class FilterByClass(Node):
    """Filter and/or sort a batch (images or clips) by a classifier's score
    (e.g. SigLIP 2 — "find the clips with a dog in them").

    Wire the same batch that went into the classifier plus its
    `classification` output. Items whose score for `label_name` (empty = the
    first label) falls below `threshold` are dropped; the rest are ordered by
    that score, best first, when `sort` is on. Threshold 0 + sort = pure sort;
    threshold set + sort off = pure filter.

    Wire the classifier's `masks` / `boxes` in as well and they are reordered
    and dropped with the items they belong to, so a viewer downstream still
    overlays each item's own annotations (e.g. PatchCore's defect regions on a
    most-unusual-first gallery)."""

    kind = "filter_class"
    # not worth caching: re-runs in milliseconds and its output aliases its
    # input batch, so a cache entry only burns budget pinning old generations.
    # ponytail: this un-caches downstream viewers too (sig poisoning) — re-
    # drawing a filtered view off a cached classifier is cheap; add a
    # cache_result flag that keeps descendants' sigs if that ever changes
    cacheable = False
    label = "Filter by Class"
    # not Classify: it produces no classification — batch in, batch out, with a
    # classification as a side input, exactly like Crop takes boxes and Blur masks
    category = "Transform"
    color = "#2563eb"
    inputs = ["image", "video", "classification", "masks", "boxes"]  # masks/boxes optional, travel along
    required = ["classification"]  # image or video, exactly one, checked in run
    outputs = ["image", "video", "masks", "boxes"]

    label_name: str = ""  # class to score by; empty = first label
    threshold: float = 0.0  # keep items with score >= this
    sort: bool = True  # order by score, best first

    def run(self, image=None, video=None, classification=None, masks=None, boxes=None):
        batch, port = _image_or_video(image, video)
        want = self.label_name.strip()

        def score(c):
            if want and want not in c["labels"]:
                raise ValueError(f"label {want!r} not among classifier labels {c['labels']}")
            return c["scores"][c["labels"].index(want) if want else 0]

        if len(classification) != len(batch):
            raise ValueError(
                f"{len(classification)} classifications for {len(batch)} {port}s — "
                "wire the same batch that went into the classifier"
            )
        scored = [(score(c), i) for i, c in enumerate(classification)]
        kept = [t for t in scored if t[0] >= self.threshold]
        if not kept:
            best = max(s for s, _ in scored)
            raise ValueError(
                f"no {port} scored >= {self.threshold:.2f} for {want or 'first label'!r} (best was {best:.2f})"
            )
        if self.sort:
            kept.sort(key=lambda t: t[0], reverse=True)
        order = [i for _, i in kept]
        out = {port: [batch[i] for i in order]}
        for name, vals in (("masks", masks), ("boxes", boxes)):
            if vals is not None:  # only claim the ports that were wired in
                out[name] = [vals[i] for i in order]
        return out


class ViewClassification(OutputNode):
    """Show a classifier's output as a sorted horizontal bar chart (rendered by
    the frontend). Consumes the `classification` port, e.g. from SigLIP 2."""

    kind = "view_classification"
    label = "View Classification"
    inputs = ["classification"]

    def run(self, classification):
        return {"classification": classification}


class CountObjects(OutputNode):
    """Count the instances a segmenter found, per image and across the batch —
    wire SAM3's `boxes` in (one row per image, the frontend renders the table).

    Wire SAM3's `labels` in too and the count breaks down by concept: one column
    per phrase of the Text Prompt ("person, truck" -> a person and a truck
    column). Without it — or with drawn prompts, which name no concept — there's
    a single "objects" column.

    On video, wire Track Objects' `ids` in alongside per-frame `boxes`: each
    row is then a clip and each track counts once, however many frames it
    appears in ("12 people walked past"), flicker tracks (id -1) excluded.
    Without `ids`, per-frame boxes (SAM3's video path, Detect Objects on video)
    count frame by frame — one row per frame, and the same object counts again
    in every frame it appears in.
    # ponytail: total is the sum of the per-concept counts, so overlapping
    # phrases ("car, vehicle") double-count the same object. Dedupe by box IoU
    # if that turns out to bite.
    """

    kind = "count"
    label = "Count Objects"
    # not Output: the View nodes display what they're handed, this one computes an
    # answer from boxes. Terminal all the same, so it keeps the Output color.
    category = "Analyze"
    inputs = ["boxes", "labels", "ids"]  # labels optional (concept names); ids optional (count tracks, not frames)
    required = ["boxes"]

    def run(self, boxes, labels=None, ids=None):
        names = labels if labels is not None and _per_instance(labels) else None
        if labels is not None and names is None and any(labels):
            raise ValueError("wire the detector's labels output here (one name per object), not per-image class labels")
        # names/ids come from the same node as the boxes, so a mismatch means two
        # different SAM3s got wired in and the breakdown would be nonsense
        if ids is not None:
            if not _per_frame(boxes):
                raise ValueError("tracked counts need per-frame boxes — wire SAM3's video-path boxes in, not sampled frames")
            if _frame_counts(ids) != _frame_counts(boxes):
                raise ValueError("ids don't line up with the boxes — wire Track Objects and its SAM3 into the same ports")
            if names is not None and _frame_counts(names) != _frame_counts(boxes):
                raise ValueError("labels don't line up with the boxes — wire both ports from the same SAM3 node")
            # one entry per track, under the concept of the frame it was first seen in
            per_item = []
            for ci, clip in enumerate(ids):
                concept = {}
                for t, frame in enumerate(clip):
                    for j, k in enumerate(frame.tolist()):
                        if k >= 0:  # -1 = flicker, filtered by Track Objects' min_frames
                            concept.setdefault(k, names[ci][t][j] if names is not None else "")
                per_item.append([concept[k] for k in sorted(concept)])
        else:
            if _per_frame(boxes):
                # no ids: each frame becomes its own row, so the same object
                # counts again in every frame — Track Objects' ids count it once
                if names is not None:
                    if _frame_counts(names) != _frame_counts(boxes):
                        raise ValueError("labels don't line up with the boxes — wire both ports from the same node")
                    names = [frame for clip in names for frame in clip]
                boxes = [frame for clip in boxes for frame in clip]
            if names is not None and (len(names) != len(boxes) or any(len(n) != len(b) for n, b in zip(names, boxes))):
                raise ValueError("labels don't line up with the boxes — wire both ports from the same node")
            per_item = [list(per) for per in names] if names is not None else [[""] * len(b) for b in boxes]
        concepts = sorted({n for row in per_item for n in row if n})
        if not concepts:  # drawn prompts / no labels wired: one plain count column
            per_image = [[len(row)] for row in per_item]
            return {"counts": {"concepts": ["objects"], "per_image": per_image, "total": [sum(b[0] for b in per_image)]}}
        per_image = [[sum(1 for n in row if n == c) for c in concepts] for row in per_item]
        total = [sum(row[j] for row in per_image) for j in range(len(concepts))]
        return {"counts": {"concepts": concepts, "per_image": per_image, "total": total}}


_CUSTOM_DEFAULT_CODE = "def run(**inputs):\n    return {}\n"


class CustomNode(Node):
    """Run your own Python inside the graph. Pick which ports to expose in
    `input_ports` / `output_ports` (the frontend offers the same fixed set of
    port names/types every other node uses, so handles line up and connect
    normally), then write a top-level `run(**inputs) -> dict` in `code`.

    An optional top-level `load()` is the hook for heavy state — your own
    Hugging Face model, anything the built-in model nodes can't handle. It
    runs once per code version (cached like any other HF model, keyed by a
    hash of `code` so editing the code reloads) and its return is passed to
    run as the first argument: `run(loaded, **inputs)`.

    Caching assumes run() is pure: an impure run() (random/network/file reads)
    serves stale cached results while its code and inputs are unchanged —
    POST /cache/clear is the remedy.

    # ponytail: `code` runs via exec() in-process, unsandboxed, with the
    # server's own privileges (filesystem, network, everything). Fine for a
    # trusted local tool; add real sandboxing (subprocess + resource limits,
    # nsjail, ...) before ever exposing this past localhost or to other users.
    """

    kind = "custom"
    label = "Custom Code"
    category = "Custom"
    color = "#57534e"
    inputs: ClassVar[list[str]] = []
    outputs: ClassVar[list[str]] = []

    input_ports: list[str] = []
    output_ports: list[str] = []
    code: str = _CUSTOM_DEFAULT_CODE

    def run(self, **inputs):
        missing = [p for p in self.input_ports if p not in inputs]
        if missing:
            raise ValueError(f"custom node: input {missing[0]!r} not connected")

        ns: dict = {}
        exec(self.code, ns)
        fn = ns.get("run")
        if not callable(fn):
            raise ValueError("custom node code must define a top-level `run(**inputs)` function")
        if callable(ns.get("load")):
            import hashlib

            key = ("custom", hashlib.sha1(self.code.encode()).hexdigest())
            if key not in _MODELS:
                _MODELS[key] = ns["load"]()
            result = fn(_MODELS[key], **inputs)
        else:
            result = fn(**inputs)

        if self.output_ports and (not isinstance(result, dict) or any(p not in result for p in self.output_ports)):
            raise ValueError(f"custom node: run() must return a dict containing {self.output_ports}")
        return result


class Note(Node):
    """A sticky note on the canvas — no ports, never runs, purely for the reader.

    It carries only its text: with no output ports it's never a run target and
    with no inputs nothing evaluates it, so the graph engine never sees it.
    """

    kind = "note"
    label = "Note"
    category = "Custom"
    color = "#ca8a04"

    text: str = ""


# RGB colors cycled per instance for overlays.
_PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def _hex_rgb(h):
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))  # "#rrggbb" -> (r, g, b)


def _track_color(i):
    """One RGB per track id, golden-angle hue rotation: consecutive ids never
    look alike and the palette can't run out (unlike _PALETTE's six). Tracks
    filtered as flicker (id -1) are grey."""
    import colorsys

    if i < 0:
        return (128, 128, 128)
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.9, 1.0)
    return (round(r * 255), round(g * 255), round(b * 255))


def _text_size(img):
    return max(12, round(img.shape[0] / 30))  # still readable after the webm preview downscale


def _draw_texts(img, items):
    """Write (x, y, text, color) items onto the frame — PIL is the text renderer
    we have (torchvision's own draw_bounding_boxes uses it too)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    draw = ImageDraw.Draw(pil := Image.fromarray(img))
    font = ImageFont.load_default(size=_text_size(img))
    for x, y, text, color in items:
        # stroke: the hue alone vanishes on same-colored footage
        draw.text((x, y), text, fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    return np.array(pil)  # copy, not a view: torch.from_numpy needs it writable


def _color_key(j, ids=None, names=None):
    """The handle a hand-picked color is looked up by: instance j's label when
    it has one (per-label coloring is the common ask — "all dogs red"), else
    its track id, else its per-frame index. Shared by _draw_overlays and
    _build_legend so a legend chip and the pixels can't disagree."""
    if names is not None and j < len(names) and names[j]:
        return names[j]
    if ids is not None:
        return str(int(ids[j]))
    return str(j)


def _draw_overlays(img, boxes=None, masks=None, ids=None, names=None, item_label=None, colors=None):
    """Tint instance masks and draw 2px box borders into one (H,W,3) uint8 array
    in place; instance j cycles _PALETTE, so a mask and its box share a color.
    With track `ids` wired the color comes from the id instead — one hue per
    track, held across frames. `colors` (the View node's legend picks) overrides
    both: a {_color_key: "#rrggbb"} dict — keys not picked keep the default.
    Text goes above each instance's box (or its mask's top-left when boxes
    aren't wired): the track id, the instance's `names[j]`, or both; flicker
    tracks (-1) get no text, just their grey box. `item_label` is the batch
    item's class label, written in the corner."""
    import numpy as np

    h, w = img.shape[:2]
    auto = (lambda j: _track_color(int(ids[j]))) if ids is not None else (lambda j: _PALETTE[j % len(_PALETTE)])

    def color_of(j):
        c = colors.get(_color_key(j, ids, names)) if colors else None
        return _hex_rgb(c) if c else auto(j)
    masks = masks if masks is not None else []
    if len(masks):
        # one index-map pass instead of N sequential boolean-index blends —
        # overlapping masks show the topmost tint instead of stacking 50% each
        idxmap = np.full((h, w), -1, dtype=np.int32)
        for j, m in enumerate(masks):
            idxmap[m.cpu().numpy().astype(bool)] = j
        lut = np.array([color_of(j) for j in range(len(masks))], dtype=np.float32)
        sel = idxmap >= 0
        img[sel] = (0.5 * img[sel] + 0.5 * lut[idxmap[sel]]).astype(np.uint8)
    for j, b in enumerate(boxes if boxes is not None else []):
        x1, y1, x2, y2 = (int(v) for v in b.tolist())
        color = color_of(j)
        t = 2  # border thickness
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)
        img[y1 : y1 + t, x1 : x2 + 1] = color  # top
        img[y2 - t + 1 : y2 + 1, x1 : x2 + 1] = color  # bottom
        img[y1 : y2 + 1, x1 : x1 + t] = color  # left
        img[y1 : y2 + 1, x2 - t + 1 : x2 + 1] = color  # right
    size = _text_size(img)
    texts = [(4, 2, item_label, (255, 255, 255))] if item_label else []
    for j in range(len(boxes) if boxes is not None else len(masks) if masks is not None else 0):
        tid = int(ids[j]) if ids is not None else None
        if tid is not None and tid < 0:
            continue
        parts = ([str(tid)] if tid is not None else []) + ([names[j]] if names is not None and names[j] else [])
        if not parts:
            continue
        if boxes is not None:
            x, y = int(boxes[j][0]), int(boxes[j][1])
        else:  # no box to sit above: the mask's top-left
            ys, xs = np.nonzero(masks[j].cpu().numpy())
            x, y = (int(xs.min()), int(ys.min())) if len(xs) else (0, 0)
        texts.append((x, max(0, y - size), " ".join(parts), color_of(j)))
    return _draw_texts(img, texts) if texts else img


def _check_hex_colors(colors):
    """A View node's `colors` config values must be '#rrggbb' — _hex_rgb would
    draw garbage from anything else (hand-POSTed graphs; the UI picker can't
    produce bad values)."""
    import re

    for c in colors.values():
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", c):
            raise ValueError(f"not a '#rrggbb' color: {c!r}")


def _thumb_url(im, x1, y1, x2, y2, side=96):
    """Crop one instance's box out of a (3,H,W) uint8 tensor -> /media URL of a
    small JPEG chip for the View legend. Content-addressed like Export's zip,
    so re-runs of an unchanged graph reuse the file."""
    import hashlib

    h, w = im.shape[-2:]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    jpeg = batch_to_previews([im[:, y1 : y2 + 1, x1 : x2 + 1]], max_side=side)[0]
    MEDIA.mkdir(exist_ok=True)
    dest = MEDIA / f"{hashlib.sha1(jpeg).hexdigest()[:20]}.jpg"
    if not dest.exists():
        dest.write_bytes(jpeg)
    return f"/media/{dest.name}"


def _build_legend(frames, colors):
    """What the View node detected, one entry per distinct color key (label /
    track id / instance index — see _color_key), first appearance wins:
    {key, color: the hex it's actually drawn in, thumb: /media JPEG crop of
    that first instance}. The frontend renders this as pick-a-color chips
    under the result, so the user maps object -> color by looking at it, never
    by guessing indices. `frames`: (chw_image, boxes, masks, ids, names)
    tuples, one per image or video frame.

    Capped at 50 chips (unnamed instances can number in the hundreds, each
    otherwise costing a thumbnail crop) — every frame is still scanned past
    the cap, but distinct keys beyond it are only counted. Returns
    (entries, more_count)."""
    import torch

    CAP = 50
    entries, skipped = {}, set()
    for im, bxs, mks, ids, names in frames:
        n = len(bxs) if bxs is not None else len(mks) if mks is not None else 0
        for j in range(n):
            if ids is not None and int(ids[j]) < 0 and not (names is not None and names[j]):
                continue  # flicker track: drawn grey, nothing stable to pick
            key = _color_key(j, ids, names)
            if key in entries:
                continue
            if len(entries) >= CAP:
                skipped.add(key)
                continue
            auto = _track_color(int(ids[j])) if ids is not None else _PALETTE[j % len(_PALETTE)]
            color = colors.get(key) or "#%02x%02x%02x" % auto
            if bxs is not None:
                x1, y1, x2, y2 = (int(v) for v in bxs[j].tolist())
            else:
                ys, xs = torch.nonzero(mks[j].bool(), as_tuple=True)
                if not len(xs):
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            entries[key] = {"key": key, "color": color.lower(), "thumb": _thumb_url(im, x1, y1, x2, y2)}
    return list(entries.values()), len(skipped)


class ViewImage(OutputNode):
    """Show the batch, overlaying whatever is wired: `boxes` draws rectangles,
    `masks` tints instances, `labels` writes names — per-instance labels (SAM3,
    Detect Objects) next to their box or mask, per-item class labels (Load Image's
    folder labels) in the image corner. A legend of the detected objects
    (thumbnail + label, or instance index) appears under the result, where each
    one's overlay color can be picked by hand — stored in `colors`, keyed by
    label / instance index."""

    kind = "view"
    label = "View Image"
    inputs = ["image", "boxes", "masks", "labels"]  # overlays optional

    colors: dict[str, str] = {}  # color key (see _color_key) -> "#rrggbb", picked on the legend

    def run(self, image, boxes=None, masks=None, labels=None):
        _check_hex_colors(self.colors)
        _check_per_image(boxes, "boxes")
        _check_per_image(masks, "masks")
        names = item_labels = None
        if labels is not None:
            if _per_instance(labels):
                ref = boxes if boxes is not None else masks
                if ref is None:
                    raise ValueError("per-instance labels need the boxes or masks they belong to — wire the detector's output in too")
                if [len(n) for n in labels] != [len(r) for r in ref]:
                    raise ValueError("labels don't line up with the boxes/masks — wire both ports from the same node")
                names = labels
            else:
                item_labels = labels  # one class label per image, drawn in the corner
        drawn = _batch_from_hwc([
            _draw_overlays(
                im.copy(),
                boxes[i] if boxes is not None else None,
                masks[i] if masks is not None else None,
                names=names[i] if names is not None else None,
                item_label=item_labels[i] if item_labels is not None else None,
                colors=self.colors,
            )
            for i, im in enumerate(_batch_to_hwc(image))
        ])
        legend, more = _build_legend(
            (
                (im, boxes[i] if boxes is not None else None, masks[i] if masks is not None else None,
                 None, names[i] if names is not None else None)
                for i, im in enumerate(image)
            ),
            self.colors,
        )
        res = {"image": drawn, "legend": legend}
        if more:
            res["legend_more"] = more
        return res


class ViewVideo(OutputNode):
    """Play clips in the node (the frontend pages clips; a native <video>
    streams each). Overlays whatever is wired, like View Image: per-frame
    `masks` tint instance masks, per-frame `boxes` draw AABBs — from SAM3
    the two are aligned 1:1, so a mask and its box share a color.

    Wire Track Objects' `ids` in as well to see the tracks themselves: color
    then comes from the track id instead of the per-frame instance order, so
    one object keeps one color for as long as it's followed, and its id is
    written above its box. `labels` writes names the same way — per-frame
    per-instance labels (SAM3's video path, Detect Objects on video) next to
    their box or mask, per-clip class labels (Load Video's folder labels) in
    the frame corner. A legend of the detected objects (thumbnail + label /
    track id) appears under the result, where each one's overlay color can be
    picked by hand — stored in `colors`, keyed by label / track id / instance
    index."""

    kind = "view_video"
    label = "View Video"
    inputs = ["video", "masks", "boxes", "labels", "ids"]  # overlays optional, per-frame
    modalities = ["video"]

    colors: dict[str, str] = {}  # color key (see _color_key) -> "#rrggbb", picked on the legend

    def run(self, video, masks=None, boxes=None, ids=None, labels=None):
        _check_hex_colors(self.colors)
        names = clip_labels = None
        if labels is not None:
            if isinstance(labels[0], str):
                clip_labels = labels  # one class label per clip, drawn in the corner
            else:
                names = labels
        if masks is None and boxes is None:
            if ids is not None:
                raise ValueError("ids need the boxes or masks they belong to — wire SAM3's output in too")
            if names is not None:
                raise ValueError("per-instance labels need the boxes or masks they belong to — wire the detector's output in too")
            if clip_labels is None:
                return {"video": video}
        for what, vals in (("masks", masks), ("boxes", boxes), ("ids", ids), ("labels", names)):
            if vals is not None:
                _check_per_frame(video, vals, what)
        ref = boxes if boxes is not None else masks
        if ids is not None and _frame_counts(ids) != _frame_counts(ref):
            raise ValueError("ids don't line up with the boxes/masks — wire Track Objects and its SAM3 into the same ports")
        if names is not None and _frame_counts(names) != _frame_counts(ref):
            raise ValueError("labels don't line up with the boxes/masks — wire both ports from the same node")
        out = []
        for ci, clip in enumerate(video):
            writer, h, w = _SpillWriter(), clip.shape[-2], clip.shape[-1]
            T = 0
            for fi, im in enumerate(_batch_to_hwc(list(clip))):
                drawn = _draw_overlays(
                    im.copy(),
                    boxes[ci][fi] if boxes is not None else None,
                    masks[ci][fi] if masks is not None else None,
                    ids[ci][fi] if ids is not None else None,
                    names=names[ci][fi] if names is not None else None,
                    item_label=clip_labels[ci] if clip_labels is not None else None,
                    colors=self.colors,
                )
                writer.add(drawn.transpose(2, 0, 1))
                T += 1
            out.append(writer.finish_stacked((T, 3, h, w)))
        legend, more = _build_legend(
            (
                (clip[fi],
                 boxes[ci][fi] if boxes is not None else None,
                 masks[ci][fi] if masks is not None else None,
                 ids[ci][fi] if ids is not None else None,
                 names[ci][fi] if names is not None else None)
                for ci, clip in enumerate(video)
                for fi in range(len(clip))
            ),
            self.colors,
        )
        res = {"video": out, "legend": legend}
        if more:
            res["legend_more"] = more
        return res


class ViewVolume(OutputNode):
    """Scroll through volumes slice by slice. Takes Load Volume's `video`
    clips (one clip = one volume, slices as frames) and hands each to the
    frontend as a raw grayscale volume that is resliced client-side: switch
    between axial, coronal and sagittal, slide through the slices. Works on
    any clip — a video is just a stack whose depth axis is time.

    Wire a model's per-frame `masks` in (SAM3's video path over the same
    clip) to see them in all three planes: a stack of per-slice masks is
    itself a (T,H,W) label volume, so a mask predicted axially still cuts
    correctly through the coronal and sagittal views. `ids` (Track Objects)
    keeps one color per tracked instance, `labels` names them in a legend."""

    kind = "view_volume"
    label = "View Volume"
    inputs = ["video", "masks", "ids", "labels"]  # overlays optional, per-frame
    modalities = ["video"]

    def run(self, video, masks=None, ids=None, labels=None):
        # the frontend's raw volume is main.py's clip.float().mean(1) — a
        # full-clip float32 copy (4x); refuse before that OOMs. Volumes (this
        # node's real use case, hundreds of slices) never hit it.
        avail = _mem_available()
        if avail and any(c.numel() > avail / 4 for c in video):
            raise ValueError("this clip is too long for the volume viewer — use View Video")
        if labels is not None and isinstance(labels[0], str):
            labels = None  # per-clip class labels — nothing per-instance to caption
        if masks is None:
            if ids is not None or labels is not None:
                raise ValueError("ids/labels need the masks they belong to — wire SAM3's masks in too")
            return {"video": video}
        for what, vals in (("masks", masks), ("ids", ids), ("labels", labels)):
            if vals is not None:
                _check_per_frame(video, vals, what)
        if ids is not None and _frame_counts(ids) != _frame_counts(masks):
            raise ValueError("ids don't line up with the masks — wire Track Objects and its SAM3 into the same ports")
        if labels is not None and _frame_counts(labels) != _frame_counts(masks):
            raise ValueError("labels don't line up with the masks — wire both ports from the same node")
        # no pixel drawing here — main.py ships a label volume and the frontend tints
        return {"video": video, "masks": masks, "ids": ids, "labels": labels}


def _jsonable(v):
    """Tensors -> nested Python lists, recursively; everything already
    JSON-shaped (str, dict, numbers) passes through."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class Export(OutputNode):
    """Download whatever is wired in, as one zip. No format settings — what you
    wire decides the contents, and everything from one batch stays aligned by
    item index in a single download:

      image          -> images/item_000.png              (lossless, source resolution)
      video          -> video/item_000/frame_000.png     (lossless, per frame)
      masks          -> masks/item_000/obj_00.png        (0/255 grayscale, one per instance;
                        per-frame masks: masks/item_000/frame_000_obj_00.png)
      embedding      -> embeddings.npy                   (float32, B x D)
      boxes, labels, ids, classification
                     -> predictions.json                 ({"items": [...]}, one record per
                        batch item, mirroring the wired structure — per-frame
                        values carry a list per frame)

    Exports the raw wired data, not viewer overlays: binary masks and box
    numbers are machine-consumable — the pretty rendered frames are a
    right-click on View Image away. The zip is content-addressed into /media
    like the previews, so re-running an unchanged graph reuses the file.
    # ponytail: a cached result can point at a zip the OS tmp-cleaner has since
    # removed (404 on click) — POST /cache/clear is the existing remedy.
    """

    kind = "export"
    label = "Export"
    inputs = ["image", "video", "masks", "boxes", "labels", "ids", "embedding", "classification"]
    required = []  # any subset — at least one wired, checked in run

    def run(self, image=None, video=None, masks=None, boxes=None, labels=None, ids=None,
            embedding=None, classification=None):
        import hashlib
        import io
        import json
        import zipfile

        import torch
        from torchvision.io import encode_png

        wired = {k: v for k, v in (("image", image), ("video", video), ("masks", masks), ("boxes", boxes),
                                   ("labels", labels), ("ids", ids), ("embedding", embedding),
                                   ("classification", classification)) if v is not None}
        if not wired:
            raise ValueError("wire something to export — any data port plugs in here")
        if image is not None and video is not None:
            raise ValueError("wire image or video, not both")
        sizes = {k: len(v) for k, v in wired.items()}
        if len(set(sizes.values())) > 1:
            raise ValueError(f"batch sizes differ across the wired ports ({sizes}) — wire ports from the same batch")
        if image is not None:
            _check_per_image(masks, "masks")
            _check_per_image(boxes, "boxes")
        if video is not None:
            for what, vals in (("masks", masks), ("boxes", boxes), ("ids", ids)):
                if vals is not None:
                    _check_per_frame(video, vals, what)

        def png(t):  # (C,H,W) uint8 -> PNG bytes
            return encode_png(t.cpu().contiguous()).numpy().tobytes()

        def mask_png(m):  # (H,W) 0/1 (or bool) -> (1,H,W) 0/255 grayscale PNG
            return png(((m > 0).to(torch.uint8) * 255).unsqueeze(0))

        files: list[tuple[str, bytes]] = []
        for i, im in enumerate(image or []):
            files.append((f"images/item_{i:03d}.png", png(im)))
        for i, clip in enumerate(video or []):
            files.extend((f"video/item_{i:03d}/frame_{t:03d}.png", png(f)) for t, f in enumerate(clip))
        for i, m in enumerate(masks or []):
            if _per_frame(masks):  # per-frame: item i is a list of (N,H,W), one per frame
                files.extend((f"masks/item_{i:03d}/frame_{t:03d}_obj_{j:02d}.png", mask_png(fm[j]))
                             for t, fm in enumerate(m) for j in range(len(fm)))
            else:
                files.extend((f"masks/item_{i:03d}/obj_{j:02d}.png", mask_png(m[j])) for j in range(len(m)))
        if embedding is not None:
            import numpy as np

            buf = io.BytesIO()
            np.save(buf, np.stack([e.detach().cpu().float().numpy().ravel() for e in embedding]))
            files.append(("embeddings.npy", buf.getvalue()))
        tabular = {k: wired[k] for k in ("boxes", "labels", "ids", "classification") if k in wired}
        if tabular:
            items = [{k: _jsonable(v[i]) for k, v in tabular.items()} for i in range(len(next(iter(tabular.values()))))]
            files.append(("predictions.json", json.dumps({"items": items}, indent=1).encode()))

        MEDIA.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=MEDIA, delete=False, suffix=".zip") as tmp:
            with zipfile.ZipFile(tmp, "w") as zf:
                for name, data in files:
                    # fixed timestamp: the zip's own bytes name the file below, so
                    # entry mtimes must not vary between builds of the same content
                    zf.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
        tmp_path = Path(tmp.name)
        sha1 = hashlib.sha1()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha1.update(chunk)
        dest = MEDIA / f"{sha1.hexdigest()[:20]}.zip"
        if dest.exists():
            tmp_path.unlink()
        else:
            tmp_path.replace(dest)
        n = dest.stat().st_size
        size = f"{n / 2**20:.1f} MB" if n >= 2**20 else f"{n / 1024:.0f} KB"
        return {"download": f"/media/{dest.name}", "summary": f"{len(files)} files, {size}"}



# --- generic model nodes (detect / depth / caption / custom) ------------------


class Detect(HfNode):
    """Detect objects with a detection model picked in the settings dropdown
    (or any Hugging Face detection checkpoint as a custom id). Plain
    fixed-vocabulary detectors (the DETR default, YOLOS, ...) run with no
    prompt; open-vocabulary ones (the OWLv2 preset, OWL-ViT, ...) are sniffed
    from `model_id`'s config (the same lookup transformers' pipeline() uses to
    route a task) and need a Text Prompt of comma-separated concepts — each
    becomes a candidate label. `threshold` filters detections by score
    (there's no separate scores port). Wire `video` instead for per-frame
    detection of a clip — the same `boxes`/`labels` ports then carry per-frame
    values (per clip, per frame, aligned 1:1), the shape Track Objects expects.
    Outputs per-image `boxes` and the detected class name per box as `labels`
    (one name per instance).

    `slice_size` (SAHI) tiles each image/frame before detecting, offsets each
    tile's boxes back to full-frame coordinates, and merges overlapping tiles'
    detections with class-aware NMS (never suppresses across different labels)."""

    kind = "detect"
    label = "Detect Objects"
    category = "Detect"
    color = "#db2777"  # detection-pipeline pink, shared with Track/Filter by Region
    inputs = ["image", "video", "prompts"]
    required = []  # either-or image/video checked in run; prompts only for zero-shot models
    outputs = ["boxes", "labels"]

    model_id: str = Field(
        "facebook/detr-resnet-50",
        json_schema_extra={"options": [
            {"value": "facebook/detr-resnet-50", "label": "DETR (fixed classes)"},
            {"value": "google/owlv2-base-patch16-ensemble", "label": "OWLv2 (text-prompted)"},
        ], "hf_search": ["object-detection", "zero-shot-object-detection"]},
    )
    threshold: float = Field(0.5, description="Confidence cutoff — detections scoring below it are dropped.")
    slice_size: int = Field(0, description="Tile side in px for sliced inference (SAHI) — the image is cut into overlapping tiles, each inferred separately, detections merged. 0 = off. Use when small objects on large images get missed.")
    slice_overlap: float = Field(0.2, description="Fraction of tile side that adjacent tiles overlap; only used when slice_size > 0.")

    def run(self, image=None, video=None, prompts=None):
        import torch
        from PIL import Image
        from torchvision.ops import batched_nms
        from transformers.models.auto.modeling_auto import MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES as ZS

        batch, port = _image_or_video(image, video)
        zero_shot = _hf_zero_shot(self.model_id, ZS, self.trust_remote_code)
        phrases = _phrases((prompts or {}).get("text") or "")
        if zero_shot and phrases == [""]:
            raise ValueError(
                f"Detect Objects: {self.model_id!r} is a zero-shot model — wire a Text Prompt (comma-separated concepts)"
            )
        task = "zero-shot-object-detection" if zero_shot else "object-detection"

        def to_boxes_labels(dets):
            if not dets:
                return torch.zeros((0, 4), dtype=torch.int32), []
            bx = torch.tensor(
                [[d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]] for d in dets],
                dtype=torch.int32,
            )
            return bx, [d["label"] for d in dets]

        def detect(sub):
            kw = {"candidate_labels": phrases} if zero_shot else {}
            return pipe(sub, threshold=self.threshold, **kw)

        def sliced(im):
            """One image's SAHI tiles -> class-aware-NMS-merged (boxes, labels).
            A frame that already fits one tile takes the untiled path, unmerged."""
            h, w = im.shape[-2:]
            grid = _slice_grid(h, w, self.slice_size, self.slice_overlap)
            if not grid:
                return to_boxes_labels(_chunked([Image.fromarray(_batch_to_hwc([im])[0])], detect)[0])
            crops = [im[:, y1:y2, x1:x2] for x1, y1, x2, y2 in grid]
            pil = [Image.fromarray(c) for c in _batch_to_hwc(crops)]
            boxes, scores, labels = [], [], []
            for (x1, y1, x2, y2), dets in zip(grid, _chunked(pil, detect)):
                for d in dets:
                    b = d["box"]
                    boxes.append([b["xmin"] + x1, b["ymin"] + y1, b["xmax"] + x1, b["ymax"] + y1])
                    scores.append(d["score"])
                    labels.append(d["label"])
            if not boxes:
                return torch.zeros((0, 4), dtype=torch.int32), []
            ids = {l: i for i, l in enumerate(dict.fromkeys(labels))}  # class-aware: never suppress across labels
            keep = batched_nms(
                torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32),
                torch.tensor([ids[l] for l in labels]), 0.5,
            )
            return torch.tensor(boxes, dtype=torch.int32)[keep], [labels[i] for i in keep.tolist()]

        if port == "video":
            if self.slice_size > 0:
                _total(sum(  # tiles (or 1, untiled) per frame, every frame of every clip
                    (len(_slice_grid(clip.shape[-2], clip.shape[-1], self.slice_size, self.slice_overlap)) or 1) * clip.shape[0]
                    for clip in batch
                ))
                pipe = _hf_pipeline(task, self.model_id, self.trust_remote_code)
                out_boxes, out_labels = [], []
                for clip in batch:
                    per_frame = [sliced(im) for im in clip]
                    out_boxes.append([x[0] for x in per_frame])
                    out_labels.append([x[1] for x in per_frame])
                return {"boxes": out_boxes, "labels": out_labels}
            _total(sum(len(clip) for clip in batch))  # before the model load: an empty bar beats no bar
            pipe = _hf_pipeline(task, self.model_id, self.trust_remote_code)
            out_boxes, out_labels = [], []
            for clip in batch:
                pil = [Image.fromarray(f) for f in _batch_to_hwc(clip)]
                per_frame = [to_boxes_labels(d) for d in _chunked(pil, detect)]
                out_boxes.append([x[0] for x in per_frame])
                out_labels.append([x[1] for x in per_frame])
            return {"boxes": out_boxes, "labels": out_labels}

        if self.slice_size > 0:
            _total(sum(len(_slice_grid(*im.shape[-2:], self.slice_size, self.slice_overlap)) or 1 for im in batch))
            pipe = _hf_pipeline(task, self.model_id, self.trust_remote_code)
            per_image = [sliced(im) for im in batch]
            return {"boxes": [x[0] for x in per_image], "labels": [x[1] for x in per_image]}

        _total(len(batch))
        pipe = _hf_pipeline(task, self.model_id, self.trust_remote_code)
        pil = [Image.fromarray(f) for f in _batch_to_hwc(batch)]
        per_image = [to_boxes_labels(d) for d in _chunked(pil, detect)]
        return {"boxes": [x[0] for x in per_image], "labels": [x[1] for x in per_image]}


class Depth(HfNode):
    """Estimate depth with a Hugging Face depth model (default:
    depth-anything/Depth-Anything-V2-Small-hf; `model_id` takes any Hub id).
    Each image's predicted depth is normalized to uint8 and replicated to 3
    channels, so the result is a standard `image` port value any downstream
    image node/viewer can take — darker/lighter reads relative within each
    image, not an absolute scale across the batch."""

    kind = "depth"
    label = "Estimate Depth"
    category = "Transform"
    color = "#2563eb"
    inputs = ["image"]
    outputs = ["image"]

    model_id: str = Field(
        "depth-anything/Depth-Anything-V2-Small-hf",
        json_schema_extra={"options": [
            {"value": "depth-anything/Depth-Anything-V2-Small-hf", "label": "Depth Anything V2 small"},
        ], "hf_search": ["depth-estimation"]},
    )

    def run(self, image):
        import numpy as np
        import torch
        from PIL import Image

        _total(len(image))  # before the model load: an empty bar beats no bar
        pipe = _hf_pipeline("depth-estimation", self.model_id, self.trust_remote_code)
        pil = [Image.fromarray(f) for f in _batch_to_hwc(image)]
        out = []
        for r in _chunked(pil, pipe):
            d = np.array(r["depth"])  # (H,W) uint8, already normalized per image by the pipeline
            out.append(torch.from_numpy(np.stack([d, d, d])).contiguous())
        return {"image": out}


class Caption(HfNode):
    """Caption each image with a Hugging Face captioning/VLM model (default:
    Salesforce/blip-image-captioning-base; `model_id` takes any Hub id), run
    through the image-text-to-text pipeline (the transformers-5 replacement
    for the old image-to-text task — it covers plain captioners like BLIP the
    same way it covers prompted VLMs). A wired Text Prompt is passed as the
    conditioning text (e.g. Florence-2's "<CAPTION>"/"<OCR>" task tokens);
    unwired, an empty prompt asks for a plain caption. Outputs the generated
    text as `labels`, one string per image. Image-only — wire Sample Frames
    first for video."""

    kind = "caption"
    label = "Caption"
    category = "Analyze"
    color = "#9333ea"  # matches Count Objects: computes an answer from the pixels
    inputs = ["image", "prompts"]
    outputs = ["labels"]

    model_id: str = Field(
        "Salesforce/blip-image-captioning-base",
        json_schema_extra={"options": [
            {"value": "Salesforce/blip-image-captioning-base", "label": "BLIP base"},
        ], "hf_search": ["image-text-to-text"]},
    )

    def run(self, image, prompts=None, video=None):
        if video is not None:
            raise ValueError("Caption is image-only — wire a Sample Frames node before it")
        from PIL import Image

        text = (prompts or {}).get("text") or ""
        _total(len(image))  # before the model load: an empty bar beats no bar
        pipe = _hf_pipeline("image-text-to-text", self.model_id, self.trust_remote_code)
        pil = [Image.fromarray(f) for f in _batch_to_hwc(image)]

        def caption(sub):
            return pipe(sub, text=[text] * len(sub))

        return {"labels": [r["generated_text"] for r in _chunked(pil, caption)]}


