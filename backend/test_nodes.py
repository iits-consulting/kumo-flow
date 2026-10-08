"""Unit tests for the KumoFlow nodes and graph evaluation.

Runs the real registry, albumentations and torchvision — no SAM3 weights are
loaded (that path is only reached with a live model). Everything is batched:
image is a list of (C,H,W) RGB uint8 tensors (per-image sizes may differ),
masks/boxes are per-image lists.

    uv run pytest
"""

import base64

import numpy as np
import pytest
import torch
from torchvision.io import encode_png

import graph
import main
from graph import Edge, NodeIn, _incoming, _merge_prompts, _sigs, evaluate
from nodes import REGISTRY


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The cross-run result cache is process-global — isolate every test."""
    graph.CACHE.clear()
    yield
    graph.CACHE.clear()


def _png_data_url(chw_uint8):
    """(C,H,W) uint8 tensor -> a 'data:image/png;base64,...' URL like the browser sends."""
    return "data:image/png;base64," + base64.b64encode(encode_png(chw_uint8).numpy().tobytes()).decode()


def _white_left_px():
    """A 1x2 RGB image with a white pixel on the left, as a (C,H,W) tensor."""
    px = torch.zeros((3, 1, 2), dtype=torch.uint8)
    px[:, 0, 0] = 255
    return px


def test_load_single_upload():
    out = REGISTRY["load"](data=_png_data_url(torch.zeros((3, 4, 5), dtype=torch.uint8))).run()["image"]
    assert len(out) == 1 and out[0].shape == (3, 4, 5)  # single file -> B=1


def test_load_multiple_uploads_as_batch():
    urls = [_png_data_url(torch.full((3, 4, 5), i * 10, dtype=torch.uint8)) for i in range(2)]
    out = REGISTRY["load"](data=urls).run()["image"]  # a folder / multi-select upload
    assert [im.shape for im in out] == [(3, 4, 5), (3, 4, 5)]
    assert out[1][0, 0, 0] == 10


def test_load_mixed_sizes():
    urls = [
        _png_data_url(torch.zeros((3, 4, 5), dtype=torch.uint8)),
        _png_data_url(torch.zeros((3, 6, 7), dtype=torch.uint8)),
    ]
    out = REGISTRY["load"](data=urls).run()["image"]  # batch = list, so sizes may differ
    assert [im.shape for im in out] == [(3, 4, 5), (3, 6, 7)]


def test_load_labels_pad_to_batch():
    urls = [_png_data_url(torch.zeros((3, 4, 5), dtype=torch.uint8))] * 3
    out = REGISTRY["load"](data=urls, labels=["Cat", "Dog"]).run()  # folder pick fills labels
    assert out["labels"] == ["Cat", "Dog", ""]  # missing tail = unlabeled


def test_load_without_data_errors():
    with pytest.raises(ValueError, match="choose"):
        REGISTRY["load"]().run()


def test_load_image_from_path(tmp_path):
    # a backend-host path skips the browser upload; a directory is a dataset
    (tmp_path / "one.png").write_bytes(encode_png(_white_left_px()).numpy().tobytes())
    out = REGISTRY["load"](data=str(tmp_path / "one.png")).run()
    assert len(out["image"]) == 1 and out["image"][0].shape == (3, 1, 2)
    # class-per-subfolder labels, same layout as the browser folder pick
    (tmp_path / "Cat").mkdir()
    (tmp_path / "Cat" / "two.png").write_bytes(encode_png(_white_left_px()).numpy().tobytes())
    out = REGISTRY["load"](data=str(tmp_path)).run()
    assert len(out["image"]) == 2 and set(out["labels"]) == {"", "Cat"}
    # list items can be raw paths too
    out = REGISTRY["load"](data=[str(tmp_path / "one.png")]).run()
    assert len(out["image"]) == 1
    with pytest.raises(ValueError, match="no images"):
        REGISTRY["load"](data=str(tmp_path / "nope")).run()


def _mp4_data_url(tmp_path, n_frames=10, fps=10):
    """A tiny real mp4 (frame i is a flat gray of value i*25) as a data URL."""
    import cv2

    path = str(tmp_path / "clip.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 16))
    for i in range(n_frames):
        w.write(np.full((16, 16, 3), i * 25, dtype=np.uint8))
    w.release()
    return "data:video/mp4;base64," + base64.b64encode((tmp_path / "clip.mp4").read_bytes()).decode()


def test_load_video_clip_is_the_batch_unit(tmp_path):
    # 10 frames @ 10fps decoded at 5fps -> one clip, B=1, (5,3,16,16)
    out = REGISTRY["load_video"](data=_mp4_data_url(tmp_path), fps=5).run()
    assert len(out["video"]) == 1
    clip = out["video"][0]
    assert clip.shape == (5, 3, 16, 16)
    assert abs(int(clip[1].float().mean()) - 50) < 15  # 2nd sample ≈ frame 2 (codec is lossy)


def test_load_video_dataset_with_labels(tmp_path):
    # a folder of clips is a video dataset: B clips + per-clip labels, padded like Load Image's
    url = _mp4_data_url(tmp_path)
    out = REGISTRY["load_video"](data=[url, url, url], labels=["Walk", "Run"]).run()
    assert len(out["video"]) == 3
    assert out["labels"] == ["Walk", "Run", ""]


def test_load_video_fps_zero_and_max_frames(tmp_path):
    url = _mp4_data_url(tmp_path)
    # defaults are fps=0 + max_frames=0: the whole video, every frame
    assert REGISTRY["load_video"](data=url).run()["video"][0].shape[0] == 10
    capped = REGISTRY["load_video"](data=url, fps=0, max_frames=3).run()["video"][0]
    assert capped.shape[0] == 3
    # the cap spreads across the whole clip, it doesn't truncate the head:
    # frame i is flat gray i*25, so the last sample must be near the end (~225)
    assert abs(int(capped[-1].float().mean()) - 225) < 15


def test_load_video_max_side_downscales(tmp_path):
    # a 4K source decodes to ~27MB/frame no consumer needs — max_side shrinks
    # each frame as it's decoded, so peak RAM is one full-res frame
    url = _mp4_data_url(tmp_path)  # 16x16 frames
    assert REGISTRY["load_video"](data=url, max_side=8).run()["video"][0].shape == (10, 3, 8, 8)
    assert REGISTRY["load_video"](data=url).run()["video"][0].shape == (10, 3, 16, 16)  # default touches only bigger frames


def test_load_video_disk_guard(tmp_path, monkeypatch):
    # whole-video decode is the default, so a video that can't fit on the
    # spill filesystem must refuse up front (with advice) instead of filling
    # the disk to death
    import nodes

    url = _mp4_data_url(tmp_path)  # 10 frames of 16x16
    monkeypatch.setattr(nodes, "_spill_free", lambda: round(1.25 * 5 * 16 * 16 * 3))  # room for 5 frames
    with pytest.raises(ValueError, match="free disk space"):
        REGISTRY["load_video"](data=url).run()
    assert REGISTRY["load_video"](data=url, max_frames=4).run()["video"][0].shape[0] == 4  # sampled down it fits


def test_load_video_decode_matches_reference_cv2_decode(tmp_path):
    # spilling to an mmap must not change a single pixel — bit-for-bit against
    # a straightforward cv2 decode of the same file
    import cv2

    url = _mp4_data_url(tmp_path, n_frames=6, fps=6)
    out = REGISTRY["load_video"](data=url).run()["video"][0]
    cap = cv2.VideoCapture(str(tmp_path / "clip.mp4"))
    ref = []
    ok, frame = cap.read()
    while ok:
        ref.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    assert out.shape[0] == len(ref)
    for t, im in enumerate(ref):
        assert torch.equal(out[t], torch.from_numpy(im).permute(2, 0, 1))


def test_spill_writer_round_trip():
    # different shapes per record, including a zero-byte one, round-trip
    # exactly and the temp file is gone once mapped
    import nodes

    w = nodes._SpillWriter()
    a = torch.arange(3 * 4 * 5, dtype=torch.uint8).reshape(3, 4, 5)
    empty = torch.zeros((0, 4, 4), dtype=torch.uint8)
    c = torch.full((2, 2, 2), 7, dtype=torch.uint8)
    for t in (a, empty, c):
        w.add(t)
    path = w._path
    out = w.finish()
    assert not path.exists()  # unlinked after mapping
    assert len(out) == 3
    assert torch.equal(out[0], a)
    assert out[1].shape == (0, 4, 4)
    assert torch.equal(out[2], c)


def test_spill_writer_finish_stacked():
    import nodes

    w = nodes._SpillWriter()
    frames = [torch.full((3, 4, 4), i, dtype=torch.uint8) for i in range(5)]
    for f in frames:
        w.add(f)
    path = w._path
    out = w.finish_stacked((5, 3, 4, 4))
    assert not path.exists()
    assert out.shape == (5, 3, 4, 4)
    for i in range(5):
        assert torch.equal(out[i], frames[i])


def test_spill_writer_all_empty_records():
    # every add() is zero-byte -> no mapping happens, still returns correctly
    # shaped empty tensors and cleans up its temp file
    import nodes

    w = nodes._SpillWriter()
    w.add(torch.zeros((0, 3, 3), dtype=torch.uint8))
    path = w._path
    out = w.finish()
    assert not path.exists()
    assert len(out) == 1 and out[0].shape == (0, 3, 3)


def test_view_video_preview_plays_at_sampled_fps(tmp_path):
    # the webm preview is encoded at load_video's sampling fps -> real-time playback
    import cv2

    from main import MEDIA, Graph, run_graph

    _mp4_data_url(tmp_path)  # writes tmp_path/clip.mp4 (10 frames @ 10fps)
    out = run_graph(Graph(
        nodes=[
            {"id": "a", "kind": "load_video", "config": {"data": str(tmp_path / "clip.mp4"), "fps": 5}},
            {"id": "b", "kind": "view_video", "config": {}},
        ],
        edges=[{"source": "a", "target": "b", "sourceHandle": "video", "targetHandle": "video"}],
        targets=["b"],
    ))["results"]["b"]
    cap = cv2.VideoCapture(str(MEDIA / out["videos"][0].rsplit("/", 1)[-1]))
    assert cap.get(cv2.CAP_PROP_FPS) == 5.0

    # fps=0 (the default) decodes at the native rate — playback reads it back
    # from the file header instead of falling back to the 8fps guess
    out = run_graph(Graph(
        nodes=[
            {"id": "a", "kind": "load_video", "config": {"data": str(tmp_path / "clip.mp4")}},
            {"id": "b", "kind": "view_video", "config": {}},
        ],
        edges=[{"source": "a", "target": "b", "sourceHandle": "video", "targetHandle": "video"}],
        targets=["b"],
    ))["results"]["b"]
    cap = cv2.VideoCapture(str(MEDIA / out["videos"][0].rsplit("/", 1)[-1]))
    assert cap.get(cv2.CAP_PROP_FPS) == 10.0  # the mp4 was written at 10fps


def test_clip_webm_streaming_content_addressed_and_playable(tmp_path, monkeypatch):
    # the streaming rewrite must still be content-addressed (stable URL across
    # calls) and produce a real, decodable webm
    import cv2

    monkeypatch.setattr(main, "MEDIA", tmp_path)
    clip = torch.randint(0, 256, (8, 3, 64, 48), dtype=torch.uint8)
    url1 = main._clip_webm(clip, 5)
    url2 = main._clip_webm(clip, 5)
    assert url1 == url2
    path = tmp_path / url1.rsplit("/", 1)[-1]
    assert path.exists() and path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    assert ok and frame is not None


def test_clip_webm_hash_matches_old_pre_streaming_formula(tmp_path, monkeypatch):
    # a re-run from before the streaming rewrite must still hit the same file
    import hashlib

    from torchvision.transforms.v2.functional import resize

    monkeypatch.setattr(main, "MEDIA", tmp_path)
    clip = torch.randint(0, 256, (8, 3, 64, 48), dtype=torch.uint8)
    fps = 5
    url = main._clip_webm(clip, fps)
    h, w = clip.shape[-2:]
    max_side = 512
    if max(h, w) > max_side:
        clip = resize(clip, [round(h * max_side / max(h, w)), round(w * max_side / max(h, w))])
    old_hash = hashlib.sha1(clip.permute(0, 2, 3, 1).cpu().numpy().tobytes() + str(fps).encode()).hexdigest()[:20]
    assert url == f"/media/{old_hash}.webm"


def test_load_video_errors():
    with pytest.raises(ValueError, match="choose"):
        REGISTRY["load_video"]().run()
    garbage = "data:video/mp4;base64," + base64.b64encode(b"not a video").decode()
    with pytest.raises(ValueError, match="decode"):
        REGISTRY["load_video"](data=garbage).run()


def test_load_video_from_path(tmp_path):
    # a backend-host path skips the browser upload; a directory is a dataset
    # with class-per-subfolder labels, files at the root stay unlabeled
    _mp4_data_url(tmp_path)  # writes tmp_path/clip.mp4
    out = REGISTRY["load_video"](data=str(tmp_path / "clip.mp4"), fps=0).run()
    assert out["video"][0].shape == (10, 3, 16, 16) and out["labels"] == [""]

    (tmp_path / "Walk").mkdir()
    _mp4_data_url(tmp_path / "Walk")
    out = REGISTRY["load_video"](data=str(tmp_path), fps=0).run()
    assert len(out["video"]) == 2
    assert sorted(out["labels"]) == ["", "Walk"]

    with pytest.raises(ValueError, match="no videos"):
        REGISTRY["load_video"](data=str(tmp_path / "nope")).run()


def test_empty_string_settings_mean_defaults(tmp_path):
    # the settings panel sends an emptied numeric field as "" — that must mean
    # "the default" (fps/max_frames 0 = load everything), not a type error
    url = _mp4_data_url(tmp_path)
    out = REGISTRY["load_video"](data=url, fps="", max_frames="", max_side="").run()
    assert out["video"][0].shape == (10, 3, 16, 16)


def _clip_file(tmp_path, name, fourcc, n_frames=10):
    """A tiny real video (frame i is flat gray i*25) in the container `name` implies."""
    import cv2

    w = cv2.VideoWriter(str(tmp_path / name), cv2.VideoWriter_fourcc(*fourcc), 10, (16, 16))
    for i in range(n_frames):
        w.write(np.full((16, 16, 3), i * 25, dtype=np.uint8))
    w.release()
    return str(tmp_path / name)


@pytest.mark.parametrize("name,fourcc", [("clip.webm", "VP80"), ("clip.mov", "mp4v")])
def test_load_video_webm_and_mov(tmp_path, name, fourcc):
    path = _clip_file(tmp_path, name, fourcc)
    assert REGISTRY["load_video"](data=path).run()["video"][0].shape == (10, 3, 16, 16)
    # max_frames still spreads across the clip — for matroska that takes a
    # counting pass, because webm/mkv headers lie about frame counts
    capped = REGISTRY["load_video"](data=path, max_frames=3).run()["video"][0]
    assert capped.shape[0] == 3 and abs(int(capped[-1].float().mean()) - 225) < 15


def test_load_video_pyav_fallback(tmp_path, monkeypatch):
    # cv2's bundled ffmpeg can't open some formats (AV1) — PyAV takes over,
    # with the same fps sampling
    import types

    import cv2

    path = _clip_file(tmp_path, "clip.mp4", "mp4v")
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: types.SimpleNamespace(isOpened=lambda: False))
    out = REGISTRY["load_video"](data=path, fps=5).run()  # 10fps source -> every 2nd frame
    assert out["video"][0].shape == (5, 3, 16, 16)


def test_load_image_pil_fallback():
    # torchvision only decodes JPEG/PNG/GIF/WEBP — anything else (BMP, TIFF,
    # PPM, TGA, ICO, HEIC, ...) must fall back to Pillow
    import io

    from PIL import Image

    def data_url(save):
        buf = io.BytesIO()
        save(buf)
        return "data:;base64," + base64.b64encode(buf.getvalue()).decode()

    img = REGISTRY["load"](data=data_url(lambda b: Image.new("RGB", (5, 4), (200, 10, 10)).save(b, format="BMP"))).run()["image"][0]
    assert img.shape == (3, 4, 5) and img[0, 0, 0] == 200

    # 16-bit TIFF: PIL's convert("RGB") clips to solid white — we scale instead
    arr = np.full((4, 5), 20000, np.uint16)
    arr[0, 0] = 40000
    img = REGISTRY["load"](data=data_url(lambda b: Image.fromarray(arr).save(b, format="TIFF"))).run()["image"][0]
    assert img[0, 0, 0] == 255 and 120 <= img[0, 1, 1] <= 135

    import pillow_heif

    pillow_heif.register_heif_opener()
    img = REGISTRY["load"](data=data_url(lambda b: Image.new("RGB", (5, 4), (0, 128, 0)).save(b, format="HEIF"))).run()["image"][0]
    assert img.shape == (3, 4, 5) and abs(int(img[1, 0, 0]) - 128) <= 2  # heif is lossy

    with pytest.raises(ValueError, match="decode"):
        REGISTRY["load"](data="data:;base64," + base64.b64encode(b"not an image").decode()).run()


def _dcm_slice(path, uid, pos, value, mono1=False):
    """A minimal single-slice DICOM: 4x5 uint16 flat gray, positioned along z."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = uid
    ds.SeriesDescription = "T1 axial"
    ds.Rows, ds.Columns = 4, 5
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, pos]
    ds.PixelData = np.full((4, 5), value, np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def test_load_volume_dicom_series(tmp_path):
    from pydicom.uid import generate_uid

    # extensionless files (common for DICOM), written in shuffled order — the
    # series must come out sorted by slice position, scaled volume-wide
    uid = generate_uid()
    _dcm_slice(tmp_path / "c", uid, 2, 200)
    _dcm_slice(tmp_path / "a", uid, 0, 0)
    _dcm_slice(tmp_path / "b", uid, 1, 100)
    (tmp_path / "notes.txt").write_text("not a dicom")  # junk in the folder is skipped
    out = REGISTRY["load_volume"](data=str(tmp_path)).run()
    clip = out["video"][0]
    assert len(out["video"]) == 1 and clip.shape == (3, 3, 4, 5)
    assert [int(clip[i].float().mean()) for i in range(3)] == [0, 127, 255]
    assert out["labels"] == ["T1 axial"]


def test_load_volume_monochrome1_inverts(tmp_path):
    from pydicom.uid import generate_uid

    _dcm_slice(tmp_path / "m1.dcm", generate_uid(), 0, 0, mono1=True)
    assert int(REGISTRY["load_volume"](data=str(tmp_path / "m1.dcm")).run()["video"][0].float().mean()) == 255


def test_load_volume_nifti(tmp_path):
    import nibabel as nib

    vol = np.zeros((4, 5, 3), np.float32)
    vol[..., 1], vol[..., 2] = 100, 200
    (tmp_path / "TumorA").mkdir()
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(tmp_path / "TumorA" / "a.nii.gz"))
    out = REGISTRY["load_volume"](data=str(tmp_path)).run()
    clip = out["video"][0]
    assert clip.shape == (3, 3, 5, 4)  # axial slices, rot90 upright
    assert [int(clip[i].float().mean()) for i in range(3)] == [0, 127, 255]
    assert out["labels"] == ["TumorA"]  # class-per-subfolder dataset layout

    vol4 = np.stack([vol, vol + 50], axis=-1)  # 4D fMRI -> first timepoint
    nib.save(nib.Nifti1Image(vol4, np.eye(4)), str(tmp_path / "b.nii"))
    assert REGISTRY["load_volume"](data=str(tmp_path / "b.nii")).run()["video"][0].shape == (3, 3, 5, 4)


def test_load_volume_errors(tmp_path):
    with pytest.raises(ValueError, match="choose"):
        REGISTRY["load_volume"]().run()
    (tmp_path / "junk.bin").write_bytes(b"junk")
    with pytest.raises(ValueError, match="no DICOM or NIfTI"):
        REGISTRY["load_volume"](data=str(tmp_path)).run()


def test_view_volume(tmp_path, monkeypatch):
    # end to end: load_volume -> view_volume shapes to a raw grayscale (T,H,W)
    # blob + dims the frontend reslices client-side
    import nibabel as nib

    from main import Graph, run_graph

    monkeypatch.setattr(main, "MEDIA", tmp_path)
    vol = np.zeros((4, 5, 3), np.float32)
    vol[..., 1], vol[..., 2] = 100, 200
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(tmp_path / "a.nii"))
    out = run_graph(Graph(
        nodes=[
            {"id": "a", "kind": "load_volume", "config": {"data": str(tmp_path / "a.nii")}},
            {"id": "b", "kind": "view_volume", "config": {}},
        ],
        edges=[{"source": "a", "target": "b", "sourceHandle": "video", "targetHandle": "video"}],
        targets=["b"],
    ))["results"]["b"]
    (raw,) = out["volumes"]
    assert raw["dims"] == [3, 5, 4]  # (T,H,W): 3 axial slices of the rot90'd plane
    data = (tmp_path / raw["url"].rsplit("/", 1)[-1]).read_bytes()
    assert len(data) == 3 * 5 * 4
    assert [set(data[k * 20 : (k + 1) * 20]) for k in range(3)] == [{0}, {127}, {255}]


def test_view_volume_overlay(tmp_path, monkeypatch):
    # per-frame masks on a volume clip are a (T,H,W) label volume: a seg blob
    # ships next to the gray one, with a name/color per instance index
    import nibabel as nib

    from main import Graph, run_graph
    from nodes import Node

    monkeypatch.setattr(main, "MEDIA", tmp_path)
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 3), np.float32), np.eye(4)), str(tmp_path / "a.nii"))

    class TMasks(Node):
        kind = "t_seg"
        inputs = ["video"]
        outputs = ["masks", "labels"]

        def run(self, video):
            m = torch.zeros((1, 5, 4), dtype=torch.uint8)
            m[0, 1, 2] = 1  # one instance, one voxel per slice at (y=1, x=2)
            return {"masks": [[m] * len(video[0])], "labels": [[["tumor"]] * len(video[0])]}

    try:
        out = run_graph(Graph(
            nodes=[
                {"id": "a", "kind": "load_volume", "config": {"data": str(tmp_path / "a.nii")}},
                {"id": "m", "kind": "t_seg", "config": {}},
                {"id": "b", "kind": "view_volume", "config": {}},
            ],
            edges=[
                {"source": "a", "target": "m", "sourceHandle": "video", "targetHandle": "video"},
                {"source": "a", "target": "b", "sourceHandle": "video", "targetHandle": "video"},
                {"source": "m", "target": "b", "sourceHandle": "masks", "targetHandle": "masks"},
                {"source": "m", "target": "b", "sourceHandle": "labels", "targetHandle": "labels"},
            ],
            targets=["b"],
        ))["results"]["b"]
    finally:
        REGISTRY.pop("t_seg", None)
    (raw,) = out["volumes"]
    assert raw["names"] == ["tumor"] and raw["colors"] == [[255, 0, 0]]  # _PALETTE[0], no ids wired
    seg = (tmp_path / raw["seg"].rsplit("/", 1)[-1]).read_bytes()
    assert len(seg) == 3 * 5 * 4  # same dims as the gray volume
    assert all(seg[t * 20 + 1 * 4 + 2] == 1 for t in range(3))  # instance value at the marked voxels
    assert sum(seg) == 3  # and nowhere else


def test_view_volume_refuses_long_clip(monkeypatch):
    # main.py's raw volume is a full-clip float32 copy (4x) — refuse before
    # that OOMs instead of leaving it to the mean() call to get killed
    import nodes

    monkeypatch.setattr(nodes, "_mem_available", lambda: 100)  # tiny "free" RAM
    clip = torch.zeros((4, 3, 5, 5), dtype=torch.uint8)  # 300 bytes > 100/4
    with pytest.raises(ValueError, match="volume viewer"):
        REGISTRY["view_volume"]().run(video=[clip])


def test_reorient():
    # coronal/sagittal frames == the slices VolumeView shows for that axis:
    # new frame axis = old H/W, rows run along old T flipped (superior on top)
    t = torch.arange(2 * 3 * 3 * 4, dtype=torch.uint8).reshape(2, 3, 3, 4)
    assert REGISTRY["reorient"](plane="axial").run(video=[t])["video"][0] is t
    cor = REGISTRY["reorient"](plane="coronal").run(video=[t])["video"][0]
    assert cor.shape == (3, 3, 2, 4)  # (H, 3, T, W)
    assert torch.equal(cor[1, :, 0, :], t[1, :, 1, :])  # frame y=1, top row = last axial slice
    sag = REGISTRY["reorient"](plane="sagittal").run(video=[t])["video"][0]
    assert sag.shape == (4, 3, 2, 3)  # (W, 3, T, H)
    assert torch.equal(sag[2, :, 0, :], t[1, :, :, 2])  # frame x=2, top row = last slice's column y-run


def test_load_stream_grabs_from_source(tmp_path):
    # a file path stands in for a live URL in tests: same cv2 open + grab loop,
    # EOF ends the capture window early instead of a webcam's timeout
    path = _clip_file(tmp_path, "clip.mp4", "mp4v")
    out = REGISTRY["load_stream"](source=path, duration=10).run()
    assert out["video"][0].shape == (10, 3, 16, 16)  # fps=0 = every frame delivered
    assert REGISTRY["load_stream"].cacheable is False  # live source: never cached
    with pytest.raises(ValueError, match="open"):
        REGISTRY["load_stream"](source=str(tmp_path / "nope.mp4")).run()


def test_sample_frames_passes_boxes_through():
    clip = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
    b0, b1 = (torch.tensor([[0, 0, i, i]], dtype=torch.int32) for i in (1, 2))
    out = REGISTRY["sample_frames"]().run(video=[clip], boxes=[[b0, b1]])
    assert len(out["boxes"]) == 2 and torch.equal(out["boxes"][1], b1)


def test_sample_frames_whole_clip_becomes_the_batch():
    # per_clip=0 on a single clip = the old "video is a frame batch" behavior
    clip = torch.zeros((4, 3, 2, 2), dtype=torch.uint8)
    out = REGISTRY["sample_frames"]().run(video=[clip])
    assert len(out["image"]) == 4 and out["image"][0].shape == (3, 2, 2)
    assert out["labels"] == [""] * 4


def test_sample_frames_even_spacing_clamp_and_labels():
    clips = [
        torch.stack([torch.full((3, 2, 2), i, dtype=torch.uint8) for i in range(9)]),
        torch.zeros((2, 3, 2, 2), dtype=torch.uint8),
    ]
    out = REGISTRY["sample_frames"](per_clip=3).run(video=clips, labels=["a", "b"])
    assert [im[0, 0, 0].item() for im in out["image"][:3]] == [0, 4, 8]  # evenly spaced
    assert len(out["image"]) == 5  # 3 + min(3, 2): per_clip clamps to the clip length
    assert out["labels"] == ["a", "a", "a", "b", "b"]  # clip label repeats per frame


def test_video_to_frames_through_graph(tmp_path):
    nodes = {
        "lv": NodeIn(id="lv", kind="load_video", config={"data": _mp4_data_url(tmp_path), "fps": 0}),
        "sf": NodeIn(id="sf", kind="sample_frames", config={"per_clip": 2}),
        "v": NodeIn(id="v", kind="view"),
    }
    edges = [
        Edge(source="lv", target="sf", source_handle="video", target_handle="video"),
        Edge(source="sf", target="v", source_handle="image", target_handle="image"),
    ]
    out = evaluate("v", nodes, _incoming(edges))["image"]
    assert len(out) == 2 and out[0].shape == (3, 16, 16)


def test_modalities_in_spec():
    # inputs name what they produce; processing nodes default to both,
    # video-only nodes (Sample Frames, future trackers) restrict themselves
    assert REGISTRY["load"].spec()["modalities"] == ["image"]
    assert REGISTRY["load_video"].spec()["modalities"] == ["video"]
    assert REGISTRY["sample_frames"].spec()["modalities"] == ["video"]
    assert REGISTRY["flip"].spec()["modalities"] == ["image", "video"]


def test_flip_horizontal_via_graph():
    # load -> flip -> view, driven through evaluate(); the white pixel moves right
    data = _png_data_url(_white_left_px())
    nodes = [
        NodeIn(id="l", kind="load", config={"data": data}),
        NodeIn(id="f", kind="flip"),
        NodeIn(id="v", kind="view"),
    ]
    edges = [Edge(source="l", target="f"), Edge(source="f", target="v")]
    out = evaluate("v", {n.id: n for n in nodes}, _incoming(edges))["image"]  # [(3,1,2)]
    assert out[0][:, 0, 1].tolist() == [255, 255, 255]


def test_stop_one_target_spares_the_shared_subgraph():
    # l feeds a flip branch (l->f->v1) and a direct view (l->v2). Stopping v1
    # alone must kill only its exclusive nodes: l stays live for v2 and its
    # cached result is reused. run_graph is synchronous, so the mid-run /stop
    # is simulated by building the RUNS entry the way run_graph does.
    from main import RUNS, stop
    from nodes import Cancelled

    data = _png_data_url(_white_left_px())
    nodes = [
        NodeIn(id="l", kind="load", config={"data": data}),
        NodeIn(id="f", kind="flip"),
        NodeIn(id="v1", kind="view"),
        NodeIn(id="v2", kind="view"),
    ]
    edges = [Edge(source="l", target="f"), Edge(source="f", target="v1"), Edge(source="l", target="v2")]
    upstream = {}
    for e in edges:
        upstream.setdefault(e.target, []).append(e.source)
    run = RUNS["r1"] = {"cancel": False, "node": "", "nodes": {"l": [1, 2], "f": [1, 2]}, "targets": ["v1", "v2"],
                        "upstream": upstream, "cancelled": set(), "live": None}
    try:
        stop("r1", target="v1")  # the per-node ■ Stop on v1
        assert run["live"] == {"l", "v2"}  # f/v1 dead; shared l still needed by v2
        assert run["nodes"] == {"l": [1, 2]}  # dead f's progress bar dropped from polls

        by_id, inc, cache = {n.id: n for n in nodes}, _incoming(edges), {}
        with pytest.raises(Cancelled):
            evaluate("v1", by_id, inc, cache, run)  # aborts at f — after l ran
        assert "l" in cache and "f" not in cache  # shared work kept, dead branch skipped
        out = evaluate("v2", by_id, inc, cache, run)["image"]  # live target completes
        assert out[0].shape == (3, 1, 2)
    finally:
        RUNS.pop("r1", None)


def test_nbytes_counts_pinned_storage():
    # a cached view (Crop's slice, SampleFrames' index) pins its parent's whole
    # storage — the budget must see that, not the view's few bytes
    parent = torch.zeros(1000, 1000)
    ram, disk = graph._nbytes(parent[:1, :1])
    assert ram >= parent.nbytes
    assert disk == 0


def test_cache_eviction_sees_storage_of_views(monkeypatch):
    monkeypatch.setattr(graph, "CACHE_BYTES", 2**20)  # 1 MiB budget
    view = torch.zeros(1000, 1000)[:1, :1]  # 4-byte view over 4 MB of storage
    graph._cache_put("v", {"image": [view]})
    assert "v" not in graph.CACHE  # accounted at storage size -> over budget -> evicted


def test_cache_evicts_oldest_first(monkeypatch):
    monkeypatch.setattr(graph, "CACHE_BYTES", 250)
    for sig in "ab":
        graph._cache_put(sig, {"x": "y" * 100})  # ~101 bytes each
    graph._cache_get("a")  # bump: "b" is now the oldest
    graph._cache_put("c", {"x": "y" * 100})  # 303 > 250 -> pop down to budget
    assert list(graph.CACHE) == ["a", "c"]


def test_cache_eviction_skips_live_runs_pinned_sigs(monkeypatch):
    monkeypatch.setattr(graph, "CACHE_BYTES", 250)
    for sig in "ab":
        graph._cache_put(sig, {"x": "y" * 100})  # ~101 bytes each
    graph.RUNS["r-pin"] = {"sigs": {"a"}}  # an in-flight run still needs "a"
    try:
        graph._cache_put("c", {"x": "y" * 100})  # 303 > 250: "a" is oldest but pinned
        assert list(graph.CACHE) == ["a", "c"]  # "b" went instead
        graph.RUNS["r-pin"]["sigs"] = {"a", "c", "d"}
        graph._cache_put("d", {"x": "y" * 100})  # everything pinned: stay over cap
        assert list(graph.CACHE) == ["a", "c", "d"]
    finally:
        graph.RUNS.pop("r-pin", None)


def test_nbytes_spill_backed_tensor_counts_as_disk_not_ram():
    import nodes

    w = nodes._SpillWriter()
    w.add(torch.full((100, 4, 4), 7, dtype=torch.uint8))
    (t,) = w.finish()
    assert t.untyped_storage().filename is not None  # spilled through the mmap

    ram, disk = graph._nbytes(t)
    assert ram == 64
    assert disk == t.untyped_storage().nbytes() > 0

    normal = torch.zeros(100, 4, 4, dtype=torch.uint8)
    ram2, disk2 = graph._nbytes(normal)
    assert ram2 >= normal.nbytes
    assert disk2 == 0


def test_cache_disk_eviction_evicts_oldest_spill_backed(monkeypatch):
    import nodes

    monkeypatch.setattr(graph, "CACHE_DISK_BYTES", 150)  # fits one ~100-byte spill entry, not two

    def spill_tensor():
        w = nodes._SpillWriter()
        w.add(torch.zeros((100, 1, 1), dtype=torch.uint8))  # 100 bytes on disk
        (t,) = w.finish()
        return t

    graph._cache_put("a", {"video": [spill_tensor()]})
    graph._cache_put("b", {"video": [spill_tensor()]})
    assert "a" not in graph.CACHE  # evicted for disk, though well under CACHE_BYTES (RAM)
    assert "b" in graph.CACHE


def test_full_stop_lands_during_sigs_and_cancels_cached_targets(monkeypatch):
    # /stop during the (expensive) _sigs window must find the run already
    # registered, and a target whose whole subgraph is then served from the
    # cross-run cache must still answer cancelled instead of repainting.
    from main import RUNS, Graph, run_graph, stop

    graph = Graph(
        nodes=[
            {"id": "l", "kind": "load", "config": {"data": _png_data_url(_white_left_px())}},
            {"id": "v", "kind": "view"},
        ],
        edges=[{"source": "l", "target": "v"}],
        targets=["v"],
        run_id="r-sig",
    )
    assert "images" in run_graph(graph)["results"]["v"]  # warm the cross-run cache

    real_sigs = main._sigs

    def stop_mid_sigs(by_id, incoming):
        assert "r-sig" in RUNS  # registration precedes the sig hash
        stop("r-sig")  # canvas-wide Stop lands in the _sigs window
        return real_sigs(by_id, incoming)

    monkeypatch.setattr(main, "_sigs", stop_mid_sigs)
    assert run_graph(graph)["results"]["v"] == {"cancelled": True}


def test_stop_one_target_end_to_end(monkeypatch):
    # the real path the hand-built test above simulates: POST /run registers
    # the run, /stop?target= recomputes the live set mid-flight, the stopped
    # target answers cancelled, the survivor gets real results, and the shared
    # upstream load ran exactly once.
    import threading
    import time

    from fastapi.testclient import TestClient

    calls = []
    orig_run = REGISTRY["load"].run
    monkeypatch.setattr(REGISTRY["load"], "run", lambda self, **kw: (calls.append(1), orig_run(self, **kw))[1])

    slow_code = (
        "def run(image):\n"
        "    import time\n"
        "    from nodes import _total, _step\n"
        "    _total(50)\n"
        "    for _ in range(50):\n"
        "        time.sleep(0.1)\n"
        "        _step()\n"
        "    return {'image': image}\n"
    )
    payload = {
        "nodes": [
            {"id": "l", "kind": "load", "config": {"data": _png_data_url(_white_left_px())}},
            {"id": "c", "kind": "custom",
             "config": {"input_ports": ["image"], "output_ports": ["image"], "code": slow_code}},
            {"id": "v1", "kind": "view", "config": {}},
            {"id": "v2", "kind": "view", "config": {}},
        ],
        "edges": [
            {"source": "l", "target": "c"},
            {"source": "c", "target": "v1"},
            {"source": "l", "target": "v2"},
        ],
        "targets": ["v1", "v2"],
        "run_id": "e2e",
    }
    runner, ctl = TestClient(main.app), TestClient(main.app)  # one per thread
    res: dict = {}
    t = threading.Thread(target=lambda: res.update(runner.post("/run", json=payload).json()))
    t.start()
    deadline = time.time() + 10  # wait until the slow node is mid-work
    while time.time() < deadline and ctl.get("/progress/e2e").json()["nodes"].get("c", [0])[0] < 1:
        time.sleep(0.01)
    assert time.time() < deadline, "slow node never reported progress"

    assert ctl.post("/stop/e2e", params={"target": "v1"}).json() == {"stopping": True}
    t.join(timeout=30)
    assert not t.is_alive()
    assert res["results"]["v1"] == {"cancelled": True}
    assert "images" in res["results"]["v2"]
    assert len(calls) == 1  # shared load ran once; v2 reused it from the per-run cache


def test_flip_vertical_noop_on_1px():
    vflip = REGISTRY["flip"](direction="vertical").run(image=[_white_left_px()])["image"]
    assert vflip[0][:, 0, 0].tolist() == [255, 255, 255]  # 1px tall: vertical flip is a no-op


def test_batch_of_mixed_sizes_is_first_class():
    # a B=2 sequence of differently-sized images flows through one run
    px = _white_left_px()  # (3,1,2), white left
    wide = torch.zeros((3, 2, 4), dtype=torch.uint8)
    wide[:, 0, 0] = 255  # white top-left
    out = REGISTRY["flip"]().run(image=[px, wide])["image"]  # horizontal flip both
    assert out[0][0, 0, 1] == 255 and out[1][0, 0, 3] == 255


def test_flip_video_keeps_clip_structure():
    clip = torch.zeros((2, 3, 1, 2), dtype=torch.uint8)
    clip[:, :, 0, 0] = 255  # white left column in both frames
    out = REGISTRY["flip"]().run(video=[clip])["video"]
    assert out[0].shape == (2, 3, 1, 2)  # still one clip, not flattened frames
    assert out[0][0, 0, 0, 1] == 255 and out[0][1, 0, 0, 1] == 255  # every frame flipped


def test_dual_port_nodes_take_exactly_one_of_image_video():
    clip = torch.zeros((1, 3, 1, 2), dtype=torch.uint8)
    with pytest.raises(ValueError, match="exactly one"):
        REGISTRY["flip"]().run()
    with pytest.raises(ValueError, match="exactly one"):
        REGISTRY["flip"]().run(image=[_white_left_px()], video=[clip])
    with pytest.raises(ValueError, match="exactly one"):  # raises before any weights load
        REGISTRY["embed"]().run()


def test_blur_only_inside_mask():
    solid = np.full((8, 8, 3), 100, dtype=np.uint8)
    solid[0, 0] = 200  # a pixel outside the mask that must survive unblurred
    img = torch.from_numpy(solid).permute(2, 0, 1)
    seg = torch.zeros((8, 8), dtype=torch.uint8)
    seg[4:, 4:] = 1  # mask the bottom-right only
    blurred = REGISTRY["blur"](blur=3).run(image=[img], masks=[seg.unsqueeze(0)])["image"]
    assert blurred[0][:, 0, 0].tolist() == [200, 200, 200]


def test_blur_video_per_frame():
    clip = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)
    clip[:, :, 4, 4] = 255  # a bright pixel that blur must spread in every frame
    out = REGISTRY["blur"](blur=3).run(video=[clip])["video"]
    assert out[0].shape == (2, 3, 8, 8)
    assert all(out[0][t, 0, 4, 3] > 0 for t in range(2))  # neighbor picked up energy
    with pytest.raises(ValueError, match="per-image"):
        REGISTRY["blur"]().run(video=[clip], masks=[torch.ones((1, 8, 8), dtype=torch.uint8)])


def test_view_draws_box_and_mask():
    base = [torch.zeros((3, 4, 4), dtype=torch.uint8)]
    mask = torch.zeros((4, 4), dtype=torch.uint8)
    mask[1:3, 1:3] = 1
    overlay = REGISTRY["view"]().run(
        image=base, boxes=[torch.tensor([[0, 0, 3, 3]], dtype=torch.int32)], masks=[mask.unsqueeze(0)]
    )["image"]
    assert overlay[0][:, 1, 1].tolist() != [0, 0, 0]  # mask tint
    assert overlay[0][:, 0, 0].tolist() != [0, 0, 0]  # box edge


def test_draw_overlays_index_map_matches_old_blend_for_disjoint_masks():
    # the index-map rewrite must match the old per-mask blend pixel-for-pixel
    # when masks don't overlap (overlapping is the one deliberate behavior change)
    from nodes import _PALETTE, _draw_overlays

    img = (np.arange(8 * 8 * 3) % 251).reshape(8, 8, 3).astype(np.uint8)
    masks = torch.zeros((3, 8, 8), dtype=torch.uint8)
    masks[0, 0:2, 0:2] = 1
    masks[1, 4:6, 0:2] = 1
    masks[2, 0:2, 4:6] = 1  # all three disjoint

    expected = img.copy()
    for j in range(3):
        m = masks[j].numpy().astype(bool)
        color = np.array(_PALETTE[j % len(_PALETTE)], dtype=np.float32)
        expected[m] = (0.5 * expected[m] + 0.5 * color).astype(np.uint8)

    out = _draw_overlays(img.copy(), masks=masks)
    assert np.array_equal(out, expected)


def test_view_draws_labels():
    base = [torch.zeros((3, 40, 40), dtype=torch.uint8)]
    boxes = [torch.tensor([[2, 20, 30, 38]], dtype=torch.int32)]
    plain = REGISTRY["view"]().run(image=base, boxes=boxes)["image"]
    named = REGISTRY["view"]().run(image=base, boxes=boxes, labels=[["person"]])["image"]
    assert not torch.equal(plain[0], named[0])  # text drawn above the box
    # per-item class labels go in the corner, no boxes needed
    corner = REGISTRY["view"]().run(image=base, labels=["cat"])["image"]
    assert corner[0][:, :15, :15].any()
    # per-instance labels on a mask anchor at the mask, no boxes needed
    m = torch.zeros((1, 40, 40), dtype=torch.uint8)
    m[0, 25:35, 5:15] = 1
    assert REGISTRY["view"]().run(image=base, masks=[m], labels=[["dog"]])["image"][0][:, 10:20, :].any()
    with pytest.raises(ValueError, match="need the boxes or masks"):
        REGISTRY["view"]().run(image=base, labels=[["person"]])
    with pytest.raises(ValueError, match="line up"):
        REGISTRY["view"]().run(image=base, boxes=boxes, labels=[["a", "b"]])


def test_view_video_draws_labels():
    clip = torch.zeros((2, 3, 40, 40), dtype=torch.uint8)
    b = torch.tensor([[2, 20, 30, 38]], dtype=torch.int32)
    plain = REGISTRY["view_video"]().run(video=[clip], boxes=[[b, b]])["video"][0]
    named = REGISTRY["view_video"]().run(video=[clip], boxes=[[b, b]], labels=[[["person"], ["person"]]])["video"][0]
    assert not torch.equal(plain, named)  # text drawn above the box in each frame
    # per-clip class label in the corner, no boxes needed
    corner = REGISTRY["view_video"]().run(video=[clip], labels=["Walk"])["video"][0]
    assert corner[0, :, :15, :15].any()
    with pytest.raises(ValueError, match="need the boxes or masks"):
        REGISTRY["view_video"]().run(video=[clip], labels=[[["person"], ["person"]]])
    with pytest.raises(ValueError, match="line up"):
        REGISTRY["view_video"]().run(video=[clip], boxes=[[b, b]], labels=[[["a", "b"], ["a"]]])


def test_view_colors_keyed_by_index_and_label():
    boxes = [torch.tensor([[0, 0, 3, 3]], dtype=torch.int32)]
    overlay = REGISTRY["view"](colors={"0": "#102030"}).run(image=[torch.zeros((3, 4, 4), dtype=torch.uint8)], boxes=boxes)
    assert overlay["image"][0][:, 0, 0].tolist() == [16, 32, 48]  # box border in the picked color
    # with labels wired, the key is the label — "all dogs red", not "instance 2 red"
    base = [torch.zeros((3, 40, 40), dtype=torch.uint8)]
    b = [torch.tensor([[2, 20, 30, 38]], dtype=torch.int32)]
    named = REGISTRY["view"](colors={"dog": "#102030"}).run(image=base, boxes=b, labels=[["dog"]])
    assert named["image"][0][:, 38, 10].tolist() == [16, 32, 48]  # bottom border, away from the label text
    with pytest.raises(ValueError, match="rrggbb"):
        REGISTRY["view"](colors={"dog": "red"}).run(image=base, boxes=b)


def test_view_legend_lists_objects_first_seen():
    base = [torch.zeros((3, 40, 40), dtype=torch.uint8)]
    boxes = [torch.tensor([[2, 2, 20, 20], [4, 4, 24, 24]], dtype=torch.int32)]
    out = REGISTRY["view"](colors={"dog": "#102030"}).run(image=base, boxes=boxes, labels=[["cat", "dog"]])
    assert [e["key"] for e in out["legend"]] == ["cat", "dog"]
    assert out["legend"][0]["color"] == "#ff0000"  # unpicked: the auto palette color it's drawn in
    assert out["legend"][1]["color"] == "#102030"  # picked color wins
    assert out["legend"][0]["thumb"].startswith("/media/") and out["legend"][0]["thumb"].endswith(".jpg")
    # no labels: keys are instance indices; nothing wired: no legend
    plain = REGISTRY["view"]().run(image=base, boxes=boxes)
    assert [e["key"] for e in plain["legend"]] == ["0", "1"]
    assert REGISTRY["view"]().run(image=base)["legend"] == []


def _grid_boxes(n):
    """n small non-overlapping boxes, for legend/overlay tests that just need distinct instances."""
    return torch.tensor(
        [[(j % 10) * 10, (j // 10) * 10, (j % 10) * 10 + 5, (j // 10) * 10 + 5] for j in range(n)], dtype=torch.int32
    )


def test_build_legend_caps_at_50():
    from nodes import _build_legend

    im = torch.zeros((3, 100, 100), dtype=torch.uint8)
    entries, more = _build_legend([(im, _grid_boxes(60), None, None, None)], {})
    assert [e["key"] for e in entries] == [str(j) for j in range(50)]
    assert more == 10


def test_view_legend_more_only_past_cap():
    base = [torch.zeros((3, 100, 100), dtype=torch.uint8)]
    few = REGISTRY["view"]().run(image=base, boxes=[_grid_boxes(2)])
    assert "legend_more" not in few
    many = REGISTRY["view"]().run(image=base, boxes=[_grid_boxes(60)])
    assert len(many["legend"]) == 50
    assert many["legend_more"] == 10


def test_view_video_colors_follow_track_ids():
    clip = torch.zeros((1, 3, 40, 40), dtype=torch.uint8)
    boxes = [[torch.tensor([[0, 0, 39, 39]], dtype=torch.int32)]]
    ids = [[torch.tensor([5], dtype=torch.int32)]]
    out = REGISTRY["view_video"](colors={"5": "#102030"}).run(video=[clip], boxes=boxes, ids=ids)
    assert out["video"][0][0, :, 39, 20].tolist() == [16, 32, 48]  # bottom border, away from the id text
    assert [e["key"] for e in out["legend"]] == ["5"]  # legend keyed the same way


def test_chunked_halves_on_oom():
    from nodes import _chunked

    calls = []

    def fn(sub):  # pretends anything over 2 items per forward OOMs the GPU
        calls.append(len(sub))
        if len(sub) > 2:
            raise torch.cuda.OutOfMemoryError("fake")
        return [x * 10 for x in sub]

    assert _chunked(list(range(7)), fn, chunk=8) == [i * 10 for i in range(7)]
    assert all(c <= 2 for c in calls[-4:])  # settled on a chunk the "GPU" can take
    with pytest.raises(torch.cuda.OutOfMemoryError):  # chunk=1 still failing re-raises
        _chunked([1], lambda sub: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("fake")))


def test_view_video_passthrough_and_mask_tint():
    clip = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)
    assert torch.equal(REGISTRY["view_video"]().run(video=[clip])["video"][0], clip)

    m = torch.zeros((1, 8, 8), dtype=torch.uint8)
    m[0, 2:5, 2:5] = 1
    tinted = REGISTRY["view_video"]().run(video=[clip], masks=[[m, m]])["video"]
    assert tinted[0].shape == (2, 3, 8, 8)
    assert tinted[0][1, :, 3, 3].tolist() != [0, 0, 0]  # tint reaches every frame
    assert tinted[0][0, :, 0, 0].tolist() == [0, 0, 0]  # outside the mask untouched


def test_view_video_boxes_port():
    clip = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)
    b = torch.tensor([[1, 1, 6, 6]], dtype=torch.int32)
    out = REGISTRY["view_video"]().run(video=[clip], boxes=[[b]])["video"]
    assert out[0][0, :, 1, 1].tolist() != [0, 0, 0]  # box border drawn
    assert out[0][0, :, 3, 3].tolist() == [0, 0, 0]  # interior untouched (no masks wired)
    with pytest.raises(ValueError, match="line up"):
        REGISTRY["view_video"]().run(video=[clip, clip], boxes=[[b]])


def test_view_video_mask_alignment():
    clip = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="line up"):
        REGISTRY["view_video"]().run(video=[clip], masks=[[torch.zeros((1, 8, 8), dtype=torch.uint8)]])
    # per-image masks on a video consumer: structurally detected, not name-gated
    with pytest.raises(ValueError, match="per-image"):
        REGISTRY["view_video"]().run(video=[clip], masks=[torch.zeros((1, 8, 8), dtype=torch.uint8)])


def test_mask_ops_fill_holes():
    ring = torch.ones((5, 5), dtype=torch.uint8)
    ring[2, 2] = 0  # a single interior hole surrounded by foreground
    filled = REGISTRY["mask_ops"](fill_holes=True).run(masks=[ring.unsqueeze(0)])["masks"][0][0]
    assert filled[2, 2] == 1
    # per-frame masks (video) go through the same op one level deeper
    vid = REGISTRY["mask_ops"](fill_holes=True).run(masks=[[ring.unsqueeze(0), ring.unsqueeze(0)]])["masks"]
    assert vid[0][1][0][2, 2] == 1


def test_mask_ops_invert():
    ring = torch.ones((5, 5), dtype=torch.uint8)
    ring[2, 2] = 0
    inv = REGISTRY["mask_ops"](invert=True).run(masks=[ring.unsqueeze(0)])["masks"][0][0]
    assert inv[0, 0] == 0 and inv[2, 2] == 1
    # multiple objects: invert = complement of the UNION -> one background mask,
    # neither object ends up inside the other's background
    a = torch.zeros((5, 5), dtype=torch.uint8)
    a[0, 0] = 1
    b = torch.zeros((5, 5), dtype=torch.uint8)
    b[4, 4] = 1
    inv = REGISTRY["mask_ops"](invert=True).run(masks=[torch.stack([a, b])])["masks"][0]
    assert inv.shape == (1, 5, 5)
    assert inv[0, 0, 0] == 0 and inv[0, 4, 4] == 0 and inv[0, 2, 2] == 1


def test_visual_prompt_outputs_only_prompts():
    base = [torch.zeros((3, 4, 4), dtype=torch.uint8)]
    outs = REGISTRY["visual_prompt"](frames=[{"points": [[1, 1]], "point_labels": [1]}]).run(image=base)
    assert outs["prompts"]["per_image"] == [
        {"points": [[1, 1]], "point_labels": [1], "boxes": [], "box_labels": []}
    ]
    assert "image" not in outs  # prompts only; the image input is just the drawing reference


def test_visual_prompt_pads_per_image_to_batch():
    # drew only on image 0; the batch has 2 -> image 1 gets an empty prompt set
    base = [torch.zeros((3, 4, 4), dtype=torch.uint8), torch.zeros((3, 4, 4), dtype=torch.uint8)]
    per = REGISTRY["visual_prompt"](frames=[{"points": [[2, 2]], "point_labels": [1]}]).run(image=base)["prompts"]["per_image"]
    assert len(per) == 2
    assert per[0]["points"] == [[2, 2]]
    assert per[1] == {"points": [], "point_labels": [], "boxes": [], "box_labels": []}


def test_text_prompt_source():
    assert REGISTRY["text_prompt"](text="cat").run()["prompts"]["text"] == "cat"


def test_empty_prompts_stop_at_the_prompt_node():
    # empty prompt sources raise themselves (the run never reaches SAM3), and
    # evaluate wraps the failure in a NodeError pinning the offending node id
    from main import NodeError

    with pytest.raises(ValueError, match="Text Prompt is empty"):
        REGISTRY["text_prompt"](text="  ").run()
    with pytest.raises(ValueError, match="Visual Prompt is empty"):
        REGISTRY["visual_prompt"]().run(image=[torch.zeros((3, 4, 4), dtype=torch.uint8)])
    with pytest.raises(NodeError, match="Text Prompt is empty") as ei:
        evaluate("t", {"t": NodeIn(id="t", kind="text_prompt")}, _incoming([]))
    assert ei.value.node_id == "t"


def test_multiple_prompt_edges_merge():
    # two Text Prompts + a Visual Prompt into one prompts port: texts comma-join, per_image coexists
    vp = REGISTRY["visual_prompt"](frames=[{"points": [[1, 1]], "point_labels": [1]}])
    merged = _merge_prompts(
        [
            REGISTRY["text_prompt"](text="car").run()["prompts"],
            REGISTRY["text_prompt"](text="not car").run()["prompts"],
            vp.run(image=[torch.zeros((3, 4, 4), dtype=torch.uint8)])["prompts"],
        ]
    )
    assert merged["text"] == "car, not car"
    assert merged["per_image"][0]["points"] == [[1, 1]] and merged["per_image"][0]["point_labels"] == [1]


def test_multiple_edges_on_non_prompt_port_error():
    data = _png_data_url(_white_left_px())
    nodes = {
        "l1": NodeIn(id="l1", kind="load", config={"data": data}),
        "l2": NodeIn(id="l2", kind="load", config={"data": data}),
        "f": NodeIn(id="f", kind="flip"),
    }
    with pytest.raises(ValueError, match="one connection"):
        evaluate("f", nodes, _incoming([Edge(source="l1", target="f"), Edge(source="l2", target="f")]))


def test_crop_from_box():
    big = torch.arange(6 * 6 * 3, dtype=torch.uint8).reshape(6, 6, 3).permute(2, 0, 1)
    cropped = REGISTRY["crop"]().run(image=[big], boxes=[torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)])["image"]
    assert cropped[0].shape == (3, 3, 3)  # rows 2..4, cols 1..3 inclusive
    assert torch.equal(cropped[0], big[:, 2:5, 1:4])


def test_crop_manual_box_clamps():
    big = torch.arange(6 * 6 * 3, dtype=torch.uint8).reshape(6, 6, 3).permute(2, 0, 1)
    manual = REGISTRY["crop"](box="0,0,10,10").run(image=[big])["image"]  # box runs past the edge
    assert manual[0].shape == (3, 6, 6)


def test_crop_differing_boxes_per_image():
    # per-image boxes may produce differently-sized crops — legal now that a batch is a list
    big = torch.zeros((3, 6, 6), dtype=torch.uint8)
    boxes = [
        torch.tensor([[0, 0, 1, 1]], dtype=torch.int32),
        torch.tensor([[0, 0, 3, 2]], dtype=torch.int32),
    ]
    out = REGISTRY["crop"]().run(image=[big, big.clone()], boxes=boxes)["image"]
    assert out[0].shape == (3, 2, 2) and out[1].shape == (3, 3, 4)


def test_crop_emits_one_image_per_box():
    # every detection gets cropped, in image-then-box order: the batch grows to the total count
    big = torch.arange(6 * 6 * 3, dtype=torch.uint8).reshape(6, 6, 3).permute(2, 0, 1)
    boxes = [
        torch.tensor([[0, 0, 1, 1], [2, 2, 5, 4]], dtype=torch.int32),
        torch.zeros((0, 4), dtype=torch.int32),  # nothing found here -> no crops from it
    ]
    out = REGISTRY["crop"](box="0,0,5,5").run(image=[big, big.clone()], boxes=boxes)["image"]
    assert [tuple(x.shape) for x in out] == [(3, 2, 2), (3, 3, 4)]  # wired boxes win over the drawn one
    assert torch.equal(out[1], big[:, 2:5, 2:6])
    with pytest.raises(ValueError, match="no boxes"):
        REGISTRY["crop"]().run(image=[big], boxes=[torch.zeros((0, 4), dtype=torch.int32)])


def test_crop_labels_follow_crops():
    # labels survive the batch growing: one per crop, in the same image-then-box order
    big = torch.arange(6 * 6 * 3, dtype=torch.uint8).reshape(6, 6, 3).permute(2, 0, 1)
    boxes = [
        torch.tensor([[0, 0, 1, 1], [2, 2, 5, 4]], dtype=torch.int32),
        torch.zeros((0, 4), dtype=torch.int32),  # no crop from the dog image -> no dog label
    ]
    images = [big, big.clone()]
    out = REGISTRY["crop"]().run(image=images, boxes=boxes, labels=["cat", "dog"])
    assert out["labels"] == ["cat", "cat"]  # the folder label repeated per crop
    per_instance = REGISTRY["crop"]().run(image=images, boxes=boxes, labels=[["person", "truck"], []])
    assert per_instance["labels"] == ["person", "truck"]  # SAM3's names become per-item labels
    with pytest.raises(ValueError, match="per-instance concept names"):
        REGISTRY["crop"](box="0,0,5,5").run(image=images, labels=[["person"], ["truck"]])
    with pytest.raises(ValueError, match="line up with the boxes"):
        REGISTRY["crop"]().run(image=images, boxes=boxes, labels=[["person"], []])
    with pytest.raises(ValueError, match="wire the same batch into both"):
        REGISTRY["crop"]().run(image=images, boxes=boxes, labels=["cat"])


def test_crop_video_with_manual_box():
    clip = torch.zeros((2, 3, 6, 6), dtype=torch.uint8)
    out = REGISTRY["crop"](box="1,1,3,3").run(video=[clip])["video"]
    assert out[0].shape == (2, 3, 3, 3)  # every frame cropped, clip structure kept
    with pytest.raises(ValueError, match="drawn/typed box"):
        REGISTRY["crop"]().run(video=[clip], boxes=[torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)])


def test_sam3_requires_prompts():
    # prompts isn't a required port (non-SAM3 models take none), so an unwired
    # prompt on the SAM3 default errors from run(), pointing at the settings
    data = _png_data_url(_white_left_px())
    with pytest.raises(ValueError, match="needs a prompt"):
        evaluate(
            "s",
            {"l": NodeIn(id="l", kind="load", config={"data": data}), "s": NodeIn(id="s", kind="segment")},
            _incoming([Edge(source="l", target="s")]),
        )


def test_sam3_prompt_routing_errors():
    # prompt mixes that would silently drop a prompt raise instead, before any weights load
    run = REGISTRY["segment"]().run
    img = [_white_left_px(), _white_left_px()]
    pts = {"points": [[0, 0]], "point_labels": [1]}
    box = {"boxes": [[0, 0, 1, 1]], "box_labels": [1]}
    with pytest.raises(ValueError, match="needs a prompt"):
        run(image=img, prompts={})
    with pytest.raises(ValueError, match="text prompt"):
        run(image=img, prompts={"text": "cat", "per_image": [pts]})  # tracker takes no text
    with pytest.raises(ValueError, match="only a box"):
        run(image=img, prompts={"per_image": [pts, box]})  # image 2's box would be dropped
    with pytest.raises(ValueError, match="no box"):
        run(image=img, prompts={"text": "cat", "per_image": [box]})  # image 2 gets neither


def test_sam3_video_prompt_validation():
    # all raise before any weights load
    clip = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
    run = REGISTRY["segment"]().run
    with pytest.raises(ValueError, match="exactly one"):
        run(prompts={"text": "cat"})
    with pytest.raises(ValueError, match="needs a prompt"):
        run(video=[clip], prompts={})
    with pytest.raises(ValueError, match="text prompt"):  # drawn prompts route to the tracker, which takes no text
        run(video=[clip], prompts={"text": "cat", "per_image": [{"points": [[1, 1]], "point_labels": [1]}]})
    with pytest.raises(ValueError, match="negative box"):  # the tracker seeds objects, negatives are point-only
        run(video=[clip], prompts={"per_image": [{"boxes": [[0, 0, 2, 2]], "box_labels": [0]}]})
    with pytest.raises(ValueError, match="negative box"):  # same check on a later frame of the clip
        run(video=[clip], prompts={"per_image": [{}, {"boxes": [[0, 0, 2, 2]], "box_labels": [0]}]})


# --- fake SAM3 tracker for windowed _track_drawn tests -----------------------
# Long clips now get split into overlapping windows instead of refused (RAM
# guard -> windowing). These fakes stand in for Sam3TrackerVideoModel/Processor
# well enough to exercise the windowing/seam-carry logic without any weights:
# a session is tagged with its creation order (`idx`), and post_process_masks
# stamps a single hot pixel at (idx, local_frame_idx) — so the emitted box for
# a frame reveals exactly which session (window) produced it, and a seam's
# duplicate output is trivially distinguishable from the real one.


class _FTSession:
    def __init__(self, idx):
        self.idx = idx
        self.frame_indices = []
        self.obj_order = []  # obj ids, in first-add order (mirrors obj_id_to_idx)

    def add_new_frame(self, frame, idx):
        self.frame_indices.append(idx)


class _FTVideoProcessor:
    def __init__(self, h, w):
        from types import SimpleNamespace

        self.size = SimpleNamespace(height=h, width=w)

    def __call__(self, videos, device, return_tensors):
        from types import SimpleNamespace

        return SimpleNamespace(pixel_values_videos=[[None for _ in videos]])


class _FTProc:
    def __init__(self, size_h=2, size_w=2):
        self.video_processor = _FTVideoProcessor(size_h, size_w)
        self.sessions = []
        self.add_calls = []  # SimpleNamespace(session, frame_idx, obj_ids, kwargs)

    def init_video_session(self, **kw):
        s = _FTSession(len(self.sessions))
        self.sessions.append(s)
        return s

    def add_inputs_to_inference_session(self, inference_session, frame_idx, obj_ids, **kw):
        from types import SimpleNamespace

        for oid in obj_ids if isinstance(obj_ids, list) else [obj_ids]:
            if oid not in inference_session.obj_order:
                inference_session.obj_order.append(oid)
        self.add_calls.append(SimpleNamespace(session=inference_session, frame_idx=frame_idx, obj_ids=obj_ids, kwargs=kw))

    def post_process_masks(self, pred_masks_list, original_sizes, binarize):
        session, local_idx = pred_masks_list[0]
        h, w = original_sizes[0]
        out = []
        for _ in session.obj_order:  # one object (obj 1) in every test below
            m = torch.zeros((h, w), dtype=torch.uint8)
            m[session.idx, local_idx] = 1
            out.append(m)
        return [out]


class _FTModel:
    def __call__(self, inference_session, frame_idx):
        pass  # the "consume" call — nothing to do, output unused by the caller

    def propagate_in_video_iterator(self, inference_session, start_frame_idx):
        from types import SimpleNamespace

        for local_idx in range(len(inference_session.frame_indices)):
            yield SimpleNamespace(
                pred_masks=(inference_session, local_idx),
                object_ids=list(inference_session.obj_order),
                frame_idx=local_idx,
            )


def _run_windowed(monkeypatch, T, prompt_t, avail):
    """Run _track_drawn on a T-frame clip with a points prompt at global frame
    `prompt_t`, tracker stubbed by the fakes above. Returns (boxes, proc,
    progress) — progress is the run's [done, total] progress slot."""
    import nodes

    proc, model = _FTProc(), _FTModel()
    monkeypatch.setattr(nodes, "_tracker_model", lambda: (model, proc))
    monkeypatch.setattr(nodes, "_mem_available", lambda: avail)
    clip = torch.zeros((T, 3, 40, 40), dtype=torch.uint8)  # H,W: rows for session idx, cols for local frame idx
    per_image = [{} for _ in range(T)]
    per_image[prompt_t] = {"points": [[1, 1]], "point_labels": [1]}
    run = {"cancel": False, "node": "s", "nodes": {}, "live": None}
    token = nodes.PROGRESS.set(run)
    try:
        out = REGISTRY["segment"]().run(video=[clip], prompts={"per_image": per_image})
    finally:
        nodes.PROGRESS.reset(token)
    return out["boxes"][0], out["masks"][0], proc, run["nodes"]["s"]


def test_track_drawn_windows_seam_and_carry(monkeypatch):
    # frame_bytes = 3*2*2*2 = 24; avail sized so avail // (4*frame_bytes) == 32,
    # the window-size floor -> _track_windows(70, 32) == [(0,32),(31,63),(62,70)]
    boxes, masks, proc, prog = _run_windowed(monkeypatch, T=70, prompt_t=0, avail=4 * 24 * 32)

    assert len(proc.sessions) == 3  # one per window
    assert [len(s.frame_indices) for s in proc.sessions] == [32, 32, 8]  # each session got exactly its window's frames

    # every frame's box encodes (local_idx, session_idx) — union of outputs
    # covers 0..T-1 exactly once, and at each seam the EARLIER window's value
    # (not the later window's duplicate local frame 0) is what's kept
    for t in range(70):
        if t <= 31:
            sess, local = 0, t
        elif t <= 62:
            sess, local = 1, t - 31
        else:
            sess, local = 2, t - 62
        assert boxes[t].tolist() == [[local, sess, local, sess]]
    assert boxes[31].tolist() == [[31, 0, 31, 0]]  # seam 0/1: window 0's value, not window 1's local frame 0 ([0,1,0,1])
    assert boxes[62].tolist() == [[31, 1, 31, 1]]  # seam 1/2: window 1's value, not window 2's local frame 0
    assert len(masks) == 70
    assert masks[0].untyped_storage().filename is not None  # spilled through the mmap

    # carried seeds: window 1 and 2's session each got a masked seed at local
    # frame 0, obj 1 (same id as window 0 seeded), equal to the earlier
    # window's seam mask
    def seed_call(session):
        return next(c for c in proc.add_calls if c.session is session and c.frame_idx == 0 and "input_masks" in c.kwargs)

    seed1 = seed_call(proc.sessions[1])
    assert seed1.obj_ids == 1
    expected1 = torch.zeros((40, 40), dtype=torch.uint8)
    expected1[0, 31] = 1  # window 0's last frame (session 0, local 31)
    assert torch.equal(seed1.kwargs["input_masks"], expected1)

    seed2 = seed_call(proc.sessions[2])
    assert seed2.obj_ids == 1
    expected2 = torch.zeros((40, 40), dtype=torch.uint8)
    expected2[1, 31] = 1  # window 1's last frame (session 1, local 31)
    assert torch.equal(seed2.kwargs["input_masks"], expected2)

    assert prog == [72, 72]  # 32+32+8: seam frames (31, 62) counted twice


def test_track_drawn_preprompt_windows_stay_empty(monkeypatch):
    # same windows as above, but the only prompt is at t=65 (inside window 2:
    # [62,70)) -> windows 0 and 1 are entirely before it and never see a session
    boxes, masks, proc, prog = _run_windowed(monkeypatch, T=70, prompt_t=65, avail=4 * 24 * 32)

    assert len(proc.sessions) == 1  # only the window containing the prompt builds one
    assert all(boxes[t].shape[0] == 0 for t in range(63))  # windows 0+1's frames: no instances
    assert boxes[65].tolist() == [[3, 0, 3, 0]]  # the one session built, local frame 65-62=3
    assert len(masks) == 70
    assert prog == [72, 72]  # progress still totals window lengths, prompt position doesn't change it


def test_track_drawn_single_window_matches_today(monkeypatch):
    # avail huge -> W == T for a small clip: exactly today's single-session path
    boxes, masks, proc, prog = _run_windowed(monkeypatch, T=10, prompt_t=0, avail=10**12)

    assert len(proc.sessions) == 1
    assert [len(s.frame_indices) for s in proc.sessions] == [10]
    for t in range(10):
        assert boxes[t].tolist() == [[t, 0, t, 0]]
    assert len(masks) == 10
    assert prog == [10, 10]  # no windowing -> no seam double-count


def test_track_windows_arithmetic():
    # pure window arithmetic, matching the spec's own example (no memory-guard
    # floor here — _track_drawn's real W has a 32-frame minimum, tested above)
    import nodes

    assert nodes._track_windows(10, 4) == [(0, 4), (3, 7), (6, 10)]
    assert nodes._track_windows(5, 32) == [(0, 5)]  # T <= W -> one window, the whole clip


def test_track_drawn_lost_at_seam_emits_empty_windows(monkeypatch):
    # the object's mask is empty on window 0's seam frame -> nothing to carry;
    # with no later prompt the remaining windows have zero objects, so they
    # must emit empty frames without a session instead of propagating an
    # object-less one (which the real tracker can't do)
    import nodes

    class _LostProc(_FTProc):
        def post_process_masks(self, pred_masks_list, original_sizes, binarize):
            session, local_idx = pred_masks_list[0]
            if session.idx == 0 and local_idx == 31:  # hidden exactly at the seam
                h, w = original_sizes[0]
                return [[torch.zeros((h, w), dtype=torch.uint8) for _ in session.obj_order]]
            return super().post_process_masks(pred_masks_list, original_sizes, binarize)

    proc, model = _LostProc(), _FTModel()
    monkeypatch.setattr(nodes, "_tracker_model", lambda: (model, proc))
    monkeypatch.setattr(nodes, "_mem_available", lambda: 4 * 24 * 32)  # W=32 -> windows [(0,32),(31,63),(62,70)]
    clip = torch.zeros((70, 3, 40, 40), dtype=torch.uint8)
    per_image = [{} for _ in range(70)]
    per_image[0] = {"points": [[1, 1]], "point_labels": [1]}
    run = {"cancel": False, "node": "s", "nodes": {}, "live": None}
    token = nodes.PROGRESS.set(run)
    try:
        out = REGISTRY["segment"]().run(video=[clip], prompts={"per_image": per_image})
    finally:
        nodes.PROGRESS.reset(token)
    boxes, masks = out["boxes"][0], out["masks"][0]
    assert len(proc.sessions) == 1  # windows 1+2 never build a session
    assert all(boxes[t].shape[0] == 0 for t in range(31, 70))  # seam frame and everything after: empty
    assert boxes[30].shape[0] == 1  # last visible frame keeps its instance
    assert len(masks) == 70  # frame alignment survives the loss
    assert run["nodes"]["s"] == [72, 72]  # empty windows still tick their full length


def test_sam3_image_ram_guard(monkeypatch):
    # same overrun risk as the video concept path, on images: must refuse before
    # OOM instead of stacking hundreds of instances into RAM. Model stubbed:
    # the guard only needs _seg_concepts' mask output.
    import nodes

    monkeypatch.setattr(nodes, "_concept_model", lambda: (None, None))
    big = torch.ones((1, 2000, 2000), dtype=torch.bool)  # one big instance per image
    monkeypatch.setattr(nodes, "_seg_concepts", lambda model, proc, hwc, phrases, threshold, **kw: [(big,)] * len(hwc))
    monkeypatch.setattr(nodes, "_mem_available", lambda: 10**6)  # 1 MB "free"
    img = [torch.zeros((3, 4, 4), dtype=torch.uint8) for _ in range(3)]
    with pytest.raises(ValueError, match="threshold"):
        REGISTRY["segment"]().run(image=img, prompts={"text": "cat"})


def test_sam3_video_mask_disk_guard(monkeypatch):
    # the video concept path projects mask storage against the spill
    # filesystem now, not RAM
    import nodes

    monkeypatch.setattr(nodes, "_concept_model", lambda: (None, None))
    big = torch.ones((1, 2000, 2000), dtype=torch.bool)  # one big instance per frame
    monkeypatch.setattr(nodes, "_seg_concepts", lambda model, proc, hwc, phrases, threshold, **kw: [(big,)] * len(hwc))
    monkeypatch.setattr(nodes, "_spill_free", lambda: 10**6)  # 1 MB "free"
    clip = torch.zeros((3, 3, 4, 4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="free disk space"):
        REGISTRY["segment"]().run(video=[clip], prompts={"text": "cat"})


def test_segment_video_masks_are_mmap_backed(monkeypatch):
    # masks off the video concept path must come back correct AND spilled —
    # finalized per frame into a writer instead of held as a RAM list
    import nodes

    def stub_seg_concepts(model, proc, hwc, phrases, threshold, **kw):
        out = []
        for im in hwc:
            h, w = im.shape[:2]
            m = torch.zeros((1, h, w), dtype=torch.bool)
            m[0, 0, 0] = True  # one instance per frame, at (0,0)
            out.append((m,))  # one phrase ("cat")
        return out

    monkeypatch.setattr(nodes, "_concept_model", lambda: (None, None))
    monkeypatch.setattr(nodes, "_seg_concepts", stub_seg_concepts)
    clip = torch.zeros((3, 3, 4, 4), dtype=torch.uint8)
    out = REGISTRY["segment"]().run(video=[clip], prompts={"text": "cat"})
    masks = out["masks"][0]
    assert len(masks) == 3
    assert all(m.shape == (1, 4, 4) and bool(m[0, 0, 0]) for m in masks)
    assert masks[0].untyped_storage().filename is not None  # spilled through the mmap


def test_visual_prompt_takes_video():
    # one prompt set per frame of each clip, flattened clip-major; undrawn frames stay empty
    clips = [torch.zeros((2, 3, 4, 4), dtype=torch.uint8)] * 2
    drawn = {"points": [[1, 1]], "point_labels": [1]}
    per = REGISTRY["visual_prompt"](frames=[{}, drawn]).run(video=clips)["prompts"]["per_image"]
    assert len(per) == 4  # 2 clips x 2 frames
    assert per[1]["points"] == [[1, 1]]  # clip 0, frame 1 — prompting isn't tied to frame 0
    empty = {"points": [], "point_labels": [], "boxes": [], "box_labels": []}
    assert per[0] == per[2] == per[3] == empty
    with pytest.raises(ValueError, match="exactly one"):
        REGISTRY["visual_prompt"](frames=[drawn]).run()


def test_sample_frames_carries_video_masks():
    clip = torch.zeros((4, 3, 4, 4), dtype=torch.uint8)
    vm = [[torch.full((1, 4, 4), t, dtype=torch.uint8) for t in range(4)]]  # frame t's mask holds value t
    out = REGISTRY["sample_frames"](per_clip=2).run(video=[clip], masks=vm)
    assert len(out["image"]) == 2 and len(out["masks"]) == 2
    assert out["masks"][1][0, 0, 0].item() == 3  # masks follow the same sampled frame indices (0, 3)
    with pytest.raises(ValueError, match="line up"):
        REGISTRY["sample_frames"]().run(video=[clip], masks=[vm[0][:2]])  # T mismatch


def test_blur_video_only_inside_video_masks():
    clip = torch.full((2, 3, 8, 8), 100, dtype=torch.uint8)
    clip[:, :, 0, 0] = 200  # outside the mask: must survive unblurred in every frame
    m = torch.zeros((8, 8), dtype=torch.uint8)
    m[4:, 4:] = 1  # mask the bottom-right only
    vm = [[m.unsqueeze(0), m.unsqueeze(0)]]
    out = REGISTRY["blur"](blur=3).run(video=[clip], masks=vm)["video"]
    assert out[0][:, :, 0, 0].tolist() == [[200, 200, 200], [200, 200, 200]]
    with pytest.raises(ValueError, match="per-frame"):
        REGISTRY["blur"]().run(image=[_white_left_px()], masks=vm)


def test_warp_overlay_fills_region():
    scene = [torch.zeros((3, 10, 10), dtype=torch.uint8)]
    logo = [torch.full((3, 4, 4), 200, dtype=torch.uint8)]
    region = torch.zeros((10, 10), dtype=torch.uint8)
    region[3:7, 3:7] = 1  # a 4x4 region to warp the logo onto
    warped = REGISTRY["warp_overlay"]().run(image=scene, overlay=logo, masks=[region.unsqueeze(0)])
    assert warped[0][:, 5, 5].tolist() == [200, 200, 200]  # logo fills the region
    assert warped[0][:, 0, 0].tolist() == [0, 0, 0]  # outside untouched


def test_warp_overlay_fills_color_when_no_overlay_wired():
    scene = [torch.zeros((3, 10, 10), dtype=torch.uint8)]
    region = torch.zeros((10, 10), dtype=torch.uint8)
    region[3:7, 3:7] = 1
    out = REGISTRY["warp_overlay"](color="#ff0000").run(image=scene, masks=[region.unsqueeze(0)])
    assert out[0][:, 5, 5].tolist() == [255, 0, 0]  # masked region takes the color
    assert out[0][:, 0, 0].tolist() == [0, 0, 0]  # outside untouched
    # a wired overlay wins over the color config
    logo = [torch.full((3, 4, 4), 200, dtype=torch.uint8)]
    warped = REGISTRY["warp_overlay"](color="#ff0000").run(image=scene, overlay=logo, masks=[region.unsqueeze(0)])
    assert warped[0][:, 5, 5].tolist() == [200, 200, 200]


def test_warp_overlay_rejects_bad_hex_at_config_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        REGISTRY["warp_overlay"](color="white")


def test_batch_serializes_to_one_downscaled_jpeg_per_frame():
    from torchvision.io import decode_jpeg

    from nodes import PREVIEW_SIDE, batch_to_previews

    small = torch.zeros((3, 4, 5), dtype=torch.uint8)
    jpegs = batch_to_previews([small] * 3)
    assert len(jpegs) == 3  # one image per batch item, for the viewer to page through
    assert all(p[:2] == b"\xff\xd8" for p in jpegs)  # JPEG SOI, not PNG: ~200x cheaper to encode
    assert decode_jpeg(torch.frombuffer(jpegs[0], dtype=torch.uint8)).shape[-2:] == (4, 5)  # already small: untouched

    # a source image is capped on its long edge, keeping the aspect ratio — the
    # full-res encode is what made a big batch take minutes
    big = torch.zeros((3, 900, 1800), dtype=torch.uint8)
    out = decode_jpeg(torch.frombuffer(batch_to_previews([big])[0], dtype=torch.uint8))
    assert out.shape[-2:] == (PREVIEW_SIDE // 2, PREVIEW_SIDE)


def test_siglip2_requires_labels():
    # empty/whitespace-only prompt text -> no candidate labels, raises before loading weights
    with pytest.raises(ValueError, match="text prompt"):
        REGISTRY["classify"]().run(image=[_white_left_px()], prompts={"text": " , "})


def test_filter_class_filters_and_sorts():
    imgs = [torch.full((3, 1, 1), v, dtype=torch.uint8) for v in (10, 20, 30)]
    cls = [
        {"labels": ["cat", "dog"], "scores": [0.4, 0.6]},
        {"labels": ["cat", "dog"], "scores": [0.9, 0.1]},
        {"labels": ["cat", "dog"], "scores": [0.1, 0.9]},
    ]
    out = REGISTRY["filter_class"](label_name="cat", threshold=0.3).run(image=imgs, classification=cls)["image"]
    assert [im[0, 0, 0].item() for im in out] == [20, 10]  # 0.1 dropped, rest sorted desc


def test_filter_class_defaults_to_first_label_no_sort():
    imgs = [torch.full((3, 1, 1), v, dtype=torch.uint8) for v in (10, 20)]
    cls = [{"labels": ["cat"], "scores": [0.2]}, {"labels": ["cat"], "scores": [0.8]}]
    out = REGISTRY["filter_class"](sort=False).run(image=imgs, classification=cls)["image"]
    assert [im[0, 0, 0].item() for im in out] == [10, 20]  # threshold 0, order preserved


def test_filter_class_on_clips():
    # per-clip classification (e.g. SigLIP 2 on video) filters the clip batch itself
    clips = [torch.full((1, 3, 1, 1), v, dtype=torch.uint8) for v in (10, 20)]
    cls = [{"labels": ["cat"], "scores": [0.2]}, {"labels": ["cat"], "scores": [0.8]}]
    out = REGISTRY["filter_class"](threshold=0.5).run(video=clips, classification=cls)["video"]
    assert len(out) == 1 and out[0][0, 0, 0, 0].item() == 20


def test_filter_class_errors():
    imgs = [torch.zeros((3, 1, 1), dtype=torch.uint8)]
    cls = [{"labels": ["cat"], "scores": [0.2]}]
    with pytest.raises(ValueError, match="not among"):
        REGISTRY["filter_class"](label_name="dgo").run(image=imgs, classification=cls)
    with pytest.raises(ValueError, match="best was 0.20"):
        REGISTRY["filter_class"](threshold=0.5).run(image=imgs, classification=cls)


def test_view_classification_passes_through():
    cls = [{"labels": ["cat", "dog"], "scores": [0.9, 0.1]}]
    assert REGISTRY["view_classification"]().run(classification=cls) == {"classification": cls}


def _boxes(n):
    """n dummy per-image boxes, the shape SAM3 emits."""
    return torch.zeros((n, 4), dtype=torch.int32)


def test_count_totals_without_labels():
    out = REGISTRY["count"]().run(boxes=[_boxes(3), _boxes(0), _boxes(2)])["counts"]
    assert out == {"concepts": ["objects"], "per_image": [[3], [0], [2]], "total": [5]}


def test_count_breaks_down_by_concept():
    # SAM3 ran one pass per phrase; labels name the concept behind each box
    names = [["person", "truck", "person"], ["truck"]]
    out = REGISTRY["count"]().run(boxes=[_boxes(3), _boxes(1)], labels=names)["counts"]
    assert out["concepts"] == ["person", "truck"]  # columns sorted, one per phrase found
    assert out["per_image"] == [[2, 1], [0, 1]]
    assert out["total"] == [2, 2]


def test_count_all_unnamed_instances_fall_back_to_one_column():
    # drawn prompts name no concept -> SAM3's labels are "" per instance
    out = REGISTRY["count"]().run(boxes=[_boxes(2)], labels=[["", ""]])["counts"]
    assert out == {"concepts": ["objects"], "per_image": [[2]], "total": [2]}


def test_count_per_frame_boxes_without_ids_count_frame_by_frame():
    # HF Detector on video / SAM3's video path, no Track Objects: a row per frame
    names = [[["person"], ["person", "truck"]]]
    out = REGISTRY["count"]().run(boxes=[[_boxes(1), _boxes(2)]], labels=names)["counts"]
    assert out == {"concepts": ["person", "truck"], "per_image": [[1, 0], [1, 1]], "total": [2, 1]}
    out = REGISTRY["count"]().run(boxes=[[_boxes(1), _boxes(2)]])["counts"]
    assert out == {"concepts": ["objects"], "per_image": [[1], [2]], "total": [3]}


def test_count_errors():
    run = REGISTRY["count"]().run
    with pytest.raises(ValueError, match="not per-image class labels"):
        run(boxes=[_boxes(1)], labels=["Cat"])  # Load Image's folder labels
    with pytest.raises(ValueError, match="line up"):
        run(boxes=[_boxes(3)], labels=[["person"]])  # labels from a different SAM3
    with pytest.raises(ValueError, match="line up"):  # per-frame, same mismatch
        run(boxes=[[_boxes(1), _boxes(2)]], labels=[[["person"]]])


def test_count_through_graph():
    counts = evaluate(
        "c",
        {
            "s": NodeIn(id="s", kind="segment"),  # stubbed by the custom node below, no weights
            "k": NodeIn(
                id="k",
                kind="custom",
                config={
                    "output_ports": ["boxes", "labels"],
                    "code": (
                        "import torch\n"
                        "def run(**inputs):\n"
                        "    b = torch.zeros((2, 4), dtype=torch.int32)\n"
                        '    return {"boxes": [b], "labels": [["cat", "cat"]]}\n'
                    ),
                },
            ),
            "c": NodeIn(id="c", kind="count"),
        },
        _incoming([
            Edge(source="k", target="c", source_handle="boxes", target_handle="boxes"),
            Edge(source="k", target="c", source_handle="labels", target_handle="labels"),
        ]),
    )["counts"]
    assert counts == {"concepts": ["cat"], "per_image": [[2]], "total": [2]}


def test_instance_names_survive_the_same_filtering_as_masks():
    from nodes import _instances_to_boxes_masks

    empty = torch.zeros((4, 4), dtype=torch.uint8)
    solid = torch.ones((4, 4), dtype=torch.uint8)
    # the middle instance is empty and gets dropped — its name must go with it
    boxes, masks, names = _instances_to_boxes_masks([solid, empty, solid], 4, 4, ["a", "b", "c"])
    assert len(boxes) == len(masks) == len(names) == 2
    assert names == ["a", "c"]


def test_phrases_split_into_one_concept_per_pass():
    from nodes import _phrases

    assert _phrases("person, truck") == ["person", "truck"]  # SAM3 takes one noun phrase per pass
    assert _phrases("") == [""]  # box exemplars only: a single unnamed pass


def test_per_item_label_ports_reject_instance_names():
    # SAM3's labels are per instance; nodes wanting one label per item say so
    clip = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="per-instance"):
        REGISTRY["sample_frames"]().run(video=[clip], labels=[["cat"]])
    with pytest.raises(ValueError, match="per-instance"):
        REGISTRY["logreg"]().run(embedding=[torch.zeros(2)], labels=[["cat"]])


def test_logreg_fits_and_predicts():
    # two well-separated clusters; the unlabeled 5th embedding sits in the "b" cluster
    emb = [
        torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]),
        torch.tensor([5.0, 5.0]), torch.tensor([5.1, 5.0]),
        torch.tensor([5.0, 4.9]),
    ]
    out = REGISTRY["logreg"](labels="a,a,b,b").run(embedding=emb)["classification"]
    assert len(out) == 5 and out[0]["labels"] == ["a", "b"]  # one prediction per image
    assert out[4]["scores"][1] > 0.5  # unlabeled image lands in "b"


def test_logreg_wired_labels_win_over_config():
    emb = [
        torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]),
        torch.tensor([5.0, 5.0]), torch.tensor([5.1, 5.0]),
    ]
    # the labels port (e.g. Load Image's folder labels) beats the typed config
    out = REGISTRY["logreg"](labels="x,x,y,y").run(embedding=emb, labels=["a", "a", "b", "b"])
    assert out["classification"][0]["labels"] == ["a", "b"]
    # all-empty wired labels fall back to the config string
    out = REGISTRY["logreg"](labels="x,x,y,y").run(embedding=emb, labels=["", "", "", ""])
    assert out["classification"][0]["labels"] == ["x", "y"]


def test_logreg_parses_label_files():
    emb = [torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]), torch.tensor([5.0, 5.0])]
    # a plain .txt: one label per line, trailing blank = predict-only
    txt = REGISTRY["logreg"](labels="a\na\nb").run(embedding=emb)["classification"]
    assert txt[0]["labels"] == ["a", "b"]
    # a .csv with "filename,label" rows: the last field per line wins
    csv = REGISTRY["logreg"](labels="img1.jpg,a\nimg2.jpg,a\nimg3.jpg,b").run(embedding=emb)["classification"]
    assert csv[0]["labels"] == ["a", "b"]


def test_logreg_errors():
    emb = [torch.zeros(2), torch.ones(2)]
    with pytest.raises(ValueError, match="no training labels"):
        REGISTRY["logreg"]().run(embedding=emb)
    with pytest.raises(ValueError, match="two classes"):
        REGISTRY["logreg"](labels="a,a").run(embedding=emb)
    with pytest.raises(ValueError, match="3 labels for 2 images"):
        REGISTRY["logreg"](labels="a,b,c").run(embedding=emb)


def _defect_embeddings():
    """Four tight "good" embeddings and one far-off part, in batch order."""
    return [
        torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]),
        torch.tensor([0.0, 0.1]), torch.tensor([0.1, 0.1]),
        torch.tensor([9.0, 9.0]),
    ]


def test_anomaly_score_flags_the_odd_one_out():
    labels = ["good"] * 4 + ["test"]
    out = REGISTRY["anomaly_score"](normal_label="good").run(embedding=_defect_embeddings(), labels=labels)
    cls = out["classification"]
    assert len(cls) == 5 and cls[0]["labels"] == ["defect", "normal"]  # SigLIP 2's shape
    assert all(abs(sum(c["scores"]) - 1) < 1e-6 for c in cls)
    assert all(c["scores"][0] <= 0.5 for c in cls[:4])  # good parts sit at/below the threshold
    assert cls[4]["scores"][0] > 0.9  # the odd part is far past it


def test_anomaly_score_tolerance_moves_the_threshold():
    labels = ["good"] * 4 + ["test"]
    emb = _defect_embeddings()
    emb[4] = torch.tensor([0.4, 0.4])  # a mild deviation, ~3x the good spread
    flagged = REGISTRY["anomaly_score"](normal_label="good").run(embedding=emb, labels=labels)
    tolerant = REGISTRY["anomaly_score"](normal_label="good", tolerance=5.0).run(embedding=emb, labels=labels)
    assert flagged["classification"][4]["scores"][0] > 0.5  # defect by default
    assert tolerant["classification"][4]["scores"][0] < 0.5  # within a raised tolerance


def test_anomaly_score_without_labels_only_ranks():
    # no normal_label: the batch is its own reference, so the worst item lands
    # exactly on the threshold — a ranking, not a verdict
    cls = REGISTRY["anomaly_score"]().run(embedding=_defect_embeddings())["classification"]
    scores = [c["scores"][0] for c in cls]
    assert scores[4] == max(scores) and abs(scores[4] - 0.5) < 1e-6
    assert scores[0] < scores[4]  # the tight cluster ranks below it


def test_anomaly_score_errors():
    emb = _defect_embeddings()
    with pytest.raises(ValueError, match="no images labeled 'good'"):
        REGISTRY["anomaly_score"](normal_label="good").run(embedding=emb, labels=["a"] * 5)
    with pytest.raises(ValueError, match="at least 2 known-good"):
        REGISTRY["anomaly_score"](normal_label="good").run(embedding=emb, labels=["good"] + ["a"] * 4)
    with pytest.raises(ValueError, match="per-instance"):  # SAM3's labels, not per-item ones
        REGISTRY["anomaly_score"]().run(embedding=emb, labels=[["cat"]] * 5)


def test_patchcore_scores_and_localizes(monkeypatch):
    # the real glue on a random-weight resnet18 instead of the pretrained
    # wide_resnet50_2 — no download, same code path
    import nodes
    from anomalib.models.image.patchcore.lightning_model import Patchcore
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    tf = Patchcore.configure_pre_processor().transform
    torch.manual_seed(0)  # random weights: seed so the run doesn't depend on test order
    model = PatchcoreModel(layers=["layer3"], backbone="resnet18", pre_trained=False).to(nodes._device())
    monkeypatch.setattr(nodes, "_patchcore", lambda: (model, tf))
    # good parts vary a little (real ones do — and identical references leave the
    # threshold on a knife edge), the odd one has a bright patch none of them has
    good = [torch.full((3, 40, 60), v, dtype=torch.uint8) for v in (10, 12, 14, 16)]
    odd = torch.full((3, 40, 60), 13, dtype=torch.uint8)
    odd[:, 10:30, 20:40] = 255
    out = REGISTRY["patchcore"](normal_label="good").run(image=[*good, odd], labels=["good"] * 4 + ["test"])
    cls = out["classification"]
    assert len(cls) == 5 and cls[0]["labels"] == ["defect", "normal"]
    # good parts score at or under the threshold — the held-out one *is* it, at 0.5
    assert cls[4]["scores"][0] > 0.5 and all(c["scores"][0] <= 0.5 for c in cls[:4])
    assert out["masks"][4].shape[1:] == (40, 60)  # region marked at the image's own size
    assert len(out["boxes"][4]) == 1 and not len(out["boxes"][1])  # a part that scores clean gets none


def test_patchcore_caps_the_memory_bank(monkeypatch):
    """Every good part costs ~6 MB of VRAM in the bank, so a folder of 1000 of
    them OOMs a 12 GB card. Only `reference_images` go in, spread over the whole
    set — the first N would be one capture session on a folder ordered by time."""
    import nodes
    from anomalib.models.image.patchcore.lightning_model import Patchcore
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    tf = Patchcore.configure_pre_processor().transform
    torch.manual_seed(0)
    model = PatchcoreModel(layers=["layer3"], backbone="resnet18", pre_trained=False).to(nodes._device())
    # each image is a distinct constant, so what the transform sees names the index
    seen = []

    def spy(x):
        seen.append(round(float(x.flatten()[0]) * 255) - 10)
        return tf(x)

    monkeypatch.setattr(nodes, "_patchcore", lambda: (model, spy))
    imgs = [torch.full((3, 40, 60), 10 + i, dtype=torch.uint8) for i in range(20)]
    run = {"cancel": False, "node": "n", "nodes": {}}
    token = nodes.PROGRESS.set(run)
    try:
        REGISTRY["patchcore"](normal_label="good", reference_images=3).run(image=imgs, labels=["good"] * 20)
    finally:
        nodes.PROGRESS.reset(token)
    bank = seen[:3]  # the bank pass runs first, then all 20 are scored
    assert run["nodes"]["n"][1] == 3 + 20  # the declared work is the capped bank, not all 16 candidates
    assert len(set(bank)) == 3 and max(bank) - min(bank) >= 10  # distinct and spread, not the first 3
    assert not {0, 5, 10, 15} & set(bank)  # still never a held-out part, which would sink the threshold


def test_filter_class_carries_masks_and_boxes_along():
    imgs = [torch.full((3, 1, 1), v, dtype=torch.uint8) for v in (10, 20, 30)]
    masks = [torch.full((1, 1, 1), v, dtype=torch.uint8) for v in (1, 2, 3)]
    boxes = [torch.full((1, 4), v, dtype=torch.int32) for v in (1, 2, 3)]
    cls = [{"labels": ["defect"], "scores": [s]} for s in (0.4, 0.9, 0.1)]
    out = REGISTRY["filter_class"](threshold=0.3).run(image=imgs, classification=cls, masks=masks, boxes=boxes)
    assert [im[0, 0, 0].item() for im in out["image"]] == [20, 10]  # 0.1 dropped, rest sorted
    # each item's annotations moved with it, so a viewer overlays the right ones
    assert [m[0, 0, 0].item() for m in out["masks"]] == [2, 1]
    assert [b[0, 0].item() for b in out["boxes"]] == [2, 1]


def test_find_defects_recipe_through_graph():
    # the "Find Defects" recipe's wiring, with the model stubbed: three images,
    # the middle one flagged with a region. What this checks is the plumbing —
    # sorting the batch by score while its masks/boxes follow it into the viewer.
    stub = (
        "import torch\n"
        "def run(**inputs):\n"
        "    scores = [0.1, 0.9, 0.4]\n"
        "    return {\n"
        '        "classification": [{"labels": ["defect", "normal"], "scores": [s, 1 - s]} for s in scores],\n'
        '        "masks": [torch.full((1, 1, 2), i, dtype=torch.uint8) for i in range(3)],\n'
        '        "boxes": [torch.full((1, 4), i, dtype=torch.int32) for i in range(3)],\n'
        "    }\n"
    )
    urls = [_png_data_url(torch.full((3, 1, 2), v, dtype=torch.uint8)) for v in (10, 20, 30)]
    out = evaluate(
        "v",
        {
            "l": NodeIn(id="l", kind="load", config={"data": urls}),
            "p": NodeIn(
                id="p",
                kind="custom",
                config={"output_ports": ["classification", "masks", "boxes"], "code": stub},
            ),
            "f": NodeIn(id="f", kind="filter_class", config={"threshold": 0.3}),
            "v": NodeIn(id="v", kind="view"),
        },
        _incoming([
            Edge(source="l", target="f", source_handle="image", target_handle="image"),
            Edge(source="p", target="f", source_handle="classification", target_handle="classification"),
            Edge(source="p", target="f", source_handle="masks", target_handle="masks"),
            Edge(source="p", target="f", source_handle="boxes", target_handle="boxes"),
            Edge(source="f", target="v", source_handle="image", target_handle="image"),
            Edge(source="f", target="v", source_handle="masks", target_handle="masks"),
            Edge(source="f", target="v", source_handle="boxes", target_handle="boxes"),
        ]),
    )
    # image "10" (score 0.1) dropped, so the viewer shows "20" (0.9) then "30"
    assert len(out["image"]) == 2
    # ...and "20" is tinted red by mask 1, its own: 0.5*20 + 0.5*255. Had the masks
    # stayed in batch order it would have got mask 0, which is empty — no tint.
    assert out["image"][0][0, 0, 0] == 137


def test_filter_class_needs_one_classification_per_item():
    imgs = [torch.zeros((3, 1, 1), dtype=torch.uint8)] * 2
    with pytest.raises(ValueError, match="1 classifications for 2 images"):
        REGISTRY["filter_class"]().run(image=imgs, classification=[{"labels": ["a"], "scores": [1.0]}])


def test_umap_needs_enough_images():
    with pytest.raises(ValueError, match="at least 4"):
        REGISTRY["umap"]().run(embedding=[torch.zeros(8)] * 3)


def test_show_embeddings_thumbnails():
    emb = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    imgs = [torch.zeros((3, 8, 8), dtype=torch.uint8)] * 2
    out = REGISTRY["show_embeddings"]().run(embedding=emb, image=imgs)
    assert len(out["thumbnails"]) == 2  # one hover thumbnail per point
    # inline, unlike the paged batch previews: too small to be worth a fetch on hover
    assert out["thumbnails"][0].startswith("data:image/jpeg;base64,")
    with pytest.raises(ValueError, match="1 images for 2 points"):
        REGISTRY["show_embeddings"]().run(embedding=emb, image=imgs[:1])


def test_show_embeddings_video_thumbnails():
    emb = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    clips = [torch.zeros((3, 3, 8, 8), dtype=torch.uint8)] * 2
    out = REGISTRY["show_embeddings"]().run(embedding=emb, video=clips)
    assert len(out["thumbnails"]) == 2  # first frame per clip
    with pytest.raises(ValueError, match="not both"):
        REGISTRY["show_embeddings"]().run(
            embedding=emb, image=[torch.zeros((3, 8, 8), dtype=torch.uint8)] * 2, video=clips
        )


def test_show_embeddings_reduces_high_dim():
    emb = [torch.arange(8).float() + i for i in range(8)]  # 8 distinct 8-d embeddings
    pts = REGISTRY["show_embeddings"]().run(embedding=emb)["points"]
    assert len(pts) == 8
    assert all(len(p) == 2 for p in pts)


def test_show_embeddings_2d_passthrough():
    emb = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    pts = REGISTRY["show_embeddings"]().run(embedding=emb)["points"]
    assert pts == [[1.0, 2.0], [3.0, 4.0]]  # untouched, no UMAP involved


def test_show_embeddings_needs_enough_to_reduce():
    with pytest.raises(ValueError, match="at least 4"):
        REGISTRY["show_embeddings"]().run(embedding=[torch.zeros(8)] * 3)


def test_registry_has_all_nodes():
    assert {
        "load", "load_video", "sample_frames", "flip", "blur", "crop", "view",
        "segment", "visual_prompt", "text_prompt", "mask_ops", "warp_overlay",
        "classify", "view_classification", "filter_class", "filter_region", "count", "track",
        "embed", "umap", "logreg", "anomaly_score", "patchcore", "show_embeddings", "custom",
        "detect", "depth", "caption",
    } <= REGISTRY.keys()


def test_custom_node_runs_declared_ports():
    node = REGISTRY["custom"](
        input_ports=["x", "y"],
        output_ports=["sum"],
        code="def run(**inputs):\n    return {'sum': inputs['x'] + inputs['y']}\n",
    )
    assert node.run(x=2, y=3) == {"sum": 5}


def test_custom_node_requires_declared_inputs_connected():
    node = REGISTRY["custom"](input_ports=["x"], code="def run(**inputs):\n    return {}\n")
    with pytest.raises(ValueError, match="'x' not connected"):
        node.run()


def test_custom_node_requires_declared_outputs_returned():
    node = REGISTRY["custom"](output_ports=["y"], code="def run(**inputs):\n    return {}\n")
    with pytest.raises(ValueError, match="containing \\['y'\\]"):
        node.run()


def test_custom_node_requires_a_run_function():
    node = REGISTRY["custom"](code="x = 1\n")
    with pytest.raises(ValueError, match="must define a top-level"):
        node.run()


def test_custom_node_through_graph_with_dynamic_ports():
    # kind "custom" has no fixed cls.inputs — kwargs must come from the wired
    # edges themselves, not a class-level port list, for this to reach run().
    nodes = {
        "l": NodeIn(id="l", kind="load", config={"data": _png_data_url(_white_left_px())}),
        "c": NodeIn(
            id="c",
            kind="custom",
            config={"input_ports": ["image"], "output_ports": ["image"], "code": "def run(**inputs):\n    return inputs\n"},
        ),
    }
    out = evaluate("c", nodes, _incoming([Edge(source="l", target="c", source_handle="image", target_handle="image")]))
    assert out["image"][0].shape == (3, 1, 2)


# --- model nodes' generic Hugging Face paths ----
# hf-internal-testing/tiny-random-* are real, tiny (few-MB, random-weight) repos
# on the Hub — verified to exist via huggingface_hub.model_info before use here.


def _two_sizes():
    """Two RGB images with different H,W — the batch-as-list contract in one call."""
    return [torch.zeros((3, 30, 40), dtype=torch.uint8), torch.zeros((3, 20, 50), dtype=torch.uint8)]


def test_hf_detect_batch_of_differing_sizes():
    node = REGISTRY["detect"](model_id="hf-internal-testing/tiny-random-DetrForObjectDetection", threshold=0.0)
    out = node.run(image=_two_sizes())
    assert len(out["boxes"]) == 2 and len(out["labels"]) == 2  # one entry per image, never stacked
    for b, l in zip(out["boxes"], out["labels"]):
        assert b.dtype == torch.int32 and b.shape[-1] == 4
        assert isinstance(l, list) and len(l) == b.shape[0] and all(isinstance(x, str) for x in l)  # per-instance names


def test_hf_classify_batch_of_differing_sizes():
    node = REGISTRY["classify"](model_id="hf-internal-testing/tiny-random-ViTForImageClassification")
    out = node.run(image=_two_sizes())["classification"]
    assert len(out) == 2
    for c in out:
        assert set(c) == {"labels", "scores"}
        assert len(c["labels"]) == len(c["scores"]) > 0
        assert all(isinstance(s, float) for s in c["scores"])


def test_hf_segment_batch_of_differing_sizes():
    imgs = _two_sizes()
    node = REGISTRY["segment"](model_id="hf-internal-testing/tiny-random-Mask2FormerForUniversalSegmentation")
    out = node.run(image=imgs)
    assert len(out["boxes"]) == len(out["masks"]) == len(out["labels"]) == 2
    for b, m, im, l in zip(out["boxes"], out["masks"], imgs, out["labels"]):
        assert m.dtype == torch.uint8
        assert m.shape[-2:] == im.shape[-2:]  # per-image masks aligned to their own H,W
        assert b.dtype == torch.int32 and b.shape == (m.shape[0], 4)  # one AABB per mask
        assert isinstance(l, list) and len(l) == m.shape[0]  # one name per instance


def test_segment_pipeline_path_rejects_prompts():
    # non-SAM3 checkpoints aren't promptable — a wired prompt raises instead of
    # being silently ignored
    node = REGISTRY["segment"](model_id="hf-internal-testing/tiny-random-Mask2FormerForUniversalSegmentation")
    with pytest.raises(ValueError, match="isn't promptable"):
        node.run(image=_two_sizes(), prompts={"text": "cat"})


def test_hf_embed_batch_of_differing_sizes():
    node = REGISTRY["embed"](model_id="hf-internal-testing/tiny-random-Dinov2Model")
    out = node.run(image=_two_sizes())["embedding"]
    assert len(out) == 2
    d = out[0].shape
    assert all(e.shape == d and e.dtype == torch.float32 for e in out)  # one D-vector per image, same D


def test_hf_depth_batch_of_differing_sizes():
    imgs = _two_sizes()
    node = REGISTRY["depth"](model_id="hf-internal-testing/tiny-random-DPTForDepthEstimation")
    out = node.run(image=imgs)["image"]
    assert len(out) == 2
    for d, im in zip(out, imgs):
        assert d.dtype == torch.uint8 and d.shape[0] == 3  # normalized depth replicated to RGB
        assert d.shape[-2:] == im.shape[-2:]  # resized back to its own source image, not a shared size


def test_hf_caption_batch_of_differing_sizes():
    node = REGISTRY["caption"](model_id="hf-internal-testing/tiny-random-BlipForConditionalGeneration")
    out = node.run(image=_two_sizes())["labels"]
    assert len(out) == 2 and all(isinstance(c, str) for c in out)  # one caption string per image


def test_hf_detect_video_boxes_feed_track():
    # video path loops frames per clip and emits SAM3-shaped per-frame boxes/labels
    # (the tracking use case) — the same _check_per_frame gate and Track Objects
    # consumer both have to accept it
    from nodes import _check_per_frame

    clip = torch.zeros((3, 3, 24, 24), dtype=torch.uint8)  # one clip, 3 frames
    node = REGISTRY["detect"](model_id="hf-internal-testing/tiny-random-DetrForObjectDetection", threshold=0.0)
    out = node.run(video=[clip])
    _check_per_frame([clip], out["boxes"], "boxes")  # raises if the shape/alignment is wrong
    _check_per_frame([clip], out["labels"], "labels")

    ids = REGISTRY["track"]().run(boxes=out["boxes"])["ids"]
    assert len(ids) == 1 and len(ids[0]) == 3  # one clip, one id-list per frame


def test_hf_segment_video_emits_per_frame_masks():
    from nodes import _check_per_frame

    clip = torch.zeros((2, 3, 24, 24), dtype=torch.uint8)
    node = REGISTRY["segment"](model_id="hf-internal-testing/tiny-random-Mask2FormerForUniversalSegmentation")
    out = node.run(video=[clip])
    _check_per_frame([clip], out["boxes"], "boxes")
    _check_per_frame([clip], out["masks"], "masks")
    _check_per_frame([clip], out["labels"], "labels")


def test_classify_pipeline_path_on_video_scores_per_clip():
    # fixed-vocab results are top-k per frame with possibly differing label sets;
    # a clip's score set is the per-name mean over its frames — one set per clip
    clips = [torch.zeros((2, 3, 24, 24), dtype=torch.uint8), torch.zeros((3, 3, 24, 24), dtype=torch.uint8)]
    node = REGISTRY["classify"](model_id="hf-internal-testing/tiny-random-ViTForImageClassification")
    out = node.run(video=clips)["classification"]
    assert len(out) == 2  # one score set per clip, not per frame
    for c in out:
        assert len(c["labels"]) == len(c["scores"]) > 0
        assert c["scores"] == sorted(c["scores"], reverse=True)  # best first, like the image path


def test_embed_video_one_vector_per_clip():
    clips = [torch.zeros((2, 3, 24, 24), dtype=torch.uint8), torch.zeros((3, 3, 24, 24), dtype=torch.uint8)]
    out = REGISTRY["embed"](model_id="hf-internal-testing/tiny-random-Dinov2Model").run(video=clips)["embedding"]
    assert len(out) == 2  # mean of frame embeddings: one vector per clip
    assert out[0].ndim == 1 and out[0].shape == out[1].shape and out[0].dtype == torch.float32


def test_kind_aliases_resolve_old_graphs():
    # old kinds arrive from saved graphs / deploys / chat transcripts; the graph
    # parser rewrites them and injects model_id only where the config lacks it —
    # crucial where the merged default shifted (hf_classify was ViT, not SigLIP)
    n = NodeIn(id="a", kind="hf_classify")
    assert n.kind == "classify" and n.config["model_id"] == "google/vit-base-patch16-224"
    n = NodeIn(id="b", kind="hf_classify", config={"model_id": "org/custom"})
    assert n.config["model_id"] == "org/custom"  # the graph's own config wins
    n = NodeIn(id="c", kind="sam3", config={"threshold": 0.7})
    assert n.kind == "segment" and n.config == {"threshold": 0.7, "model_id": "facebook/sam3"}
    assert NodeIn(id="d", kind="hf_segment").config["model_id"] == "facebook/mask2former-swin-tiny-coco-instance"
    assert NodeIn(id="e", kind="dinov2").kind == "embed"
    n = NodeIn(id="f", kind="custom_model")  # merged into custom; fixed handles restored
    assert n.kind == "custom" and n.config["input_ports"] == ["image", "prompts"]
    assert NodeIn(id="g", kind="hf_custom").kind == "custom"
    assert NodeIn(id="h", kind="segment").config == {}  # new kinds pass through untouched


def test_spec_carries_config_options():
    opts = REGISTRY["segment"].spec()["config_options"]["model_id"]
    assert opts[0] == {"value": "facebook/sam3", "label": "SAM3 (promptable)"}
    # plain-string options (no Custom… escape needed frontend-side)
    assert REGISTRY["flip"].spec()["config_options"]["direction"] == ["horizontal", "vertical"]
    assert REGISTRY["blur"].spec()["config_options"] == {}  # nothing declared, nothing exposed


def test_spec_carries_hf_search_tags():
    # every HF model node exposes Hub pipeline tags for the Custom… search
    tags = {k: REGISTRY[k].spec()["config_hf"].get("model_id") for k in ("segment", "classify", "embed", "detect", "depth", "caption")}
    assert tags["detect"] == ["object-detection", "zero-shot-object-detection"]
    assert all(tags.values())
    # depth/caption gained preset options too, so the panel shows the Custom… search
    assert REGISTRY["depth"].spec()["config_options"]["model_id"]
    assert REGISTRY["blur"].spec()["config_hf"] == {}  # non-model nodes expose nothing


def test_hf_detect_zero_shot_requires_prompts():
    # OWL-ViT is sniffed as zero-shot; with nothing wired to prompts there are
    # no candidate labels to detect, so it must fail loudly instead of guessing
    node = REGISTRY["detect"](model_id="hf-internal-testing/tiny-random-OwlViTForObjectDetection")
    with pytest.raises(ValueError, match="Text Prompt"):
        node.run(image=[_white_left_px()])


def test_hf_detect_zero_shot_runs_with_prompts():
    node = REGISTRY["detect"](model_id="hf-internal-testing/tiny-random-OwlViTForObjectDetection", threshold=0.0)
    out = node.run(image=[_white_left_px()], prompts={"text": "cat, dog"})
    assert len(out["boxes"]) == 1 and len(out["labels"]) == 1
    assert out["boxes"][0].dtype == torch.int32 and out["boxes"][0].shape[-1] == 4
    assert isinstance(out["labels"][0], list)


def test_hf_classify_bad_model_id_raises_a_clean_error():
    # a nonexistent repo must surface as one readable line, not an hf_hub/transformers stack dump
    node = REGISTRY["classify"](model_id="hf-internal-testing/this-repo-does-not-exist-kumoflow")
    with pytest.raises(ValueError) as ei:
        node.run(image=[_white_left_px()])
    msg = str(ei.value)
    assert "\n" not in msg and "Traceback" not in msg
    assert "this-repo-does-not-exist-kumoflow" in msg  # names the model, per spec


def test_custom_node_optional_load_feeds_run():
    # a top-level load() runs once and its return is handed to run() as the
    # first argument — the hook that replaced the old Custom Model node
    code = (
        "def load():\n"
        "    return {'greeting': 'hi'}\n"
        "def run(loaded, image=None):\n"
        "    return {'labels': [loaded['greeting']] * len(image)}\n"
    )
    node = REGISTRY["custom"](code=code, output_ports=["labels"])
    assert node.run(image=_two_sizes())["labels"] == ["hi", "hi"]


def test_custom_node_load_cache_keyed_by_code_hash():
    # load()'s cache key is a hash of the code string: identical code reuses
    # the cached result, a single edited character busts it and reruns load()
    code_a = (
        "import uuid\n"
        "def load():\n"
        "    return uuid.uuid4().hex\n"
        "def run(loaded, image=None):\n"
        "    return {'labels': [loaded] * len(image)}\n"
    )
    imgs = _two_sizes()
    first = REGISTRY["custom"](code=code_a, output_ports=["labels"]).run(image=imgs)["labels"][0]
    again = REGISTRY["custom"](code=code_a, output_ports=["labels"]).run(image=imgs)["labels"][0]
    assert again == first  # same code string -> cached load(), not rerun

    code_b = code_a + "\n"  # one trailing newline still changes the hash
    edited = REGISTRY["custom"](code=code_b, output_ports=["labels"]).run(image=imgs)["labels"][0]
    assert edited != first  # edited code -> cache busted, load() ran again


# --- tracking: identity across frames, its overlay, and tracked counts --------


def _clip_boxes(*frames):
    """Per-frame boxes for one clip, the shape SAM3's video path emits:
    one (N,4) int32 tensor per frame, each arg a list of [x1,y1,x2,y2]."""
    return [torch.tensor(f, dtype=torch.int32).reshape(-1, 4) for f in frames]


def _ids(out):
    """A clip's `ids` output as plain lists, one per frame."""
    return [i.tolist() for i in out["ids"][0]]


def test_track_holds_one_id_while_the_box_drifts():
    clip = _clip_boxes([[0, 0, 10, 10]], [[2, 0, 12, 10]], [[4, 0, 14, 10]])
    assert _ids(REGISTRY["track"]().run(boxes=[clip])) == [[1], [1], [1]]


def test_track_survives_a_gap_and_keeps_distant_objects_apart():
    a, b = [0, 0, 10, 10], [100, 100, 110, 110]
    clip = _clip_boxes([a, b], [a], [a, b])  # b is missed in the middle frame
    assert _ids(REGISTRY["track"]().run(boxes=[clip])) == [[1, 2], [1], [1, 2]]
    # max_lost=0 closes b the frame it vanishes, so its return is a third track
    # (both b tracks are one frame long -> flicker -> -1)
    assert _ids(REGISTRY["track"](max_lost=0).run(boxes=[clip])) == [[1, -1], [1], [1, -1]]


def test_track_drops_single_frame_flicker():
    clip = _clip_boxes([[0, 0, 10, 10], [100, 100, 110, 110]], [[0, 0, 10, 10]])
    assert _ids(REGISTRY["track"]().run(boxes=[clip])) == [[1, -1], [1]]
    assert _ids(REGISTRY["track"](min_frames=1).run(boxes=[clip])) == [[1, 2], [1]]


def test_track_never_matches_across_concepts():
    clip = _clip_boxes([[0, 0, 10, 10]], [[0, 0, 10, 10]])  # same place, different concept
    names = [[["person"], ["truck"]]]
    node = REGISTRY["track"](min_frames=1)
    assert _ids(node.run(boxes=[clip], labels=names)) == [[1], [2]]
    assert _ids(node.run(boxes=[clip])) == [[1], [1]]  # nothing keeping them apart -> one track


def test_track_at_threshold_zero_still_needs_real_overlap():
    # the slider reaches 0, which reads as "match anything" — but zero overlap is
    # never the same object, and the cross-concept mask writes exactly zero
    node = REGISTRY["track"](min_frames=1, iou_threshold=0.0)
    far = _clip_boxes([[0, 0, 10, 10]], [[500, 500, 510, 510]])
    assert _ids(node.run(boxes=[far])) == [[1], [2]]
    same_place = _clip_boxes([[0, 0, 10, 10]], [[0, 0, 10, 10]])
    assert _ids(node.run(boxes=[same_place], labels=[[["person"], ["truck"]]])) == [[1], [2]]


def test_track_errors():
    run = REGISTRY["track"]().run
    with pytest.raises(ValueError, match="per-frame boxes"):  # per-image: no frames, no identity
        run(boxes=[torch.zeros((2, 4), dtype=torch.int32)])
    clip = _clip_boxes([[0, 0, 10, 10]])
    with pytest.raises(ValueError, match="not per-item class labels"):
        run(boxes=[clip], labels=["Cat"])  # Load Video's per-clip labels
    with pytest.raises(ValueError, match="line up"):
        run(boxes=[clip], labels=[[["person", "truck"]]])  # labels from a different SAM3


def test_count_counts_each_track_once():
    clip = _clip_boxes([[0, 0, 10, 10], [100, 100, 110, 110]], [[1, 0, 11, 10], [100, 100, 110, 110]])
    names = [[["person", "truck"], ["person", "truck"]]]
    ids = REGISTRY["track"]().run(boxes=[clip], labels=names)["ids"]
    out = REGISTRY["count"]().run(boxes=[clip], labels=names, ids=ids)["counts"]
    assert out == {"concepts": ["person", "truck"], "per_image": [[1, 1]], "total": [1, 1]}  # 4 boxes, 2 objects


def test_count_ignores_flicker_tracks():
    clip = _clip_boxes([[0, 0, 10, 10], [100, 100, 110, 110]], [[0, 0, 10, 10]])
    ids = REGISTRY["track"]().run(boxes=[clip])["ids"]
    assert REGISTRY["count"]().run(boxes=[clip], ids=ids)["counts"] == {
        "concepts": ["objects"], "per_image": [[1]], "total": [1]
    }
    with pytest.raises(ValueError, match="ids don't line up"):
        REGISTRY["count"]().run(boxes=[clip], ids=[[torch.zeros(3, dtype=torch.int32)]])


def test_filter_region_per_image():
    boxes = [torch.tensor([[0, 0, 2, 2], [6, 6, 9, 9], [4, 4, 6, 6]], dtype=torch.int32)]
    masks = [torch.stack([torch.full((10, 10), j, dtype=torch.uint8) for j in range(3)])]
    out = REGISTRY["filter_region"](box="0,0,5,5").run(boxes=boxes, masks=masks, labels=[["a", "b", "c"]])
    # centers: (1,1) in, (7.5,7.5) out, (5,5) exactly on the border -> in (inclusive)
    assert out["boxes"][0].tolist() == [[0, 0, 2, 2], [4, 4, 6, 6]]
    assert out["labels"][0] == ["a", "c"]
    assert out["masks"][0].shape == (2, 10, 10) and out["masks"][0][1, 0, 0] == 2
    assert "ids" not in out  # only claims the ports that were wired in


def test_filter_region_per_frame_follows_ids():
    clip = _clip_boxes([[0, 0, 2, 2], [6, 6, 9, 9]], [[6, 6, 9, 9]])
    ids = [[torch.tensor([1, 2], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)]]
    out = REGISTRY["filter_region"](box="0,0,5,5").run(boxes=[clip], ids=ids)
    assert out["boxes"][0][0].tolist() == [[0, 0, 2, 2]] and len(out["boxes"][0][1]) == 0
    assert out["ids"][0][0].tolist() == [1] and out["ids"][0][1].tolist() == []


def test_filter_region_then_count_in_zone():
    # the zone-counting pipeline: track the whole frame (ids stay stable outside
    # the region), then filter to the zone, then count each id once
    clip = _clip_boxes(
        [[0, 0, 10, 10], [100, 100, 110, 110]],  # object 1 in the zone, object 2 far outside
        [[2, 0, 12, 10], [100, 100, 110, 110]],
    )
    ids = REGISTRY["track"]().run(boxes=[clip])["ids"]
    zone = REGISTRY["filter_region"](box="0,0,50,50").run(boxes=[clip], ids=ids)
    counts = REGISTRY["count"]().run(boxes=zone["boxes"], ids=zone["ids"])["counts"]
    assert counts == {"concepts": ["objects"], "per_image": [[1]], "total": [1]}


def test_filter_region_errors():
    bx = [torch.tensor([[0, 0, 2, 2]], dtype=torch.int32)]
    with pytest.raises(ValueError, match="no region"):
        REGISTRY["filter_region"]().run(boxes=bx)
    with pytest.raises(ValueError, match="x1,y1,x2,y2"):
        REGISTRY["filter_region"](box="1,2,3").run(boxes=bx)
    with pytest.raises(ValueError, match="not per-item class labels"):
        REGISTRY["filter_region"](box="0,0,5,5").run(boxes=bx, labels=["cat"])
    with pytest.raises(ValueError, match="line up"):
        REGISTRY["filter_region"](box="0,0,5,5").run(boxes=bx, labels=[["a", "b"]])
    with pytest.raises(ValueError, match="ids are per-frame"):
        REGISTRY["filter_region"](box="0,0,5,5").run(boxes=bx, ids=[[torch.tensor([1], dtype=torch.int32)]])
    with pytest.raises(ValueError, match="not both"):
        REGISTRY["filter_region"](box="0,0,5,5").run(
            image=[torch.zeros((3, 4, 4), dtype=torch.uint8)],
            video=[torch.zeros((1, 3, 4, 4), dtype=torch.uint8)],
            boxes=bx,
        )
    with pytest.raises(ValueError, match="Sample Frames"):  # per-frame boxes next to an image input
        REGISTRY["filter_region"](box="0,0,5,5").run(
            image=[torch.zeros((3, 4, 4), dtype=torch.uint8)], boxes=[_clip_boxes([[0, 0, 2, 2]])]
        )


def test_view_video_colors_by_track_id():
    from nodes import _track_color

    clip = torch.zeros((2, 3, 40, 40), dtype=torch.uint8)
    boxes = [_clip_boxes([[2, 2, 20, 20]], [[6, 2, 24, 20]])]
    ids = [[torch.tensor([1], dtype=torch.int32)] * 2]
    out = REGISTRY["view_video"]().run(video=[clip], boxes=boxes, ids=ids)["video"][0]
    # one track, one hue, held across frames — and it's the track color, not palette entry 0
    assert out[0][:, 2, 10].tolist() == out[1][:, 2, 14].tolist() == list(_track_color(1))


def test_view_video_ids_need_something_to_color():
    clip = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="ids need the boxes"):
        REGISTRY["view_video"]().run(video=[clip], ids=[[torch.zeros(1, dtype=torch.int32)]])


def test_track_and_count_through_graph():
    # the ids port has to flow over a real edge, not just a direct run() call
    stub = (
        "import torch\n"
        "def run(**inputs):\n"
        "    b = lambda x: torch.tensor([[x, 0, x + 10, 10]], dtype=torch.int32)\n"
        '    return {"boxes": [[b(0), b(2)]], "labels": [[["cat"], ["cat"]]]}\n'
    )
    counts = evaluate(
        "c",
        {
            "k": NodeIn(id="k", kind="custom", config={"output_ports": ["boxes", "labels"], "code": stub}),
            "t": NodeIn(id="t", kind="track"),
            "c": NodeIn(id="c", kind="count"),
        },
        _incoming([
            Edge(source="k", target="t", source_handle="boxes", target_handle="boxes"),
            Edge(source="k", target="t", source_handle="labels", target_handle="labels"),
            Edge(source="k", target="c", source_handle="boxes", target_handle="boxes"),
            Edge(source="k", target="c", source_handle="labels", target_handle="labels"),
            Edge(source="t", target="c", source_handle="ids", target_handle="ids"),
        ]),
    )["counts"]
    assert counts == {"concepts": ["cat"], "per_image": [[1]], "total": [1]}  # one cat, two frames


def test_track_re_links_an_object_that_kept_moving_while_hidden():
    # 10px/frame, 40px box: after a 3-frame occlusion the boxes no longer overlap
    # at all (IoU 0.14), so only carrying the track forward at its own speed
    # re-links it instead of counting a second object
    clip = _clip_boxes([[0, 0, 40, 10]], [[10, 0, 50, 10]], [[20, 0, 60, 10]], [], [], [[50, 0, 90, 10]])
    assert _ids(REGISTRY["track"]().run(boxes=[clip])) == [[1], [1], [1], [], [], [1]]


def test_run_all_computes_shared_work_once(monkeypatch):
    # "Run All": two viewers hanging off one Load Image is a diamond across two
    # targets — the shared cache must run the upstream once, not once per target
    from main import Graph, run_graph
    from nodes import LoadImage

    ran = []
    orig = LoadImage.run
    monkeypatch.setattr(LoadImage, "run", lambda self: (ran.append(1), orig(self))[1])
    out = run_graph(Graph(
        nodes=[
            {"id": "a", "kind": "load", "config": {"data": _png_data_url(_white_left_px())}},
            {"id": "v1", "kind": "view", "config": {}},
            {"id": "v2", "kind": "view", "config": {}},
        ],
        edges=[{"source": "a", "target": "v1"}, {"source": "a", "target": "v2"}],
        targets=["v1", "v2"],
    ))["results"]
    assert len(ran) == 1
    assert "images" in out["v1"] and "images" in out["v2"]


def test_stop_gives_up_at_the_next_node(monkeypatch):
    # POST /stop flips the run's flag; the next node that hasn't started yet
    # raises Cancelled, which comes back as "cancelled" rather than an error
    from main import RUNS, Graph, run_graph, stop
    from nodes import LoadImage

    orig = LoadImage.run
    monkeypatch.setattr(LoadImage, "run", lambda self: (stop("r1"), orig(self))[1])
    out = run_graph(Graph(
        nodes=[
            {"id": "a", "kind": "load", "config": {"data": _png_data_url(_white_left_px())}},
            {"id": "v", "kind": "view", "config": {}},
        ],
        edges=[{"source": "a", "target": "v"}],
        targets=["v"],
        run_id="r1",
    ))["results"]
    assert out["v"] == {"cancelled": True}
    assert "r1" not in RUNS  # the run cleans up its slot either way


def test_progress_reports_work_units_and_stops():
    # what a node's _total/_step write for GET /progress, and that a set cancel
    # flag surfaces as Cancelled at the next work unit
    import nodes

    run = {"cancel": False, "node": "m", "nodes": {}}
    token = nodes.PROGRESS.set(run)
    try:
        masks = [[torch.ones((1, 4, 4), dtype=torch.uint8)] * 3]  # one clip, three frames
        REGISTRY["mask_ops"](fill_holes=True).run(masks=masks)
        assert run["nodes"]["m"] == [3, 3]  # per-frame total, ticked off frame by frame

        run["cancel"] = True
        with pytest.raises(nodes.Cancelled):
            REGISTRY["mask_ops"](fill_holes=True).run(masks=masks)
    finally:
        nodes.PROGRESS.reset(token)


# --- cross-run result cache (CACHE + _sigs in main.py) ------------------------


@pytest.fixture
def counted():
    """Tiny fake nodes with per-tag run counters, on a clean global CACHE.
    Subclassing Node auto-registers; the teardown unregisters and re-clears."""
    from nodes import Node

    calls: dict[str, int] = {}

    def tick(tag):
        calls[tag] = calls.get(tag, 0) + 1

    class TSrc(Node):
        kind = "t_src"
        outputs = ["image"]
        tag: str = "src"
        value: int = 1

        def run(self):
            tick(self.tag)
            return {"image": torch.full((3, 2, 2), self.value, dtype=torch.uint8)}

    class TWork(Node):
        kind = "t_work"
        inputs = ["image"]
        outputs = ["image"]
        tag: str = "work"
        add: int = 0

        def run(self, image):
            tick(self.tag)
            return {"image": image + self.add}

    class TImpure(Node):
        kind = "t_impure"
        cacheable = False
        inputs = ["image"]
        outputs = ["image"]
        tag: str = "impure"

        def run(self, image):
            tick(self.tag)
            return {"image": image}

    class TFlaky(Node):
        kind = "t_flaky"
        outputs = ["image"]
        tag: str = "flaky"

        def run(self):
            tick(self.tag)
            if calls[self.tag] == 1:
                raise ValueError("boom")
            return {"image": torch.zeros((3, 2, 2), dtype=torch.uint8)}

    class TPrompt(Node):
        kind = "t_prompt"
        outputs = ["prompts"]
        tag: str = "prompt"
        text: str = ""

        def run(self):
            tick(self.tag)
            return {"prompts": {"text": self.text}}

    class TMerge(Node):
        kind = "t_merge"
        inputs = ["prompts"]
        outputs = ["image"]
        tag: str = "merge"

        def run(self, prompts):
            tick(self.tag)
            return {"image": torch.zeros((3, 2, 2), dtype=torch.uint8)}

    yield calls
    for k in ("t_src", "t_work", "t_impure", "t_flaky", "t_prompt", "t_merge"):
        REGISTRY.pop(k, None)


def _run_cached(nodes, edges, target, run=None):
    """One /run-shaped evaluation: structural sigs + the global CACHE."""
    by_id = {n.id: n for n in nodes}
    incoming = _incoming(edges)
    run = run if run is not None else {"cancel": False, "node": "", "live": None}
    return evaluate(target, by_id, incoming, {}, run, _sigs(by_id, incoming))


def test_cache_rewiring_downstream_keeps_upstream_cached(counted):
    src = NodeIn(id="s", kind="t_src")
    exp = NodeIn(id="e", kind="t_work", config={"tag": "exp"})
    view = NodeIn(id="v", kind="t_work", config={"tag": "view"})
    edges = [Edge(source="s", target="e"), Edge(source="e", target="v")]
    _run_cached([src, exp, view], edges, "v")
    assert counted == {"src": 1, "exp": 1, "view": 1}
    # rewire the tail: a new transform between the expensive node and the viewer
    mid = NodeIn(id="m", kind="t_work", config={"tag": "mid", "add": 1})
    edges = [Edge(source="s", target="e"), Edge(source="e", target="m"), Edge(source="m", target="v")]
    _run_cached([src, exp, mid, view], edges, "v")
    assert counted == {"src": 1, "exp": 1, "mid": 1, "view": 2}  # only the tail re-ran


def test_cache_config_change_reruns_node_and_descendants(counted):
    nodes = [
        NodeIn(id="s", kind="t_src"),
        NodeIn(id="e", kind="t_work", config={"tag": "exp"}),
        NodeIn(id="v", kind="t_work", config={"tag": "view"}),
    ]
    edges = [Edge(source="s", target="e"), Edge(source="e", target="v")]
    _run_cached(nodes, edges, "v")
    nodes[1] = NodeIn(id="e", kind="t_work", config={"tag": "exp", "add": 5})
    _run_cached(nodes, edges, "v")
    assert counted == {"src": 1, "exp": 2, "view": 2}


def test_cache_ignores_node_ids(counted):
    def graph(sid, eid):
        nodes = [NodeIn(id=sid, kind="t_src"), NodeIn(id=eid, kind="t_work", config={"tag": "exp"})]
        return nodes, [Edge(source=sid, target=eid)]

    _run_cached(*graph("a", "b"), "b")
    _run_cached(*graph("x", "y"), "y")  # same structure, renamed ids
    assert counted == {"src": 1, "exp": 1}


def test_cache_hit_stops_recursion_into_upstream(counted):
    nodes = [NodeIn(id="s", kind="t_src"), NodeIn(id="e", kind="t_work", config={"tag": "exp"})]
    edges = [Edge(source="s", target="e")]
    sigs = _sigs({n.id: n for n in nodes}, _incoming(edges))
    _run_cached(nodes, edges, "e")
    del graph.CACHE[sigs["s"]]  # even with the source's entry evicted...
    _run_cached(nodes, edges, "e")  # ...the hit at "e" never visits "s"
    assert counted == {"src": 1, "exp": 1}


def test_cache_eviction_is_global_lru_and_never_hurts_the_live_run(counted, monkeypatch):
    monkeypatch.setattr(graph, "CACHE_BYTES", 40)  # each entry is ~17 bytes
    nodes1 = [NodeIn(id="s", kind="t_src"), NodeIn(id="e", kind="t_work", config={"tag": "exp"})]
    _run_cached(nodes1, [Edge(source="s", target="e")], "e")
    assert len(graph.CACHE) == 2
    nodes2 = [
        NodeIn(id="s", kind="t_src", config={"value": 9}),
        NodeIn(id="e", kind="t_work", config={"tag": "exp2", "add": 1}),
        NodeIn(id="v", kind="t_work", config={"tag": "view2", "add": 2}),
    ]
    edges2 = [Edge(source="s", target="e"), Edge(source="e", target="v")]
    # completes even though eviction drops its own oldest entry mid-run: the
    # per-run cache holds the references the global LRU no longer does
    out = _run_cached(nodes2, edges2, "v")
    assert out["image"][0, 0, 0] == 12  # 9 + 1 + 2
    sigs2 = _sigs({n.id: n for n in nodes2}, _incoming(edges2))
    # pure oldest-first down to budget: run1's entries and run2's "s" are gone
    assert set(graph.CACHE) == {sigs2[n] for n in ("e", "v")}


def test_cache_uncacheable_poisons_descendants(counted):
    nodes = [
        NodeIn(id="s", kind="t_src"),
        NodeIn(id="i", kind="t_impure"),
        NodeIn(id="v", kind="t_work", config={"tag": "view"}),
    ]
    edges = [Edge(source="s", target="i"), Edge(source="i", target="v")]
    sigs = _sigs({n.id: n for n in nodes}, _incoming(edges))
    assert sigs["s"] is not None and sigs["i"] is None and sigs["v"] is None
    _run_cached(nodes, edges, "v")
    _run_cached(nodes, edges, "v")
    assert counted == {"src": 1, "impure": 2, "view": 2}  # src is above the impure node: still cached


def test_load_video_cache_extra_tracks_files(tmp_path):
    import os

    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")  # never decoded — cache_extra only stats
    inst = REGISTRY["load_video"](data=str(tmp_path))
    before = inst.cache_extra()
    assert "a.mp4" in before
    os.utime(clip, ns=(0, 1))  # touch: editing a video on disk must change the sig
    assert inst.cache_extra() != before
    assert "missing" in REGISTRY["load_video"](data=str(tmp_path / "nope.mp4")).cache_extra()
    assert REGISTRY["load_video"](data="data:video/mp4;base64,AAAA").cache_extra() == ""


def test_cache_never_stores_failures(counted):
    from main import NodeError

    nodes = [NodeIn(id="f", kind="t_flaky")]
    with pytest.raises(NodeError):
        _run_cached(nodes, [], "f")
    _run_cached(nodes, [], "f")  # nothing was cached: re-runs, succeeds
    _run_cached(nodes, [], "f")  # now it is cached
    assert counted == {"flaky": 2}


def test_cache_multi_edge_order_is_part_of_the_sig(counted):
    nodes = [
        NodeIn(id="p1", kind="t_prompt", config={"text": "car"}),
        NodeIn(id="p2", kind="t_prompt", config={"text": "truck"}),
        NodeIn(id="m", kind="t_merge"),
    ]
    fwd = [
        Edge(source="p1", target="m", source_handle="prompts", target_handle="prompts"),
        Edge(source="p2", target="m", source_handle="prompts", target_handle="prompts"),
    ]
    by_id = {n.id: n for n in nodes}
    # reordered merge edges = a miss (never a wrong hit) ...
    assert _sigs(by_id, _incoming(fwd))["m"] != _sigs(by_id, _incoming(list(reversed(fwd))))["m"]
    _run_cached(nodes, fwd, "m")
    _run_cached(nodes, fwd, "m")  # ... and identical wiring reuses
    assert counted["merge"] == 1


# --- Export ------------------------------------------------------------------


def _export_zip(out, media):
    import zipfile

    assert out["download"].startswith("/media/")
    return zipfile.ZipFile(media / out["download"].removeprefix("/media/"))


def test_export_bundles_everything_wired(tmp_path, monkeypatch):
    import json

    import nodes

    monkeypatch.setattr(nodes, "MEDIA", tmp_path)
    image = [torch.zeros((3, 4, 5), dtype=torch.uint8), torch.zeros((3, 6, 7), dtype=torch.uint8)]
    masks = [torch.ones((2, 4, 5), dtype=torch.uint8), torch.zeros((0, 6, 7), dtype=torch.uint8)]
    boxes = [torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=torch.int32), torch.zeros((0, 4), dtype=torch.int32)]
    out = REGISTRY["export"]().run(
        image=image,
        masks=masks,
        boxes=boxes,
        labels=[["cat", "dog"], []],
        embedding=[torch.arange(4, dtype=torch.float32), torch.ones(4)],
        classification=[{"labels": ["a"], "scores": [0.5]}, {"labels": ["a"], "scores": [0.1]}],
    )
    zf = _export_zip(out, tmp_path)
    assert set(zf.namelist()) == {
        "images/item_000.png",
        "images/item_001.png",
        "masks/item_000/obj_00.png",
        "masks/item_000/obj_01.png",
        "embeddings.npy",
        "predictions.json",
    }
    preds = json.loads(zf.read("predictions.json"))
    assert preds["items"][0] == {
        "boxes": [[0, 0, 2, 2], [1, 1, 3, 3]],
        "labels": ["cat", "dog"],
        "classification": {"labels": ["a"], "scores": [0.5]},
    }
    assert preds["items"][1]["boxes"] == []
    import io

    emb = np.load(io.BytesIO(zf.read("embeddings.npy")))
    assert emb.shape == (2, 4) and emb[0, 3] == 3.0
    # exported masks are 0/255 grayscale at source size
    from torchvision.io import decode_image

    m = decode_image(torch.frombuffer(bytearray(zf.read("masks/item_000/obj_00.png")), dtype=torch.uint8))
    assert m.shape == (1, 4, 5) and int(m.max()) == 255
    assert out["summary"].startswith("6 files, ")


def test_export_video_layout_and_determinism(tmp_path, monkeypatch):
    import nodes

    monkeypatch.setattr(nodes, "MEDIA", tmp_path)
    video = [torch.zeros((2, 3, 4, 5), dtype=torch.uint8)]
    masks = [[torch.ones((1, 4, 5), dtype=torch.uint8), torch.zeros((0, 4, 5), dtype=torch.uint8)]]
    ids = [[torch.tensor([7], dtype=torch.int32), torch.zeros((0,), dtype=torch.int32)]]
    run = lambda: REGISTRY["export"]().run(video=video, masks=masks, ids=ids)
    out = run()
    zf = _export_zip(out, tmp_path)
    assert set(zf.namelist()) == {
        "video/item_000/frame_000.png",
        "video/item_000/frame_001.png",
        "masks/item_000/frame_000_obj_00.png",
        "predictions.json",
    }
    assert run()["download"] == out["download"]  # content-addressed: same data, same file


def test_export_errors(tmp_path, monkeypatch):
    import nodes

    monkeypatch.setattr(nodes, "MEDIA", tmp_path)
    image = [torch.zeros((3, 4, 5), dtype=torch.uint8)]
    with pytest.raises(ValueError, match="wire something"):
        REGISTRY["export"]().run()
    with pytest.raises(ValueError, match="not both"):
        REGISTRY["export"]().run(image=image, video=[torch.zeros((2, 3, 4, 5), dtype=torch.uint8)])
    with pytest.raises(ValueError, match="batch sizes differ"):
        REGISTRY["export"]().run(image=image, labels=["a", "b"])
    with pytest.raises(ValueError, match="per-frame"):  # video masks alongside an image batch
        REGISTRY["export"]().run(image=image, masks=[[torch.ones((1, 4, 5), dtype=torch.uint8)]])


def test_export_content_addressed_zip_reopens(tmp_path, monkeypatch):
    # the zip is now streamed to a temp file and renamed in, not built whole
    # in a BytesIO — still content-addressed, still a valid zip
    import zipfile

    import nodes

    monkeypatch.setattr(nodes, "MEDIA", tmp_path)
    image = [torch.zeros((3, 4, 5), dtype=torch.uint8)]
    out1 = REGISTRY["export"]().run(image=image)
    out2 = REGISTRY["export"]().run(image=image)
    assert out1["download"] == out2["download"]  # same content -> same file, no duplicate
    path = tmp_path / out1["download"].removeprefix("/media/")
    assert path.exists()
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == ["images/item_000.png"]


def test_shape_caps_video_flattened_preview(monkeypatch):
    # video target: every frame flattened clip-major, capped at PREVIEW_MAX_FRAMES
    # evenly-spaced true indices — a real 100k-frame run would otherwise be
    # 100k JPEG encodes, so the cap is monkeypatched small instead
    monkeypatch.setattr(main, "PREVIEW_MAX_FRAMES", 5)
    clips = [torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8) for _ in range(3)]  # 12 frames total
    outs = {"video": clips}
    by_id = {"t": NodeIn(id="t", kind="visual_prompt")}  # any kind but view_video/view_volume

    res = main._shape("t", outs, by_id, [])
    assert len(res["images"]) == 5 == len(res["sizes"])
    fi = res["frame_indices"]
    assert len(fi) == 5 and fi[0] == 0 and fi[-1] == 11
    assert all(fi[i] < fi[i + 1] for i in range(len(fi) - 1))  # strictly increasing

    monkeypatch.setattr(main, "PREVIEW_MAX_FRAMES", 20)  # 12 <= 20: uncapped
    res = main._shape("t", outs, by_id, [])
    assert len(res["images"]) == 12
    assert "frame_indices" not in res


def test_export_through_graph(tmp_path, monkeypatch):
    import nodes

    monkeypatch.setattr(nodes, "MEDIA", tmp_path)
    n = [
        NodeIn(id="l", kind="load", config={"data": _png_data_url(torch.zeros((3, 4, 5), dtype=torch.uint8))}),
        NodeIn(id="e", kind="export"),
    ]
    edges = [Edge(source="l", target="e")]  # default handles: image -> image
    by_id = {x.id: x for x in n}
    outs = evaluate("e", by_id, _incoming(edges))
    shaped = main._shape("e", outs, by_id, edges)  # /run hands the URL straight through
    assert shaped["download"].endswith(".zip") and shaped["summary"].startswith("1 files")


class _FakeProc:
    """Stands in for subprocess.Popen so deploy-lifecycle tests never spawn servers."""

    def __init__(self, args, **kwargs):
        self.args = args
        self.env = kwargs.get("env", {})
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._alive = False


@pytest.fixture
def deploy_backend(tmp_path, monkeypatch):
    """POST /deploy against a temp artifact dir, with process spawning stubbed."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "DEPLOYMENTS", tmp_path)
    monkeypatch.setattr(main.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(main, "PROCS", {})
    return TestClient(main.app)


def test_deploy_saves_and_serves_a_standalone_pipeline(tmp_path, deploy_backend):
    """POST /deploy snapshots the graph; deploy.create_app serves it without the frontend."""
    import json

    from fastapi.testclient import TestClient

    import deploy

    graph = {
        "nodes": [
            {"id": "in", "kind": "load", "config": {"data": _png_data_url(_white_left_px())}},
            {"id": "out", "kind": "view", "config": {}},
        ],
        "edges": [{"source": "in", "target": "out"}],
        "targets": ["out"],
        "run_id": "editor-run",  # must not be baked into the snapshot
    }
    saved = deploy_backend.post("/deploy", json={"graph": graph, "name": "My Pipeline!"}).json()
    path = tmp_path / "My_Pipeline.json"
    assert saved["path"] == str(path) and "run_id" not in json.loads(path.read_text())
    assert saved["url"] == "http://localhost:8001"  # spawned live on the default port

    api = TestClient(deploy.create_app(path))
    info = api.get("/info").json()
    assert info["workflow"] == "My_Pipeline" and info["targets"] == ["out"]
    assert info["inputs"] == [{"id": "in", "kind": "load"}]
    assert info["ui"] is None and info["kinds"] == {"in": "load", "out": "view"}
    assert api.get("/").status_code == 404  # API-only artifact: no app page
    assert api.post("/run").json()["results"]["out"]["sizes"] == [[2, 1]]  # the deployed inputs


def test_deploy_lifecycle_replace_conflict_stop(tmp_path, deploy_backend):
    """Deployment identity = name: replace in place, 409 on a foreign port, list + stop."""
    import json

    graph = {
        "nodes": [{"id": "in", "kind": "load", "config": {"data": _png_data_url(_white_left_px())}},
                  {"id": "out", "kind": "view", "config": {}}],
        "edges": [{"source": "in", "target": "out"}],
        "targets": ["out"],
    }
    ui = {"title": "Counter", "sections": [{"node": "out", "label": "Result"}]}
    first = deploy_backend.post("/deploy", json={"graph": graph, "ui": ui, "name": "app", "port": 8002}).json()
    assert first["url"] == "http://localhost:8002"
    assert json.loads((tmp_path / "app.json").read_text())["ui"] == ui
    assert deploy_backend.get("/deployments").json() == [{"name": "app", "port": 8002, "has_ui": True, "alive": True}]
    proc1 = main.PROCS["app"]["proc"]
    assert proc1.env["PORT"] == "8002"  # deploy.py reads the port from its env

    # same name = replace in place: old process stopped, same URL stays valid
    deploy_backend.post("/deploy", json={"graph": graph, "ui": ui, "name": "app", "port": 8002})
    assert proc1.poll() is not None and main.PROCS["app"]["proc"] is not proc1

    # a *different* deployment asking for a held port is refused with the holder's name
    r = deploy_backend.post("/deploy", json={"graph": graph, "name": "other", "port": 8002})
    assert r.status_code == 409 and "app" in r.json()["detail"]

    # stop: the process goes, the artifact file stays (one command from live again)
    assert deploy_backend.delete("/deployments/app").status_code == 200
    assert deploy_backend.get("/deployments").json() == []
    assert (tmp_path / "app.json").exists()
    assert deploy_backend.delete("/deployments/app").status_code == 404


def test_deployed_app_page_serves_ui(tmp_path, monkeypatch):
    """An artifact with a ui spec serves the app page at / and its spec at /info."""
    import json

    from fastapi.testclient import TestClient

    import deploy

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "app.html").write_text("<!doctype html><title>app</title>")
    monkeypatch.setattr(deploy, "DIST", dist)
    ui = {"title": "Counter", "sections": [{"node": "out", "label": "Result"}]}
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({
        "nodes": [
            {"id": "in", "kind": "load", "config": {"data": _png_data_url(_white_left_px())}},
            {"id": "out", "kind": "view", "config": {}},
        ],
        "edges": [{"source": "in", "target": "out"}],
        "targets": ["out"],
        "ui": ui,
    }))
    api = TestClient(deploy.create_app(wf))
    info = api.get("/info").json()
    assert info["ui"] == ui and info["kinds"] == {"in": "load", "out": "view"}
    assert "app" in api.get("/").text  # the page, not JSON
    assert api.post("/run").json()["results"]["out"]["sizes"] == [[2, 1]]  # ui never reaches the Graph


def test_deploy_run_swaps_uploaded_input(tmp_path):
    import json

    from fastapi.testclient import TestClient

    import deploy

    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({
        "nodes": [
            {"id": "in", "kind": "load", "config": {"data": _png_data_url(_white_left_px()), "labels": ["old"]}},
            {"id": "out", "kind": "view", "config": {}},
        ],
        "edges": [{"source": "in", "target": "out"}],
        "targets": ["out"],
    }))
    api = TestClient(deploy.create_app(wf))
    png = encode_png(torch.zeros((3, 4, 5), dtype=torch.uint8)).numpy().tobytes()
    # single input node: any field name lands on it ("image" reads well client-side)
    r = api.post("/run", files=[("image", ("a.png", png, "image/png"))]).json()["results"]["out"]
    assert r["sizes"] == [[5, 4]]  # the upload, not the deployed input
    assert api.post("/run").json()["results"]["out"]["sizes"] == [[2, 1]]  # override didn't stick


def test_deploy_run_rejects_ambiguous_input_field(tmp_path):
    import json

    from fastapi.testclient import TestClient

    import deploy

    url = _png_data_url(_white_left_px())
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({
        "nodes": [
            {"id": "a", "kind": "load", "config": {"data": url}},
            {"id": "b", "kind": "load", "config": {"data": url}},
            {"id": "out", "kind": "view", "config": {}},
        ],
        "edges": [{"source": "a", "target": "out"}],
        "targets": ["out"],
    }))
    api = TestClient(deploy.create_app(wf))
    png = encode_png(torch.zeros((3, 4, 5), dtype=torch.uint8)).numpy().tobytes()
    # two input nodes: an unmatched field name can't be assigned
    assert api.post("/run", files=[("image", ("x.png", png, "image/png"))]).status_code == 400
    r = api.post("/run", files=[("a", ("x.png", png, "image/png"))]).json()["results"]["out"]
    assert r["sizes"] == [[5, 4]]  # field "a" swapped exactly node a


def test_client_against_live_deployed_pipeline(tmp_path):
    """The stdlib client end-to-end: real uvicorn server, real HTTP."""
    import json
    import socket
    import threading
    import time

    import uvicorn

    import client
    import deploy

    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({
        "nodes": [
            {"id": "in", "kind": "load", "config": {"data": _png_data_url(_white_left_px())}},
            {"id": "out", "kind": "view", "config": {}},
        ],
        "edges": [{"source": "in", "target": "out"}],
        "targets": ["out"],
    }))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(deploy.create_app(wf), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        pipe = client.Pipeline(f"127.0.0.1:{port}")  # scheme-less, like the docstring example
        assert pipe.info()["inputs"] == [{"id": "in", "kind": "load"}]
        img = tmp_path / "cat.png"
        img.write_bytes(encode_png(torch.zeros((3, 4, 5), dtype=torch.uint8)).numpy().tobytes())
        result = pipe.run(images={"image": img})
        assert result["out"]["sizes"] == [[5, 4]]
        assert pipe.fetch(result["out"]["images"][0])[:2] == b"\xff\xd8"  # the jpeg preview bytes
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_vram_mutual_eviction(monkeypatch):
    # /run evicts the chat model and survives Ollama being down (error
    # swallowed); free_models — /chat's mirror move — drops the cached models
    import chat
    import nodes
    from main import Graph, run_graph

    calls = []

    def down():
        calls.append(1)
        raise OSError("ollama unreachable")

    monkeypatch.setattr(chat, "unload_model", down)
    g = Graph(
        nodes=[{"id": "a", "kind": "load", "config": {"data": _png_data_url(_white_left_px())}}],
        edges=[],
        targets=["a"],
    )
    assert "images" in run_graph(g)["results"]["a"] and calls

    nodes._MODELS["fake"] = object()
    nodes.free_models()
    assert nodes._MODELS == {}


# --- SAHI sliced inference ---


def test_slice_grid_covers_frame_with_requested_overlap():
    from nodes import _slice_grid

    for h, w, side, overlap in [(10, 10, 4, 0.2), (23, 37, 10, 0.25), (17, 50, 20, 0.4)]:
        grid = _slice_grid(h, w, side, overlap)
        assert grid[-1] == (0, 0, w, h)  # full-frame pass always last
        assert len(grid) == len(set(grid))  # no duplicate tiles
        cnt = torch.zeros((h, w), dtype=torch.int32)
        for x1, y1, x2, y2 in grid:
            cnt[y1:y2, x1:x2] += 1
        assert (cnt >= 1).all()  # union of tiles covers the frame exactly
        stride = max(1, round(side * (1 - overlap)))
        req_overlap = side - stride
        xs = sorted({x1 for x1, y1, x2, y2 in grid[:-1]})
        for a, b in zip(xs, xs[1:]):
            assert (a + side) - b >= req_overlap  # neighbor overlap >= requested


def test_slice_grid_empty_when_it_fits_one_tile():
    from nodes import _slice_grid

    assert _slice_grid(20, 30, 40, 0.2) == []
    assert _slice_grid(40, 40, 40, 0.2) == []  # exactly flush also counts as fitting


def test_detect_slicing_offsets_and_merges_class_aware(monkeypatch):
    # two overlapping tiles: tile 1's detections use different LOCAL coords than
    # tile 0's, but land on the exact same GLOBAL box once offset by tile 1's
    # origin — proves the offset math, not just coincidentally-equal boxes
    import nodes

    monkeypatch.setattr(nodes, "_slice_grid", lambda h, w, side, overlap: [(0, 0, 10, 10), (2, 0, 12, 10)])

    def fake_pipe(sub, threshold=0.0, **kw):
        per_tile = [
            [{"score": 0.9, "label": "cat", "box": {"xmin": 2, "ymin": 2, "xmax": 8, "ymax": 8}}],
            [
                {"score": 0.6, "label": "cat", "box": {"xmin": 0, "ymin": 2, "xmax": 6, "ymax": 8}},
                {"score": 0.5, "label": "dog", "box": {"xmin": 0, "ymin": 2, "xmax": 6, "ymax": 8}},
            ],
        ]
        return per_tile[: len(sub)]

    monkeypatch.setattr(nodes, "_hf_pipeline", lambda *a, **kw: fake_pipe)
    node = REGISTRY["detect"](model_id="dummy", threshold=0.0, slice_size=10, slice_overlap=0.2)
    out = node.run(image=[torch.zeros((3, 10, 12), dtype=torch.uint8)])
    boxes, labels = out["boxes"][0], out["labels"][0]
    assert sorted(labels) == ["cat", "dog"]  # same position, different labels: never suppressed across classes
    assert labels.count("cat") == 1  # same-label duplicates from the two overlapping tiles merged
    assert boxes.tolist() == [[2, 2, 8, 8]] * len(boxes)  # tile 1's differing local box lands on tile 0's global one


def test_segment_slicing_rejects_drawn_prompts_and_pipeline_models():
    # drawn prompts (points, or boxes alone, or boxes alongside text) are global
    # coords tied to the whole image — meaningless once slicing tiles it
    node = REGISTRY["segment"](slice_size=64)
    img = [_white_left_px()]
    with pytest.raises(ValueError, match="Text Prompt only"):
        node.run(image=img, prompts={"per_image": [{"points": [[0, 0]], "point_labels": [1]}]})
    with pytest.raises(ValueError, match="Text Prompt only"):
        node.run(image=img, prompts={"text": "cat", "per_image": [{"boxes": [[0, 0, 1, 1]], "box_labels": [1]}]})
    clip = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="Text Prompt only"):
        node.run(video=[clip], prompts={"per_image": [{"points": [[1, 1]], "point_labels": [1]}]})

    pipeline_node = REGISTRY["segment"](
        model_id="hf-internal-testing/tiny-random-Mask2FormerForUniversalSegmentation", slice_size=64
    )
    with pytest.raises(ValueError, match="use Detect"):
        pipeline_node.run(image=img)


def test_segment_slicing_pastes_tiles_and_merges_duplicates(monkeypatch):
    # frame (10,20) sliced at side=10/overlap=0.5 -> tiles (0,0,10,10), (5,0,15,10),
    # (10,0,20,10), plus the full-frame pass (0,0,20,10). Stub places an instance
    # at local (2:5, 2:5) in both the first grid tile and the full-frame tile —
    # both share origin (0,0), so they paste to the SAME global position and
    # must merge into one instance.
    import nodes

    def stub_seg_concepts(model, proc, hwc, phrases, threshold, with_scores=False, **kw):
        out = []
        for idx, im in enumerate(hwc):
            h, w = im.shape[:2]
            if idx in (0, len(hwc) - 1):
                m = torch.zeros((1, h, w), dtype=torch.bool)
                m[0, 2:5, 2:5] = True
            else:
                m = torch.zeros((0, h, w), dtype=torch.bool)
            entry = (m, torch.ones(m.shape[0])) if with_scores else m
            out.append((entry,))  # one phrase ("cat")
        return out

    monkeypatch.setattr(nodes, "_concept_model", lambda: (None, None))
    monkeypatch.setattr(nodes, "_seg_concepts", stub_seg_concepts)
    node = REGISTRY["segment"](slice_size=10, slice_overlap=0.5)
    out = node.run(image=[torch.zeros((3, 10, 20), dtype=torch.uint8)], prompts={"text": "cat"})
    assert out["masks"][0].shape[0] == 1  # duplicate pasted instances merged
    ys, xs = out["masks"][0][0].nonzero(as_tuple=True)
    assert (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) == (2, 4, 2, 4)  # pasted at tile 0's offset
    assert out["boxes"][0].tolist() == [[2, 2, 4, 4]]
    assert out["labels"][0] == ["cat"]


def test_hf_model_cache_keys_by_model_id_and_free_models_clears_them():
    # unlike SAM3's fixed-string keys, the hf loader's key includes model_id, so
    # two different models land side by side instead of colliding or evicting each other
    import nodes

    nodes._MODELS[("hf", "image-classification", "model-a")] = object()
    nodes._MODELS[("hf", "image-classification", "model-b")] = object()
    assert {("hf", "image-classification", "model-a"), ("hf", "image-classification", "model-b")} <= nodes._MODELS.keys()
    nodes.free_models()
    assert nodes._MODELS == {}
