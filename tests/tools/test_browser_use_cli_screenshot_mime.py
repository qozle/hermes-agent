"""LOCAL COMMIT (frankentrait): the browser tool attaches the screenshot a script printed with the
image type its BYTES say, never a hard-coded ``image/png``.

``_find_screenshot`` accepts any printed ``.png/.jpg/.webp`` path, so a worker that wrote a CDP
``format='webp'`` capture and printed its path had the webp attached as ``image/png``; Anthropic
rejected that request and every later one of the run (the mislabelled image stays in history).
"""
from __future__ import annotations

import base64
import io

import pytest

from tools import browser_use_cli


@pytest.fixture()
def native_vision_on(monkeypatch):
    from tools import vision_tools
    monkeypatch.setattr(vision_tools, "_should_use_native_vision_fast_path", lambda: True)


def _tiny_png_bytes() -> bytes:
    PIL = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    PIL.new("RGB", (4, 4), (0, 200, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_a_webp_capture_is_attached_as_webp(tmp_path, native_vision_on):
    path = tmp_path / "thumb.webp"
    path.write_bytes(b"RIFF" + (100).to_bytes(4, "little") + b"WEBPVP8 " + bytes(64))
    out = browser_use_cli._native_screenshot_result({"success": True}, str(path))
    assert out is not None and out["_multimodal"] is True
    url = out["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/webp;base64,")
    assert base64.b64decode(url.split(",", 1)[1]).startswith(b"RIFF")


def test_a_real_png_is_still_attached_as_png(tmp_path, native_vision_on):
    path = tmp_path / "shot.png"
    path.write_bytes(_tiny_png_bytes())
    out = browser_use_cli._native_screenshot_result({"success": True}, str(path))
    assert out is not None
    assert out["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_file_that_is_no_image_is_not_attached(tmp_path, native_vision_on):
    path = tmp_path / "notes.png"          # the extension lies: it is text
    path.write_text("just a log line with a .png name\n")
    assert browser_use_cli._native_screenshot_result({"success": True}, str(path)) is None
    assert browser_use_cli._sniffed_image_mime(str(tmp_path / "missing.png")) is None
