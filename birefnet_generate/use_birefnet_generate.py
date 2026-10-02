"""
Local, open-source replacement for use_gpt_generate.py.

Background removal: rembg + BiRefNet (MIT licensed, runs on CPU, no API key).
Formatting: crop to the garment, center it on a transparent 1024x1536 (3:4-ish)
canvas with even padding, optional mild lighting touch-up.

Sharpening: Real-ESRGAN (realesrgan-ncnn-vulkan, MIT licensed, no API key),
4x super-resolution on the cropped garment before it's placed on the canvas.
Small source photos (most of test_images/) would otherwise look blurry once
stretched up to CANVAS_SIZE.

Install:
    pip install "rembg[cpu]" pillow
    # or "rembg[gpu]" if you have CUDA

First run downloads the birefnet-general model (~1 GB) to ~/.rembg/models/,
and the realesrgan-ncnn-vulkan binary + models (~150 MB, macOS arm64/x64) to
birefnet_generate/bin/.
"""

import os
import re
import subprocess
import tempfile
import urllib.request
import zipfile
from datetime import datetime

from PIL import Image, ImageEnhance
from rembg import new_session, remove

# These are always our own photos, never untrusted uploads, and the 4x
# super-res pass legitimately produces large images, so disable PIL's
# decompression-bomb safety cap.
Image.MAX_IMAGE_PIXELS = None

# ---- Settings ---------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "test_images")

# Paste a space- or newline-separated list of paths (relative to the project
# root, e.g. "test_images/foo.jpg") straight in here.
IMAGE_LIST_RAW = """

"""

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_image_list(raw: str) -> list[str]:
    """Split a pasted, space/newline-separated path list, tolerating spaces
    inside filenames by merging tokens until one ends in an image extension.

    Splits on plain ASCII space/tab/newline only (not str.split()'s default,
    which treats any Unicode whitespace as a separator) because macOS
    timestamp-suffixed filenames (e.g. "photo 5.02.43 AM.jpg") contain a
    narrow no-break space before AM/PM that must stay part of the filename.
    """
    paths, buffer = [], []
    for token in re.split(r"[ \t\r\n]+", raw.strip()):
        if not token:
            continue
        buffer.append(token)
        if token.lower().endswith(IMAGE_EXTENSIONS):
            paths.append(os.path.join(PROJECT_ROOT, " ".join(buffer)))
            buffer = []
    return paths


IMAGE_PATHS = parse_image_list(IMAGE_LIST_RAW)

# Model options (quality vs. speed/size):
#   "birefnet-general"      best edges, slowest (~1 GB)
#   "birefnet-general-lite" good compromise
#   "isnet-general-use"     fast, decent (~170 MB)
#   "u2net"                 classic, fastest, rougher edges
MODEL_NAME = "birefnet-general"

CANVAS_SIZE = (1024, 1536)  # same as your GPT output
PADDING = 0.08              # fraction of canvas kept empty around the item
ENHANCE = False             # True = slight brightness/contrast lift (changes colors a bit)

ENABLE_SUPER_RES = True
REALESRGAN_DIR = os.path.join(os.path.dirname(__file__), "bin")
REALESRGAN_BIN = os.path.join(REALESRGAN_DIR, "realesrgan-ncnn-vulkan")
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"

# Only super-res photos that are smaller than the box the garment needs to
# fill on the canvas; upscaling something already sharp is where the GAN
# starts hallucinating fake detail instead of helping.
_SUPER_RES_TARGET_W = int(CANVAS_SIZE[0] * (1 - 2 * PADDING))
_SUPER_RES_TARGET_H = int(CANVAS_SIZE[1] * (1 - 2 * PADDING))
# -----------------------------------------------------------------------------


def needs_super_res(img: Image.Image) -> bool:
    """True if the source photo is smaller (by pixel count) than the box the
    garment must fill on the canvas."""
    return (img.width * img.height) < (_SUPER_RES_TARGET_W * _SUPER_RES_TARGET_H)


def cut_out(img: Image.Image, session) -> Image.Image:
    """Remove background, returning an RGBA image."""
    # alpha_matting=False: on low-contrast photos (dark garment/dark floor,
    # light garment/light floor) matting feathers uncertain regions into
    # transparency instead of giving a clean silhouette.
    return remove(img, session=session, alpha_matting=False).convert("RGBA")


def crop_to_content(img: Image.Image) -> Image.Image:
    """Trim fully transparent borders."""
    bbox = img.getchannel("A").point(lambda a: 255 if a > 10 else 0).getbbox()
    return img.crop(bbox) if bbox else img


def ensure_realesrgan() -> None:
    """Download+unzip the realesrgan-ncnn-vulkan binary and models if missing."""
    if os.path.exists(REALESRGAN_BIN):
        return
    print("Downloading realesrgan-ncnn-vulkan (first run only, ~150 MB)...")
    os.makedirs(REALESRGAN_DIR, exist_ok=True)
    zip_path = os.path.join(REALESRGAN_DIR, "realesrgan.zip")
    urllib.request.urlretrieve(REALESRGAN_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(REALESRGAN_DIR)
    os.remove(zip_path)
    os.chmod(REALESRGAN_BIN, 0o755)


def super_resolve(img: Image.Image, scale: int = 4) -> Image.Image:
    """4x upscale via Real-ESRGAN (runs as a subprocess; it ships as a binary,
    not a Python package, since its real pip package depends on the
    unmaintained basicsr, which doesn't build on current Python)."""
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.png")
        out_path = os.path.join(tmp, "out.png")
        img.save(in_path)
        subprocess.run(
            [REALESRGAN_BIN, "-i", in_path, "-o", out_path, "-n", "realesrgan-x4plus", "-s", str(scale)],
            cwd=REALESRGAN_DIR,  # the binary looks for ./models relative to its own cwd
            check=True,
            capture_output=True,
        )
        return Image.open(out_path).convert("RGBA").copy()


def place_on_canvas(img: Image.Image, size, padding) -> Image.Image:
    """Scale to fit inside the padded area and center on a transparent canvas."""
    cw, ch = size
    max_w, max_h = int(cw * (1 - 2 * padding)), int(ch * (1 - 2 * padding))
    scale = min(max_w / img.width, max_h / img.height)
    resized = img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.LANCZOS,
    )
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.paste(resized, ((cw - resized.width) // 2, (ch - resized.height) // 2), resized)
    return canvas


def light_touch_up(img: Image.Image) -> Image.Image:
    """Gentle catalog-style lift on RGB only; alpha untouched."""
    rgb, alpha = img.convert("RGB"), img.getchannel("A")
    rgb = ImageEnhance.Brightness(rgb).enhance(1.05)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.05)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def process_one(image_path: str, session) -> None:
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find {image_path}, skipping.")
        return

    result = cut_out(img, session)
    result = crop_to_content(result)
    if ENABLE_SUPER_RES and needs_super_res(img):
        result = super_resolve(result)
    result = place_on_canvas(result, CANVAS_SIZE, PADDING)
    if ENHANCE:
        result = light_touch_up(result)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    stem = re.sub(r"_\d{10}_.*$", "", stem)
    output_name = os.path.join(
        os.path.dirname(__file__),
        f"birefnet_{stem}_{datetime.now():%Y%m%d_%H%M%S}.png",
    )
    result.save(output_name)
    print(f"Success! Saved {output_name}")


def main():
    if not IMAGE_PATHS:
        print("Error: IMAGE_PATHS is empty. Add at least one image path!")
        return

    if ENABLE_SUPER_RES:
        ensure_realesrgan()

    print(f"Removing background with {MODEL_NAME}...")
    session = new_session(MODEL_NAME)  # reused across every image in the batch

    for image_path in IMAGE_PATHS:
        process_one(image_path, session)


if __name__ == "__main__":
    main()
