"""Draw the app icon (assets/icon.ico) so the repo needs no binary art."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def make(size=256) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(size // 2):
        t = i / (size / 2)
        col = (int(108 + 60 * t), int(92 + 40 * t), int(231 - 30 * t), 255)
        d.rounded_rectangle([i * 0.15, i * 0.15, size - i * 0.15, size - i * 0.15],
                            radius=int(size * 0.22), outline=col)
    d.rounded_rectangle([size * 0.06, size * 0.06, size * 0.94, size * 0.94], radius=int(size * 0.2),
                        fill=(108, 92, 231, 255))
    try:
        font = ImageFont.truetype("segoeuib.ttf", int(size * 0.42))
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(size * 0.42))
        except OSError:
            font = ImageFont.load_default()
    d.text((size / 2, size * 0.47), "UE", font=font, fill="white", anchor="mm")
    d.polygon([(size * 0.62, size * 0.70), (size * 0.84, size * 0.70), (size * 0.84, size * 0.88)],
              fill=(255, 255, 255, 230))
    return img


if __name__ == "__main__":
    out = ROOT / "assets" / "icon.ico"
    out.parent.mkdir(exist_ok=True)
    make().save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    make().save(ROOT / "assets" / "icon.png")
    print("wrote", out)
