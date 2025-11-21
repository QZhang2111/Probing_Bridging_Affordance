#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_PATH = Path(
    "/home/li325/qing_workspace/exps/affordance-experiments/Section2_exp/data/mug_2.png"
).resolve()
OUT_PATH = IMAGE_PATH.with_name(f"{IMAGE_PATH.stem}_with_point.png")

# X, Y pixel coordinates (column, row) to highlight
POINT = (61, 178)

# Appearance for the point marker
MARK_COLOR = (255, 0, 0)  # bright red
CIRCLE_RADIUS = 8
CIRCLE_WIDTH = 2
CROSSHAIR_LENGTH = 12
CROSSHAIR_WIDTH = 2


def mark_point(img_path: Path, point: tuple[int, int], out_path: Path) -> None:
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    x, y = point
    r = CIRCLE_RADIUS
    circle_bbox = (x - r, y - r, x + r, y + r)
    draw.ellipse(circle_bbox, outline=MARK_COLOR, width=CIRCLE_WIDTH)

    half_len = CROSSHAIR_LENGTH // 2
    draw.line((x - half_len, y, x + half_len, y), fill=MARK_COLOR, width=CROSSHAIR_WIDTH)
    draw.line((x, y - half_len, x, y + half_len), fill=MARK_COLOR, width=CROSSHAIR_WIDTH)

    image.save(out_path)


def main() -> None:
    mark_point(IMAGE_PATH, POINT, OUT_PATH)
    print(f"Saved marked image to {OUT_PATH}")


if __name__ == "__main__":
    main()
