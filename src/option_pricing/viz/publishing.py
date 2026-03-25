from __future__ import annotations

import base64
import re
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from html import escape
from pathlib import Path

LIGHT_THEME = "light"
DARK_THEME = "dark"
PUBLISHING_THEMES = (LIGHT_THEME, DARK_THEME)
SVG_TEXT_FONT_STACK = "DejaVu Sans, Liberation Sans, Arial, sans-serif"
_DATA_URI_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}
_SVG_TEXT_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AssetVariants:
    base: Path
    light: Path
    dark: Path

    def path_for(self, theme: str) -> Path:
        normalized = str(theme).strip().lower()
        if normalized == DARK_THEME:
            return self.dark
        if normalized == LIGHT_THEME:
            return self.light
        raise ValueError(f"Unsupported publishing theme: {theme}")


@dataclass(frozen=True)
class SvgTextStyle:
    font_family: str = SVG_TEXT_FONT_STACK
    font_size: float = 16.0
    font_weight: str | int = "400"
    fill: str = "#000000"
    line_height: float | None = None
    letter_spacing: float = 0.0

    @property
    def resolved_line_height(self) -> float:
        if self.line_height is not None:
            return float(self.line_height)
        return float(self.font_size) * 1.25


@dataclass(frozen=True)
class SvgTextLayout:
    style: SvgTextStyle
    lines: tuple[str, ...]
    max_width: float
    measured_width: float
    height: float
    fits_width: bool
    fits_height: bool
    max_height: float | None = None

    @property
    def fits(self) -> bool:
        return self.fits_width and self.fits_height

    def to_svg(
        self,
        *,
        x: float,
        y: float,
        extra_attrs: Mapping[str, str | float | int | None] | None = None,
    ) -> str:
        if not self.lines:
            return ""

        attrs: dict[str, str | float | int | None] = {
            "x": x,
            "y": y,
            "font-family": self.style.font_family,
            "font-size": self.style.font_size,
            "font-weight": self.style.font_weight,
            "fill": self.style.fill,
        }
        if self.style.letter_spacing:
            attrs["letter-spacing"] = self.style.letter_spacing
        if extra_attrs:
            attrs.update(extra_attrs)

        parts = [f"<text {_svg_attr_string(attrs)}>"]
        x_value = _svg_attr_value(x)
        for index, line in enumerate(self.lines):
            dy = 0 if index == 0 else self.style.resolved_line_height
            line_text = escape(line)
            parts.append(
                f'<tspan x="{x_value}" dy="{_svg_attr_value(dy)}">{line_text}</tspan>'
            )
        parts.append("</text>")
        return "".join(parts)


@dataclass(frozen=True)
class SvgRasterLayout:
    source_width: float
    source_height: float
    slot_x: float
    slot_y: float
    slot_width: float
    slot_height: float
    image_x: float
    image_y: float
    image_width: float
    image_height: float


@dataclass(frozen=True)
class SvgRasterBlock:
    layout: SvgRasterLayout
    clip_path: str
    svg: str


def themed_asset_paths(out_path: str | Path) -> AssetVariants:
    path = Path(out_path)
    suffix = path.suffix
    stem = path.stem
    return AssetVariants(
        base=path,
        light=path.with_name(f"{stem}.light{suffix}"),
        dark=path.with_name(f"{stem}.dark{suffix}"),
    )


def copy_light_variant(variants: AssetVariants) -> Path:
    variants.base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(variants.light, variants.base)
    return variants.base


def file_to_data_uri(path: str | Path) -> str:
    asset_path = Path(path)
    mime_type = _DATA_URI_MIME_TYPES.get(asset_path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported asset type for data URI embedding: {asset_path}")

    encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid PNG signature for {path}")
        handle.read(4)  # IHDR payload length
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"Missing IHDR chunk in {path}")
        width = int.from_bytes(handle.read(4), "big")
        height = int.from_bytes(handle.read(4), "big")
    return width, height


def _read_jpeg_dimensions(path: Path) -> tuple[int, int]:
    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    standalone_markers = {0x01, *range(0xD0, 0xD8)}

    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError(f"Invalid JPEG signature for {path}")

        while True:
            marker_start = handle.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue

            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker:
                break

            marker_value = marker[0]
            if marker_value in standalone_markers:
                continue
            if marker_value == 0xD9:
                break

            segment_length_bytes = handle.read(2)
            if len(segment_length_bytes) != 2:
                break
            segment_length = int.from_bytes(segment_length_bytes, "big")
            if segment_length < 2:
                raise ValueError(f"Invalid JPEG segment length in {path}")

            if marker_value in start_of_frame_markers:
                handle.read(1)  # sample precision
                height = int.from_bytes(handle.read(2), "big")
                width = int.from_bytes(handle.read(2), "big")
                return width, height

            handle.seek(segment_length - 2, 1)

    raise ValueError(f"Could not find JPEG dimensions for {path}")


def read_image_dimensions(path: str | Path) -> tuple[int, int]:
    asset_path = Path(path)
    suffix = asset_path.suffix.lower()
    if suffix == ".png":
        return _read_png_dimensions(asset_path)
    if suffix in {".jpg", ".jpeg"}:
        return _read_jpeg_dimensions(asset_path)
    raise ValueError(f"Unsupported image type for dimension lookup: {asset_path}")


def layout_svg_contained_raster(
    *,
    source_width: float,
    source_height: float,
    slot_x: float,
    slot_y: float,
    slot_width: float,
    slot_height: float,
) -> SvgRasterLayout:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Contained raster layout requires positive source dimensions")
    if slot_width <= 0 or slot_height <= 0:
        raise ValueError("Contained raster layout requires positive slot dimensions")

    scale = min(
        float(slot_width) / float(source_width),
        float(slot_height) / float(source_height),
    )
    image_width = float(source_width) * scale
    image_height = float(source_height) * scale
    image_x = float(slot_x) + (float(slot_width) - image_width) / 2.0
    image_y = float(slot_y) + (float(slot_height) - image_height) / 2.0

    return SvgRasterLayout(
        source_width=float(source_width),
        source_height=float(source_height),
        slot_x=float(slot_x),
        slot_y=float(slot_y),
        slot_width=float(slot_width),
        slot_height=float(slot_height),
        image_x=image_x,
        image_y=image_y,
        image_width=image_width,
        image_height=image_height,
    )


def render_svg_contained_raster(
    *,
    block_id: str,
    image_path: str | Path,
    slot_x: float,
    slot_y: float,
    slot_width: float,
    slot_height: float,
    frame_radius: float,
    frame_fill: str,
    clip_id: str | None = None,
    source_label: str | None = None,
    frame_extra_attrs: Mapping[str, str | float | int | None] | None = None,
    image_extra_attrs: Mapping[str, str | float | int | None] | None = None,
) -> SvgRasterBlock:
    asset_path = Path(image_path)
    source_width, source_height = read_image_dimensions(asset_path)
    layout = layout_svg_contained_raster(
        source_width=source_width,
        source_height=source_height,
        slot_x=slot_x,
        slot_y=slot_y,
        slot_width=slot_width,
        slot_height=slot_height,
    )

    resolved_clip_id = clip_id or f"{block_id}Clip"
    clip_radius = min(
        float(frame_radius),
        layout.image_width / 2.0,
        layout.image_height / 2.0,
    )
    clip_path = (
        f'    <clipPath id="{resolved_clip_id}">\n'
        f'      <rect x="{layout.image_x:g}" y="{layout.image_y:g}" '
        f'width="{layout.image_width:g}" height="{layout.image_height:g}" '
        f'rx="{clip_radius:g}" ry="{clip_radius:g}" />\n'
        "    </clipPath>"
    )

    frame_attrs: dict[str, str | float | int | None] = {
        "x": layout.slot_x,
        "y": layout.slot_y,
        "width": layout.slot_width,
        "height": layout.slot_height,
        "rx": frame_radius,
        "ry": frame_radius,
        "fill": frame_fill,
    }
    if frame_extra_attrs:
        frame_attrs.update(frame_extra_attrs)

    resolved_source_label = source_label or asset_path.name
    image_attrs: dict[str, str | float | int | None] = {
        "id": block_id,
        "x": layout.image_x,
        "y": layout.image_y,
        "width": layout.image_width,
        "height": layout.image_height,
        "href": file_to_data_uri(asset_path),
        "preserveAspectRatio": "none",
        "clip-path": f"url(#{resolved_clip_id})",
        "data-fit": "contain",
        "data-slot-x": layout.slot_x,
        "data-slot-y": layout.slot_y,
        "data-slot-width": layout.slot_width,
        "data-slot-height": layout.slot_height,
        "data-image-x": layout.image_x,
        "data-image-y": layout.image_y,
        "data-image-width": layout.image_width,
        "data-image-height": layout.image_height,
        "data-source-width": layout.source_width,
        "data-source-height": layout.source_height,
        "data-source-path": resolved_source_label,
    }
    if image_extra_attrs:
        image_attrs.update(image_extra_attrs)

    frame_svg = f"  <rect {_svg_attr_string(frame_attrs)} />"
    image_svg = f"  <image {_svg_attr_string(image_attrs)} />"
    return SvgRasterBlock(
        layout=layout,
        clip_path=clip_path,
        svg="\n".join((frame_svg, image_svg)),
    )


def _numeric_font_weight(font_weight: str | int) -> int:
    if isinstance(font_weight, int):
        return font_weight
    normalized = str(font_weight).strip().lower()
    if normalized == "normal":
        return 400
    if normalized == "bold":
        return 700
    try:
        return int(float(normalized))
    except ValueError:
        return 400


def _svg_attr_value(value: str | float | int) -> str:
    if isinstance(value, str):
        return escape(value, quote=True)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):g}"


def _svg_attr_string(attrs: Mapping[str, str | float | int | None]) -> str:
    return " ".join(
        f'{name}="{_svg_attr_value(value)}"'
        for name, value in attrs.items()
        if value is not None
    )


def estimate_svg_text_width(
    text: str,
    *,
    font_size: float,
    font_weight: str | int = "400",
    letter_spacing: float = 0.0,
) -> float:
    if not text:
        return 0.0

    total = 0.0
    for char in text:
        if char.isspace():
            factor = 0.34
        elif char in "ilIjt'`.,:;!|":
            factor = 0.31
        elif char in "fr()[]{}-/\\":
            factor = 0.38
        elif char in "mwMW@%&QG#":
            factor = 0.82
        elif char.isdigit():
            factor = 0.56
        elif char.isupper():
            factor = 0.63
        else:
            factor = 0.55
        total += factor * float(font_size)

    if len(text) > 1 and letter_spacing:
        total += (len(text) - 1) * float(letter_spacing)

    weight = _numeric_font_weight(font_weight)
    if weight >= 700:
        total *= 1.04
    elif weight >= 600:
        total *= 1.02
    return total + 0.06 * float(font_size)


def _normalize_svg_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for raw_line in str(text).splitlines():
        paragraph = _SVG_TEXT_WHITESPACE_RE.sub(" ", raw_line).strip()
        if paragraph:
            paragraphs.append(paragraph)
        elif paragraphs and paragraphs[-1] != "":
            paragraphs.append("")
    return paragraphs


def _split_svg_word_to_width(
    word: str,
    *,
    max_width: float,
    style: SvgTextStyle,
) -> list[str]:
    if not word:
        return [word]
    if (
        estimate_svg_text_width(
            word,
            font_size=style.font_size,
            font_weight=style.font_weight,
            letter_spacing=style.letter_spacing,
        )
        <= max_width
    ):
        return [word]

    parts: list[str] = []
    current = ""
    for char in word:
        candidate = f"{current}{char}"
        if current and (
            estimate_svg_text_width(
                candidate,
                font_size=style.font_size,
                font_weight=style.font_weight,
                letter_spacing=style.letter_spacing,
            )
            > max_width
        ):
            parts.append(current)
            current = char
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [word]


def wrap_svg_text(
    text: str,
    *,
    max_width: float,
    style: SvgTextStyle,
    max_height: float | None = None,
) -> SvgTextLayout:
    if max_width <= 0:
        raise ValueError("SVG text wrapping requires a positive width budget")

    lines: list[str] = []
    for paragraph in _normalize_svg_paragraphs(text):
        if paragraph == "":
            if lines and lines[-1] != "":
                lines.append("")
            continue

        current = ""
        for word in paragraph.split(" "):
            for token in _split_svg_word_to_width(
                word,
                max_width=max_width,
                style=style,
            ):
                candidate = token if not current else f"{current} {token}"
                if current and (
                    estimate_svg_text_width(
                        candidate,
                        font_size=style.font_size,
                        font_weight=style.font_weight,
                        letter_spacing=style.letter_spacing,
                    )
                    > max_width
                ):
                    lines.append(current)
                    current = token
                else:
                    current = candidate
        if current or not lines:
            lines.append(current)

    widths = [
        estimate_svg_text_width(
            line,
            font_size=style.font_size,
            font_weight=style.font_weight,
            letter_spacing=style.letter_spacing,
        )
        for line in lines
    ]
    measured_width = max(widths, default=0.0)
    height = style.resolved_line_height * len(lines) if lines else 0.0
    fits_width = all(width <= max_width + 1e-6 for width in widths)
    fits_height = max_height is None or height <= max_height + 1e-6

    return SvgTextLayout(
        style=style,
        lines=tuple(lines),
        max_width=float(max_width),
        measured_width=measured_width,
        height=height,
        fits_width=fits_width,
        fits_height=fits_height,
        max_height=float(max_height) if max_height is not None else None,
    )


def render_svg_text_block(
    *,
    block_id: str,
    text: str,
    x: float,
    y: float,
    max_width: float,
    max_height: float | None,
    style: SvgTextStyle,
    overflow_label: str | None = None,
    extra_attrs: Mapping[str, str | float | int | None] | None = None,
) -> str:
    layout = wrap_svg_text(
        text,
        max_width=max_width,
        max_height=max_height,
        style=style,
    )
    if not layout.fits:
        limits: list[str] = []
        if not layout.fits_width:
            limits.append(f"width={max_width:g}")
        if max_height is not None and not layout.fits_height:
            limits.append(f"height={max_height:g}")
        joined_limits = ", ".join(limits) or "unknown constraints"
        label = overflow_label or block_id
        raise ValueError(f"{label} exceeded its layout budget ({joined_limits})")

    attrs: dict[str, str | float | int | None] = {
        "id": block_id,
        "data-max-width": max_width,
        "data-max-height": max_height,
        "data-line-height": style.resolved_line_height,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return layout.to_svg(
        x=x,
        y=y,
        extra_attrs=attrs,
    )


def publishing_palette(theme: str = LIGHT_THEME) -> dict[str, str]:
    normalized = str(theme).strip().lower()
    if normalized == DARK_THEME:
        return {
            "figure_face": "#0F172A",
            "axes_face": "#0F172A",
            "text": "#E5EDF7",
            "muted_text": "#A8B5C7",
            "grid": "#7C8BA0",
            "spine": "#94A3B8",
            "legend_face": "#162033",
            "legend_edge": "#314056",
            "reference": "#E5EDF7",
        }
    if normalized == LIGHT_THEME:
        return {
            "figure_face": "#FFFFFF",
            "axes_face": "#FFFFFF",
            "text": "#0B1F33",
            "muted_text": "#5A7185",
            "grid": "#A7B8C8",
            "spine": "#A7B8C8",
            "legend_face": "#FFFFFF",
            "legend_edge": "#D7E0EA",
            "reference": "#243447",
        }
    raise ValueError(f"Unsupported publishing theme: {theme}")


def style_colorbar(colorbar, *, theme: str = LIGHT_THEME) -> None:
    palette = publishing_palette(theme)
    colorbar.outline.set_edgecolor(palette["spine"])
    colorbar.ax.tick_params(colors=palette["text"])
    colorbar.ax.yaxis.label.set_color(palette["text"])


def load_pyplot():
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Publishing visuals requires matplotlib. Install it with: pip install -e '.[plot]'"
        ) from e
    return plt


@contextmanager
def publishing_style(theme: str = LIGHT_THEME):
    plt = load_pyplot()
    palette = publishing_palette(theme)
    with plt.rc_context(
        {
            "figure.dpi": 120,
            "figure.facecolor": palette["figure_face"],
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.facecolor": palette["axes_face"],
            "axes.edgecolor": palette["spine"],
            "axes.labelcolor": palette["text"],
            "axes.titlecolor": palette["text"],
            "grid.alpha": 0.18,
            "grid.color": palette["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.titlepad": 8.0,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.color": palette["text"],
            "xtick.labelsize": 9,
            "ytick.color": palette["text"],
            "ytick.labelsize": 9,
            "text.color": palette["text"],
            "legend.facecolor": palette["legend_face"],
            "legend.edgecolor": palette["legend_edge"],
            "font.family": "DejaVu Sans",
            "savefig.facecolor": palette["figure_face"],
            "savefig.edgecolor": palette["figure_face"],
        }
    ):
        yield plt


def save_figure(fig, out_path: str | Path, *, dpi: int) -> Path:
    plt = load_pyplot()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return path
