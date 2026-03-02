from __future__ import annotations

from pathlib import Path
import struct
import zlib


def _render_with_matplotlib(out: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except Exception:
        return False

    def add_box(ax, text, x, y, w=0.22, h=0.1):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.01", linewidth=1.5, edgecolor="#1f2937", facecolor="#f3f4f6")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
        return {"x": x, "y": y, "w": w, "h": h}

    def connect(ax, a, b):
        start = (a["x"] + a["w"], a["y"] + a["h"] / 2)
        end = (b["x"], b["y"] + b["h"] / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5, color="#111827"))

    def down(ax, a, b):
        start = (a["x"] + a["w"] / 2, a["y"])
        end = (b["x"] + b["w"] / 2, b["y"] + b["h"])
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5, color="#111827"))

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    agent = add_box(ax, "AI Agent / App", 0.03, 0.78)
    sdk = add_box(ax, "HippoCortex SDK", 0.30, 0.78)
    router = add_box(ax, "Memory Router\npolicy + budgets", 0.57, 0.78)
    working = add_box(ax, "Working Memory\ncontext window manager", 0.03, 0.52)
    hippo = add_box(ax, "Hippocampus Layer\nEpisodic Store", 0.30, 0.52)
    cons = add_box(ax, "Consolidation Engine\nreplay + distill", 0.57, 0.52)
    adapters = add_box(ax, "Storage Adapters\nSQLite/Postgres\nVector index + Optional graph", 0.03, 0.26)
    cortex = add_box(ax, "Cortex Layer\nSemantic Index", 0.30, 0.26, w=0.49)
    llm = add_box(ax, "LLM Provider\nOpenAI / Anthropic / local", 0.30, 0.04)
    rag = add_box(ax, "Tools / RAG\nDocs, Web, Codebase", 0.60, 0.04, w=0.19)

    connect(ax, agent, sdk)
    connect(ax, sdk, router)
    connect(ax, working, hippo)
    connect(ax, hippo, cons)
    down(ax, agent, working)
    down(ax, hippo, cortex)
    down(ax, cons, cortex)
    connect(ax, adapters, cortex)
    down(ax, adapters, llm)
    down(ax, cortex, llm)
    connect(ax, llm, rag)

    ax.set_title("HippoCortex: Dual-Memory OS Architecture (Hippocampus ↔ Neocortex)", fontsize=18, weight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    return True


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(pixels[start : start + stride])
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = zlib.compress(bytes(raw), level=9)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", data) + _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _fill(pixels: bytearray, w: int, h: int, color: tuple[int, int, int]) -> None:
    for y in range(h):
        row = y * w * 3
        for x in range(w):
            i = row + x * 3
            pixels[i : i + 3] = bytes(color)


def _rect(pixels: bytearray, w: int, x0: int, y0: int, x1: int, y1: int, fill: tuple[int, int, int], border=(30, 30, 30)) -> None:
    for y in range(y0, y1):
        for x in range(x0, x1):
            i = (y * w + x) * 3
            pixels[i : i + 3] = bytes(border if x in (x0, x1 - 1) or y in (y0, y1 - 1) else fill)


def _line(pixels: bytearray, w: int, x0: int, y0: int, x1: int, y1: int, color=(20, 20, 20)) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        i = (y0 * w + x0) * 3
        pixels[i : i + 3] = bytes(color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _arrow(pixels: bytearray, w: int, x0: int, y0: int, x1: int, y1: int) -> None:
    _line(pixels, w, x0, y0, x1, y1)
    _line(pixels, w, x1, y1, x1 - 8, y1 - 4)
    _line(pixels, w, x1, y1, x1 - 8, y1 + 4)


def _render_stdlib_fallback(out: Path) -> None:
    w, h = 1600, 900
    px = bytearray(w * h * 3)
    _fill(px, w, h, (245, 246, 248))
    _rect(px, w, 80, 80, 520, 220, (230, 230, 232))
    _rect(px, w, 580, 80, 1020, 220, (230, 230, 232))
    _rect(px, w, 1080, 80, 1520, 220, (230, 230, 232))
    _rect(px, w, 80, 310, 520, 510, (230, 230, 232))
    _rect(px, w, 580, 310, 1020, 510, (230, 230, 232))
    _rect(px, w, 1080, 310, 1520, 510, (230, 230, 232))
    _rect(px, w, 80, 560, 520, 760, (230, 230, 232))
    _rect(px, w, 580, 560, 1520, 760, (230, 230, 232))
    _rect(px, w, 580, 790, 1020, 880, (230, 230, 232))
    _rect(px, w, 1080, 790, 1520, 880, (230, 230, 232))
    _arrow(px, w, 520, 150, 580, 150)
    _arrow(px, w, 1020, 150, 1080, 150)
    _arrow(px, w, 300, 220, 300, 310)
    _arrow(px, w, 520, 410, 580, 410)
    _arrow(px, w, 1020, 410, 1080, 410)
    _arrow(px, w, 800, 510, 800, 560)
    _arrow(px, w, 1300, 510, 1300, 560)
    _arrow(px, w, 520, 660, 580, 660)
    _arrow(px, w, 300, 760, 300, 790)
    _arrow(px, w, 1050, 835, 1080, 835)
    _write_png(out, w, h, px)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "assets" / "hippocortex_architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    if _render_with_matplotlib(out):
        print(f"Wrote {out} with matplotlib")
    else:
        _render_stdlib_fallback(out)
        print(f"Wrote {out} with stdlib fallback")


if __name__ == "__main__":
    main()
