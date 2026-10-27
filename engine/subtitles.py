"""
Subtitle and transcript formatting.

Whisper already produces per-segment start/end times for every run, and the
translator produces one string per segment aligned to that same list. Both were
computed and then thrown away. This turns them into the files people actually
ask for after a dub.

No model runs here. This is formatting over data the pipeline already has.
"""

from __future__ import annotations

import os


def _clock(seconds: float, comma: bool) -> str:
    """HH:MM:SS,mmm for SRT, HH:MM:SS.mmm for WebVTT."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def cues(segments: list, texts: list | None) -> list:
    """
    Pair each segment's timing with the text to show. `texts` overrides the
    segment's own text when present, which is how the translated track reuses
    the source timings.
    """
    out = []
    for i, seg in enumerate(segments):
        body = (texts[i] if texts and i < len(texts) else seg.get("text", "")) or ""
        body = body.strip()
        if not body:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        # A zero or inverted duration produces a cue no player will show.
        if end <= start:
            end = start + 0.6
        out.append((start, end, body))
    return out


def srt(segments: list, texts: list | None = None) -> str:
    lines = []
    for n, (start, end, body) in enumerate(cues(segments, texts), 1):
        lines.append(str(n))
        lines.append(f"{_clock(start, True)} --> {_clock(end, True)}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)


def vtt(segments: list, texts: list | None = None) -> str:
    lines = ["WEBVTT", ""]
    for start, end, body in cues(segments, texts):
        lines.append(f"{_clock(start, False)} --> {_clock(end, False)}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)


def plain(segments: list, texts: list | None = None) -> str:
    """One line per spoken segment, which reads better than one long blob."""
    return "\n".join(body for _, _, body in cues(segments, texts)) + "\n"


# Every font that might hold the glyphs. The choice is made per cue, by
# checking which of these actually covers the characters in the text - a
# fixed order was the bug: Devanagari Sangam MN exists on every Mac and has
# no Telugu in it at all, so every Telugu subtitle rendered as a row of
# boxes. Per-script fonts come first so they win over the broad ones when
# coverage is equal; they are drawn for that script.
_FONT_CANDIDATES = [
    # Latin first: a Latin-only cue then lands in a Latin face rather than
    # whichever Indic font happens to carry ASCII as well.
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    # macOS, one per script
    "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Telugu Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Kannada Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Malayalam Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Bangla Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Gujarati Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Gurmukhi Sangam MN.ttc",
    # Oriya Sangam MN makes FreeType choke on conjuncts under raqm, so
    # the Noto face that ships beside it comes first.
    "/System/Library/Fonts/NotoSansOriya.ttc",
    "/System/Library/Fonts/Supplemental/Oriya Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Sinhala Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Myanmar Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf",
    "/System/Library/Fonts/Supplemental/Lao Sangam MN.ttf",
    "/System/Library/Fonts/GeezaPro.ttc",
    "/System/Library/Fonts/ThonburiUI.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    # Linux, one per script
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMalayalam-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansGurmukhi-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansOriya-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSinhala-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    # Windows: Nirmala carries every Indian script
    "C:/Windows/Fonts/Nirmala.ttc",
    # broad, for whatever is left
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/arialuni.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]

_CMAP_CACHE: dict = {}


def _cmap(path: str):
    """The codepoints a font file can draw, read once and kept."""
    if path in _CMAP_CACHE:
        return _CMAP_CACHE[path]
    try:
        import logging
        from fontTools.ttLib import TTFont
        # fontTools grumbles about padding bytes in some system fonts; it
        # is nothing anyone can act on.
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        cmap = set(TTFont(path, fontNumber=0, lazy=True).getBestCmap().keys())
    except Exception:                                        # noqa: BLE001
        cmap = set()
    _CMAP_CACHE[path] = cmap
    return cmap

# Beyond this many cues the filtergraph gets unreasonable, and a muxed track
# is the better trade.
MAX_BURN_CUES = 80


_FONT_CACHE: dict = {}


def _load(path: str, size: int):
    """An ImageFont, loaded once per file and size."""
    from PIL import ImageFont
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def _run_fonts(text: str, size: int) -> list:
    """
    Split a cue into runs, each in a font that actually holds it.

    A cue is rarely one script: a Telugu line naming a degree reads
    "నా పేరు ముడిత్ B.Tech", and no installed font holds both Telugu and
    Latin - drawing the whole line in the Telugu face turned the Latin into
    boxes. Each maximal run of characters one font covers is drawn with that
    font instead, so every letter lands somewhere real. Returns
    [(text, font, path)].
    """
    from PIL import Image, ImageDraw
    present = [p for p in _FONT_CANDIDATES if os.path.exists(p)]
    if not present:
        return []

    # Pick a font per character: keep the run in progress for anything that
    # is not a letter, so a space or a full stop does not split a word;
    # otherwise the first candidate that covers it, script fonts before the
    # broad ones.
    chosen = []
    for ch in text:
        cp = ord(ch)
        path = None
        if chosen and not ch.isalpha() and cp in _cmap(chosen[-1]):
            path = chosen[-1]
        if path is None:
            for p in present:
                if cp in _cmap(p):
                    path = p
                    break
        if path is None:
            path = present[0]
        chosen.append(path)

    # Group consecutive characters that share a font.
    runs = []
    for ch, path in zip(text, chosen, strict=True):
        if runs and runs[-1][2] == path:
            runs[-1][0] += ch
        else:
            runs.append([ch, None, path])

    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    return [(body, _pick_run_font(probe, body, size, path, present), path)
            for body, _, path in runs]


def _pick_run_font(probe, body: str, size: int, path: str, present: list):
    """
    The font for one run: the chosen one if it can draw the run, else the
    next candidate that covers its letters. A font can hold every letter and
    still fail to shape a conjunct (Oriya Sangam MN does), and that failure
    is a FreeType exception rather than a box.
    """
    need = {ord(c) for c in body if c.isalpha()}
    for p in [path] + [q for q in present if q != path]:
        if need and not need.issubset(_cmap(p)):
            continue
        try:
            font = _load(p, size)
            probe.textlength(body, font=font)
            return font
        except Exception as exc:                             # noqa: BLE001
            print(f"[Subtitles] {os.path.basename(p)} could not draw "
                  f"{body[:20]!r} ({exc}); trying the next font")
    return _load(path, size)


def coverage(text: str) -> set:
    """Every codepoint the fonts chosen for `text` can actually draw."""
    covered = set()
    for body, _, path in _run_fonts(text, 32):
        covered |= {ord(c) for c in body if ord(c) in _cmap(path)}
    return covered


def font_names(text: str) -> list:
    """The font files used to draw `text`, in order, for diagnostics."""
    seen = []
    for _, _, path in _run_fonts(text, 32):
        name = os.path.basename(path)
        if name not in seen:
            seen.append(name)
    return seen


def _font(size: int, text: str = ""):
    """The font for the dominant run of `text` (or the first installed one)."""
    if text:
        runs = _run_fonts(text, size)
        if runs:
            return max(runs, key=lambda r: sum(c.isalpha() for c in r[0]))[1]
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return _load(p, size)
    return None


def _width(draw, text: str, size: int) -> float:
    """Width of `text` across every font it is drawn in."""
    return sum(draw.textlength(body, font=font)
               for body, font, _ in _run_fonts(text, size))


def _pieces(draw, word: str, size: int, max_width: int) -> list:
    """
    A word no wider than the line, in as many pieces as that takes.

    Chinese, Japanese and Thai have no spaces to wrap on, so a whole cue
    arrives as one "word"; a URL read out loud does the same in any language.
    Without this they run off both edges of the picture.
    """
    if _width(draw, word, size) <= max_width:
        return [word]
    import unicodedata
    pieces, cur = [], ""
    for ch in word:
        # A vowel sign or tone mark stays with the letter it sits on.
        joins = unicodedata.category(ch) in ("Mn", "Mc")
        if cur and not joins and _width(draw, cur + ch, size) > max_width:
            pieces.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        pieces.append(cur)
    return pieces


def _wrap(draw, text: str, size: int, max_width: int) -> list:
    """Greedy wrap on spaces, measured across the fonts actually drawn."""
    words = [p for w in text.split() for p in _pieces(draw, w, size, max_width)]
    lines, line = [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if _width(draw, trial, size) <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines[:3]


def _draw_line(d, cx: int, y: int, line: str, size: int, fill,
               dx: int = 0, dy: int = 0) -> None:
    """Draw one line centred on `cx`, run by run, each script in its font."""
    runs = [(body, font, d.textlength(body, font=font))
            for body, font, _ in _run_fonts(line, size)]
    x = cx - sum(w for _, _, w in runs) / 2 + dx
    for body, font, w in runs:
        d.text((x, y + dy), body, font=font, fill=fill, anchor="lm")
        x += w


def _render_cue(text: str, video_w: int, font_size: int, path: str) -> bool:
    """
    Draw one caption to a transparent PNG.

    Pillow with raqm shapes Devanagari and Arabic properly - conjuncts,
    matras, joining forms - which is the part a naive per-glyph renderer gets
    wrong.
    """
    from PIL import Image, ImageDraw

    margin = max(24, video_w // 16)
    max_width = video_w - 2 * margin
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    lines = _wrap(probe, text, font_size, max_width)

    line_h = int(font_size * 1.45)
    img = Image.new("RGBA", (video_w, line_h * len(lines) + 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = 8 + line_h * i + line_h // 2
        # An outline, so white text stays readable over a white shirt.
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if dx or dy:
                    _draw_line(d, video_w // 2, y, line, font_size,
                               (0, 0, 0, 220), dx=dx, dy=dy)
        _draw_line(d, video_w // 2, y, line, font_size, (255, 255, 255, 255))
    img.save(path)
    return True


def _video_size(path: str):
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30).stdout.strip()
        w, h = (int(x) for x in out.split(",")[:2])
        return w, h
    except Exception:                                        # noqa: BLE001
        return 0, 0


def can_burn() -> bool:
    """
    Whether captions can be drawn onto the picture.

    Not ffmpeg's `subtitles` filter, which needs libass - Homebrew's ffmpeg
    does not have it, so on macOS that path is simply unavailable. This
    renders the text with Pillow and composites with `overlay`, which every
    build has.
    """
    try:
        from PIL import features
        return bool(features.check("freetype2")) and _font(28) is not None
    except Exception:                                        # noqa: BLE001
        return False


def attach(video_path: str, cues: list, out_path: str, workdir: str,
           srt_text: str = "", language: str = "und",
           crf: int = 18, preset: str = "medium", font_size: int = 0) -> str:
    """
    Put subtitles on a video, timed to the dialogue.

    `cues` is [(start, end, text)]. Drawn onto the picture when the text can be
    rendered, so they are visible anywhere without being switched on;
    otherwise muxed as a track, which needs nothing and costs a second.

    Returns "burned" or "muxed" so the caller can say which happened.
    """
    import subprocess

    srt_path = os.path.join(workdir, "subs.srt")
    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write(srt_text)

    width, height = _video_size(video_path)
    usable = [c for c in cues if (c[2] or "").strip()][:MAX_BURN_CUES]

    if can_burn() and width and usable:
        size = font_size or max(18, min(46, int(height * 0.055)))
        inputs, filters, label = [], [], "[0:v]"
        made = 0
        for i, (start, end, text) in enumerate(usable):
            png = os.path.join(workdir, f"cue{i}.png")
            if not _render_cue(text.strip(), width, size, png):
                continue
            inputs += ["-i", png]
            made += 1
            nxt = f"[v{made}]"
            filters.append(
                f"{label}[{made}:v]overlay=(W-w)/2:H-h-{max(16, height // 22)}"
                f":enable='between(t,{start:.3f},{end:.3f})'{nxt}")
            label = nxt

        if made:
            cmd = (["ffmpeg", "-y", "-i", os.path.abspath(video_path)] + inputs +
                   ["-filter_complex", ";".join(filters),
                    "-map", label, "-map", "0:a?",
                    "-c:a", "copy", "-c:v", "libx264",
                    "-crf", str(crf), "-preset", preset,
                    os.path.abspath(out_path)])
            proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
            if proc.returncode == 0:
                return "burned"
            tail = " ".join((proc.stderr or "").strip().splitlines()[-3:])
            print(f"[Subtitles] Could not draw them on ({tail}); muxing instead")

    cmd = ["ffmpeg", "-y", "-i", os.path.abspath(video_path),
           "-i", "subs.srt", "-c", "copy", "-c:s", "mov_text",
           "-metadata:s:s:0", f"language={language}",
           os.path.abspath(out_path)]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = " ".join((proc.stderr or "").strip().splitlines()[-3:])
        raise RuntimeError(f"Could not add the subtitles: {tail}")
    return "muxed"
