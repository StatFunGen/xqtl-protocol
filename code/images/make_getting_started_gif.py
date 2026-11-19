"""Draw the Environment Setup getting-started animation.

Four cards in a 2 x 2 grid, one per step on the page. Each card has a small
diagram of what the step does and, under it, the real command for the step.
The steps play in order (the diagram builds up, the command types out, the card
is ticked), then the finished state holds. The 2 x 2 layout keeps the text
readable when the page is narrow.

Run from the repository root:  python code/images/make_getting_started_gif.py
Writes code/images/xqtl_getting_started.gif. Needs Pillow and matplotlib (fonts).
"""
from PIL import Image, ImageDraw, ImageFont
import os, matplotlib

FD = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
F_B = os.path.join(FD, 'DejaVuSans-Bold.ttf')
F_R = os.path.join(FD, 'DejaVuSans.ttf')
F_M = os.path.join(FD, 'DejaVuSansMono.ttf')

# Site colors (website/fungen.css): logo blue, dark ink, borders.
BG = (255, 255, 255); BORDER = (215, 222, 229)
INK = (25, 53, 74); MUTED = (95, 110, 124); BLUE = (61, 98, 137)
BSOFT = (214, 226, 238); PALE = (241, 245, 249); IDLE = (232, 237, 242)
TERM = (25, 38, 52); TERM_TX = (214, 225, 234); PROMPT = (143, 176, 208)
OK = (46, 125, 70); OK_SOFT = (222, 240, 227); LINE = (160, 174, 188)

W, H = 1040, 700
PAD, GAP = 24, 20
CW = (W - 2 * PAD - GAP) // 2
CH = (H - 2 * PAD - GAP) // 2
POS = [(PAD, PAD), (PAD + CW + GAP, PAD), (PAD, PAD + CH + GAP), (PAD + CW + GAP, PAD + CH + GAP)]

TITLES = ['Install the software', 'Clone the protocol', 'Example data', 'Run a pipeline']
CMDS = ['bash pixi-setup.sh', 'git clone .../xqtl-protocol.git',
        'ls tests/fixtures/', 'sos run pipeline/<name>.ipynb ...']

f_num = ImageFont.truetype(F_B, 22)
f_title = ImageFont.truetype(F_B, 26)
f_mono = ImageFont.truetype(F_M, 19)
f_lab = ImageFont.truetype(F_B, 15)
f_small = ImageFont.truetype(F_R, 14)
f_file = ImageFont.truetype(F_M, 14)


def ramp(p, a, b):
    """0 before a, 1 after b, linear in between."""
    return 0.0 if p <= a else 1.0 if p >= b else (p - a) / (b - a)


def ctext(d, x, y, t, f, c):
    w = d.textbbox((0, 0), t, font=f)[2]
    d.text((x - w / 2, y), t, font=f, fill=c)


def arrow(d, x1, y1, x2, y2, c, p=1.0, width=3):
    """Arrow from (x1,y1) towards (x2,y2), drawn to fraction p."""
    if p <= 0:
        return
    xe, ye = x1 + (x2 - x1) * p, y1 + (y2 - y1) * p
    d.line([(x1, y1), (xe, ye)], fill=c, width=width)
    if p >= 1:
        if x2 != x1:
            s = 1 if x2 > x1 else -1
            d.polygon([(x2, y2), (x2 - 9 * s, y2 - 6), (x2 - 9 * s, y2 + 6)], fill=c)
        else:
            s = 1 if y2 > y1 else -1
            d.polygon([(x2, y2), (x2 - 6, y2 - 9 * s), (x2 + 6, y2 - 9 * s)], fill=c)


def tick(d, cx, cy, r, fill=OK, mark=(255, 255, 255)):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    d.line([(cx - r * 0.45, cy), (cx - r * 0.1, cy + r * 0.38), (cx + r * 0.5, cy - r * 0.35)], fill=mark, width=max(2, int(r / 4)))


def folder(d, x, y, w, h, fill, outline):
    d.polygon([(x, y + 10), (x + w * 0.38, y + 10), (x + w * 0.46, y), (x + w, y), (x + w, y + h), (x, y + h)],
              fill=fill, outline=outline)


# ---- diagrams: each draws into the box (x, y, w, h) at progress p in 0..1 ----
def d_install(d, x, y, w, h, p):
    # installer -> environment box that fills with packages
    ix, iy = x + 18, y + h // 2 - 34
    d.rounded_rectangle([ix, iy, ix + 92, iy + 68], 8, fill=PALE, outline=LINE, width=2)
    ctext(d, ix + 46, iy + 12, 'pixi', f_lab, INK)
    ctext(d, ix + 46, iy + 36, 'installer', f_small, MUTED)
    arrow(d, ix + 100, y + h // 2, ix + 150, y + h // 2, BLUE, ramp(p, 0.05, 0.2))
    ex, ey, ew, eh = ix + 160, y + 10, w - 196, h - 20
    d.rounded_rectangle([ex, ey, ex + ew, ey + eh], 10, fill=(250, 252, 254), outline=(BLUE if p > 0.2 else LINE), width=2)
    d.text((ex + 12, ey + 8), 'pixi environment', font=f_lab, fill=INK)
    pk = ['R', 'Python', 'SoS', 'plink', 'samtools', 'tensorqtl']
    cw, chh = (ew - 36) // 3, 30
    for k, name in enumerate(pk):
        on = ramp(p, 0.22 + k * 0.07, 0.3 + k * 0.07)
        cx, cy = ex + 12 + (k % 3) * (cw + 6), ey + 38 + (k // 3) * (chh + 10)
        d.rounded_rectangle([cx, cy, cx + cw, cy + chh], 6, fill=(BSOFT if on >= 1 else IDLE))
        if on >= 1:
            ctext(d, cx + cw / 2, cy + 7, name, f_small, INK)
    if p >= 0.8:
        d.text((ex + 12, ey + eh - 26), 'choose "full" (~35 GB)', font=f_small, fill=MUTED)


def d_clone(d, x, y, w, h, p):
    # GitHub repository -> local folder tree
    gx, gy = x + 18, y + h // 2 - 34
    d.rounded_rectangle([gx, gy, gx + 98, gy + 68], 8, fill=PALE, outline=LINE, width=2)
    ctext(d, gx + 49, gy + 12, 'GitHub', f_lab, INK)
    ctext(d, gx + 49, gy + 36, 'StatFunGen', f_small, MUTED)
    arrow(d, gx + 106, y + h // 2, gx + 140, y + h // 2, BLUE, ramp(p, 0.05, 0.2))
    tx, ty = gx + 150, y + 8
    folder(d, tx, ty, 34, 26, BSOFT, BLUE)
    d.text((tx + 44, ty + 4), 'xqtl-protocol/', font=f_file, fill=INK)
    items = [('pipeline/', 'run these'), ('code/', 'source'), ('tests/fixtures/', 'examples')]
    for k, (name, note) in enumerate(items):
        on = ramp(p, 0.25 + k * 0.15, 0.35 + k * 0.15)
        if on <= 0:
            continue
        ry = ty + 44 + k * 34
        d.line([(tx + 16, ty + 28), (tx + 16, ry + 11)], fill=LINE, width=2)
        d.line([(tx + 16, ry + 11), (tx + 34, ry + 11)], fill=LINE, width=2)
        folder(d, tx + 40, ry, 26, 21, (OK_SOFT if k == 2 else BSOFT), (OK if k == 2 else BLUE))
        d.text((tx + 74, ry + 2), name, font=f_file, fill=INK)
        nx = tx + 74 + d.textbbox((0, 0), name, font=f_file)[2] + 10
        d.text((nx, ry + 2), note, font=f_small, fill=MUTED)


def d_data(d, x, y, w, h, p):
    # chromosome 22 -> small example files, one folder per module
    cx, cy = x + 18, y + 18
    d.text((cx, cy), 'chr22 subset', font=f_lab, fill=INK)
    d.rounded_rectangle([cx, cy + 26, cx + 96, cy + 44], 9, fill=BSOFT, outline=BLUE, width=2)
    d.line([(cx + 30, cy + 26), (cx + 30, cy + 44)], fill=BLUE, width=2)
    d.text((cx, cy + 56), '~120 MB', font=f_small, fill=MUTED)
    d.text((cx, cy + 76), 'in the repo', font=f_small, fill=MUTED)
    arrow(d, cx + 112, y + h // 2, cx + 152, y + h // 2, BLUE, ramp(p, 0.05, 0.2))
    files = [('VCF', 'genotypes'), ('BED', 'phenotypes'), ('GTF', 'genes'), ('TSV', 'covariates')]
    fx0, fy0 = cx + 166, y + 12
    for k, (ext, what) in enumerate(files):
        on = ramp(p, 0.22 + k * 0.12, 0.32 + k * 0.12)
        if on <= 0:
            continue
        fx, fy = fx0 + (k % 2) * 132, fy0 + (k // 2) * 74
        d.polygon([(fx, fy), (fx + 30, fy), (fx + 42, fy + 12), (fx + 42, fy + 52), (fx, fy + 52)], fill=OK_SOFT, outline=OK)
        ctext(d, fx + 21, fy + 20, ext, f_file, OK)
        d.text((fx + 50, fy + 18), what, font=f_small, fill=INK)
    if p >= 0.8:
        d.text((fx0, fy0 + 152), 'no download needed', font=f_lab, fill=OK)


def d_run(d, x, y, w, h, p):
    # inputs -> SoS module -> outputs
    mid = y + h // 2
    ix = x + 18
    for k, ext in enumerate(['BED', 'GTF']):
        fy = mid - 58 + k * 64
        d.polygon([(ix, fy), (ix + 30, fy), (ix + 42, fy + 12), (ix + 42, fy + 52), (ix, fy + 52)], fill=OK_SOFT, outline=OK)
        ctext(d, ix + 21, fy + 20, ext, f_file, OK)
    mx, mw = ix + 112, 150
    arrow(d, ix + 52, mid, mx - 8, mid, BLUE, ramp(p, 0.05, 0.25))
    on = p > 0.25
    d.rounded_rectangle([mx, mid - 46, mx + mw, mid + 46], 10, fill=(BSOFT if on else PALE), outline=(BLUE if on else LINE), width=2)
    ctext(d, mx + mw / 2, mid - 32, 'SoS module', f_lab, INK)
    ctext(d, mx + mw / 2, mid - 8, 'gene_annotation', f_file, INK)
    # progress bar inside the module
    bx, bw = mx + 16, mw - 32
    d.rounded_rectangle([bx, mid + 20, bx + bw, mid + 30], 5, fill=(255, 255, 255))
    fw = int(bw * ramp(p, 0.3, 0.7))
    if fw > 6:
        d.rounded_rectangle([bx, mid + 20, bx + fw, mid + 30], 5, fill=BLUE)
    ox = mx + mw + 56
    arrow(d, mx + mw + 8, mid, ox - 8, mid, BLUE, ramp(p, 0.7, 0.82))
    if p >= 0.82:
        for k, ext in enumerate(['bed.gz', 'regions']):
            fy = mid - 58 + k * 64
            d.polygon([(ox, fy), (ox + 30, fy), (ox + 42, fy + 12), (ox + 42, fy + 52), (ox, fy + 52)], fill=BSOFT, outline=BLUE)
            d.text((ox + 50, fy + 18), ext, font=f_file, fill=INK)


DIAGRAMS = [d_install, d_clone, d_data, d_run]


def card(d, i, phase):
    """phase: <0 not started, 0..1 in progress, >=1 done."""
    x, y = POS[i]
    active, done = 0 <= phase < 1, phase >= 1
    d.rounded_rectangle([x, y, x + CW, y + CH], 14, fill=BG,
                        outline=(BLUE if active else BORDER), width=(3 if active else 2))
    bx, by = x + 40, y + 40
    if done:
        tick(d, bx, by, 18)
    else:
        d.ellipse([bx - 18, by - 18, bx + 18, by + 18], fill=(BLUE if active else IDLE))
        ctext(d, bx, by - 14, str(i + 1), f_num, ((255, 255, 255) if active else MUTED))
    d.text((x + 70, y + 24), TITLES[i], font=f_title, fill=(INK if phase >= 0 else MUTED))
    # diagram area
    dx, dy, dw, dh = x + 18, y + 70, CW - 36, CH - 140
    if phase >= 0:
        DIAGRAMS[i](d, dx, dy, dw, dh, min(phase / 0.8, 1.0))
    # command strip
    tx, ty, tw, th = x + 18, y + CH - 62, CW - 36, 44
    d.rounded_rectangle([tx, ty, tx + tw, ty + th], 8, fill=(TERM if phase >= 0 else IDLE))
    if phase < 0:
        return
    cmd = CMDS[i]
    typed = cmd if phase >= 0.5 else cmd[:int(len(cmd) * phase / 0.5)]
    d.text((tx + 14, ty + 11), '$', font=f_mono, fill=PROMPT)
    d.text((tx + 36, ty + 11), typed, font=f_mono, fill=TERM_TX)
    if active and phase < 0.5:
        cw = d.textbbox((0, 0), typed, font=f_mono)[2]
        d.rectangle([tx + 38 + cw, ty + 12, tx + 48 + cw, ty + 34], fill=TERM_TX)


def frame(t):
    """t runs 0..4: step i is in progress while i <= t < i+1."""
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    for i in range(4):
        card(d, i, t - i if t >= i else -1)
    return im


frames, dur = [], []
STEP_FRAMES = 16
for i in range(4):
    for k in range(STEP_FRAMES):
        frames.append(frame(i + k / STEP_FRAMES)); dur.append(75)
    frames.append(frame(i + 1)); dur.append(500)
frames.append(frame(4)); dur.append(3000)

pal = frames[-1].convert('P', palette=Image.ADAPTIVE, colors=128)
q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
o = 'code/images/xqtl_getting_started.gif'
q[0].save(o, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True, disposal=1)
print('frames=%d %.0f KB loop=%.1fs' % (len(q), os.path.getsize(o) / 1024, sum(dur) / 1000))
