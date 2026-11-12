"""
Handwritten Digit Recognition - upgraded edition
=================================================
Run it:   streamlit run app.py
Needs:    streamlit, numpy, pillow, scikit-learn   (scipy comes with scikit-learn)

Everything runs on your own computer: no API key and no internet connection are
needed for the AI. (The page only asks Google Fonts for one handwriting font;
without internet it simply falls back to another font.)

What is smarter than the original?
  1. Bug fix: the model was trained on pixels scaled 0..1, but the uploaded picture
     was divided by 16 instead of 255, so the AI saw digits ~16x too bright.
  2. prepare_digit(): instead of squashing the whole picture to 8x8, it finds the digit,
     crops it, thickens thin pens, centres it and shrinks it like the training data.
  3. Three different AI models vote (neural network + SVM + extra trees) and they are
     trained on shifted / rotated / blurred copies of the digits (data augmentation).
  4. Test-time augmentation: the picture is looked at from a few slightly different
     angles and the answers are averaged.
  5. The app says "not sure" when the vote is close instead of always sounding certain.
"""
import time
import warnings
import zlib

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_RS = getattr(Image, "Resampling", Image)  # works with old and new Pillow versions
BOX, BICUBIC = _RS.BOX, _RS.BICUBIC

FIT = 28   # longest side of the digit inside a 32x32 canvas (like the original dataset)
PEN = 0.20  # minimum pen thickness relative to the digit size (thin pens get widened)
EXPERTS = {"net": "Neural network", "svm": "Support vector machine", "trees": "Extra trees"}


# ----------------------------------------------------------------------------
# 1. THE AI BRAIN: three models, trained on the digits plus augmented copies
# ----------------------------------------------------------------------------
def augment(X, y, copies=4, seed=0):
    """Make shifted, rotated, zoomed, thicker and noisy copies of the 8x8 training digits."""
    rng = np.random.default_rng(seed)
    centre = np.array([3.5, 3.5])
    all_X, all_y = [X], [y]
    for _ in range(copies):
        batch = np.empty_like(X)
        for i, img in enumerate(X):
            angle = np.deg2rad(rng.uniform(-15, 15))
            m = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            m = m / rng.uniform(0.88, 1.12)                       # zoom
            move = rng.uniform(-1, 1, 2)                          # shift by up to 1 pixel
            out = ndi.affine_transform(img, m, offset=centre - m @ (centre + move), order=1)
            if rng.random() < 0.5:                                # pretend the pen was thicker
                out = ndi.gaussian_filter(out, rng.uniform(0.3, 0.7))
                out = out / max(out.max(), 1e-6)
            batch[i] = np.clip(out + rng.normal(0, 0.03, out.shape), 0, 1)
        all_X.append(batch)
        all_y.append(y)
    return np.concatenate(all_X), np.concatenate(all_y)


@st.cache_resource(show_spinner="Training the three AI models (only on the first start)...")
def load_model():
    digits = load_digits()
    X, y = digits.images / 16.0, digits.target                   # pixels are 0..1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_big, y_big = augment(X_train, y_train)   # augment AFTER the split: test digits stay unseen
    flat = lambda a: a.reshape(len(a), -1)

    model = VotingClassifier(
        [
            ("net", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300,
                                  early_stopping=True, random_state=42)),
            ("svm", SVC(C=10, probability=True, random_state=42)),
            ("trees", ExtraTreesClassifier(n_estimators=300, random_state=42)),
        ],
        voting="soft",                         # average the probabilities of all three
    )
    model.fit(flat(X_big), y_big)

    scores = {"ensemble": model.score(flat(X_test), y_test)}
    for name, est in model.named_estimators_.items():
        scores[name] = est.score(flat(X_test), y_test)
    return model, scores, X_test, y_test


# ----------------------------------------------------------------------------
# 2. TEACHING THE AI TO SEE: turn any picture into the 8x8 grid it was trained on
# ----------------------------------------------------------------------------
def otsu(u8):
    """Best brightness threshold that separates ink from paper."""
    p = np.bincount(u8.ravel(), minlength=256) / u8.size
    w, m = np.cumsum(p), np.cumsum(p * np.arange(256))
    between = (m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-12)
    return int(between.argmax())


def to_gray(image):
    image = ImageOps.exif_transpose(image)                        # phone photos can be rotated
    if image.mode in ("RGBA", "LA", "P"):                         # transparent PNG -> white paper
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    gray = image.convert("L")
    gray.thumbnail((400, 400))
    return np.asarray(gray, dtype=np.float32)


def find_ink(a):
    """True where the pen touched the paper (specks and noise are ignored)."""
    thr = np.clip(otsu((a * 255).astype(np.uint8)) / 255, 0.2, 0.6)
    mask = a > thr
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n > 1:
        sizes = ndi.sum(mask, labels, range(1, n + 1))
        mask = np.isin(labels, 1 + np.flatnonzero(sizes >= 0.08 * sizes.max()))
    return mask


def resize_to(arr, longest):
    h, w = arr.shape
    s = longest / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.resize(size, BOX if s < 1 else BICUBIC), dtype=np.float32) / 255


def thicken(work):
    """Thin pens turn faint when shrunk to 8x8, so widen them to the training thickness."""
    binary = work > 0.4
    if binary.sum() < 10:
        return work
    pen = 4 * ndi.distance_transform_edt(binary)[binary].mean()   # rough pen width in pixels
    r = int(round((PEN * max(work.shape) - pen) / 2))
    if r < 1:
        return work
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    out = ndi.grey_dilation(np.pad(work, r), footprint=(xx * xx + yy * yy) <= r * r)
    ys, xs = np.nonzero(out > 0.2)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def prepare_digit(image, background="auto"):
    """Picture in -> 8x8 grid (values 0..1, ink = bright) out. Returns None if no digit is found."""
    a = to_gray(image)

    # ink must be bright on a dark background, like in the training data
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    light_background = np.median(border) > 127 if background == "auto" else background == "light"
    if light_background:
        a, border = 255 - a, 255 - border
    paper = np.median(border)                                     # brightness of the empty background
    inked = a > otsu(np.clip(a, 0, 255).astype(np.uint8))
    if not inked.any() or a[inked].mean() - paper < 25:           # flat colour, nothing drawn
        return None
    a = np.clip((a - paper) / (a[inked].mean() - paper), 0, 1)

    mask = find_ink(a)
    if background == "auto" and mask.mean() > 0.5:                # colours were probably the wrong way round
        a = 1 - a
        mask = find_ink(a)
    if mask.sum() < 12:
        return None

    # crop to the digit (keeping soft edges), widen thin pens, then fit it into a 32x32 canvas
    ys, xs = np.nonzero(mask)
    near = ndi.binary_dilation(mask, iterations=2)
    crop = (a * near)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    small = resize_to(thicken(resize_to(crop, 64)), FIT)
    canvas = np.zeros((32, 32), np.float32)
    top, left = (32 - small.shape[0]) // 2, (32 - small.shape[1]) // 2
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small

    # centre by "centre of mass" (as in the dataset), then average every 4x4 block -> 8x8
    cy, cx = ndi.center_of_mass(canvas)
    canvas = ndi.shift(canvas, (np.clip(15.5 - cy, -4, 4), np.clip(15.5 - cx, -4, 4)), order=1)
    grid = canvas.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    return grid / max(grid.max(), 1e-6)


# ----------------------------------------------------------------------------
# 3. ASKING THE AI: vote of three models, looking from several angles
# ----------------------------------------------------------------------------
def variants(grid, tta):
    """The same digit nudged and tilted a little (test-time augmentation)."""
    out = [grid]
    if tta:
        for dy, dx in [(0.6, 0), (-0.6, 0), (0, 0.6), (0, -0.6)]:
            out.append(ndi.shift(grid, (dy, dx), order=1))
        for angle in (-10, 10):
            out.append(ndi.rotate(grid, angle, reshape=False, order=1))
    return np.array(out).reshape(len(out), -1)


def predict(model, grid, tta=True):
    batch = variants(grid, tta)
    probs = model.predict_proba(batch).mean(axis=0)
    experts = {}
    for name, est in model.named_estimators_.items():
        p = est.predict_proba(batch).mean(axis=0)
        experts[name] = (int(p.argmax()), float(p.max()))
    return probs, experts


# ----------------------------------------------------------------------------
# 4. LOOKS: squared notebook paper, blue ballpoint ink, a teacher's pen
# ----------------------------------------------------------------------------
STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&display=swap');
@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }
@property --n { syntax: '<integer>'; inherits: true; initial-value: 0; }
:root { --paper:#fbfcff; --grid:rgba(52,84,180,.14); --line:rgba(52,84,180,.3); --ink:#1f3fbf; --ink-dark:#172055; --hi:#ffe45e; --red:#d63b3b; --green:#1b9a5f; --pencil:#656b80; --hand:'Caveat','Segoe Print','Bradley Hand','Comic Sans MS',cursive; }
.paper { position:relative; background-color:var(--paper); background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px); background-size:24px 24px; color:var(--ink-dark); border:1px solid var(--line); border-radius:4px; padding:1.1rem 1.4rem; box-shadow:0 14px 26px -18px rgba(23,32,85,.6); line-height:1.4; }
.paper .note { color:var(--pencil); font-size:.92rem; }
.paper .sec { margin:1.1rem 0 .45rem; padding-top:.7rem; border-top:1px dashed var(--line); font-weight:600; }
.title { font-family:var(--hand); font-size:3.3rem; font-weight:700; line-height:1; color:var(--ink); }
.title::after { content:''; display:block; width:min(100%,440px); height:5px; margin-top:6px; border-radius:3px; background:var(--ink); transform:rotate(-.6deg); }
.lv-sure, .lv-think { --ring:var(--ink); }
.lv-unsure { --ring:var(--red); }
.answer { display:flex; align-items:center; gap:1.8rem; flex-wrap:wrap; }
.digit-wrap { position:relative; display:grid; place-items:center; width:150px; height:150px; }
.digit { font-family:var(--hand); font-size:6.6rem; font-weight:700; line-height:1; color:var(--ring); }
.circle { position:absolute; inset:4px 6px; border:4px solid var(--ring); border-radius:46% 54% 50% 50% / 55% 45% 55% 45%; transform:rotate(-5deg); --p:100; -webkit-mask:conic-gradient(from 210deg,#000 calc(var(--p) * 1%),transparent 0); mask:conic-gradient(from 210deg,#000 calc(var(--p) * 1%),transparent 0); }
.verdict { font-size:1.35rem; font-weight:600; }
.mark { background:linear-gradient(transparent 55%,var(--hi) 55%); padding:0 .25rem; }
.conf { margin-top:.3rem; display:flex; align-items:baseline; gap:.5rem; }
.pct { font-size:2.1rem; font-weight:700; counter-reset:num var(--n); }
.pct::after { content:counter(num) '%'; }
.grade { display:inline-block; margin-top:.5rem; font-family:var(--hand); font-size:1.8rem; font-weight:700; transform:rotate(-4deg); }
.grade.ok { color:var(--green); }
.grade.bad { color:var(--red); }
.bar { display:grid; grid-template-columns:1.6rem 1fr 3.6rem; align-items:center; gap:.6rem; margin:.3rem 0; }
.bar .d { font-family:var(--hand); font-size:1.6rem; font-weight:700; color:var(--ink); line-height:1; }
.bar .track { height:16px; border-bottom:1px solid var(--line); }
.bar .fill { height:100%; background:rgba(31,63,191,.35); border-radius:2px; }
.bar.top .fill { background:var(--hi); box-shadow:0 0 0 1px rgba(0,0,0,.08); }
.bar .v { text-align:right; font-variant-numeric:tabular-nums; color:var(--pencil); font-size:.9rem; }
.bar.wide { grid-template-columns:7.6rem 1fr 3.4rem; }
.bar.wide .d { font-family:inherit; font-size:.88rem; font-weight:500; color:var(--ink-dark); }
.vote { display:flex; justify-content:space-between; align-items:baseline; margin:.3rem 0; }
.vote small { color:var(--pencil); margin-left:.4rem; }
.vote .g { font-family:var(--hand); font-size:1.8rem; font-weight:700; line-height:1; }
.vote.agree .g { color:var(--green); }
.vote.differ .g { color:var(--red); }
.pixwrap { display:inline-block; padding:10px; background:var(--paper); border:1px solid var(--line); border-radius:4px; box-shadow:0 14px 26px -18px rgba(23,32,85,.6); }
.pix { display:grid; grid-template-columns:repeat(8,26px); border-top:1px solid var(--line); border-left:1px solid var(--line); }
.pix i { display:block; box-sizing:border-box; width:26px; height:26px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); }
.empty .row { display:flex; gap:1rem; margin-top:.5rem; flex-wrap:wrap; }
.empty .row span { font-family:var(--hand); font-size:3.2rem; font-weight:700; color:var(--ink); line-height:1; }
@media (prefers-reduced-motion: no-preference) {
  .title::after { animation:penline 1s .25s cubic-bezier(.6,0,.2,1) both; }
  @keyframes penline { from { clip-path:inset(0 100% 0 0); } to { clip-path:inset(0 0 0 0); } }
  .empty .row span { animation:rise .5s ease-out both; }
  @keyframes rise { from { opacity:0; transform:translateY(10px) scale(.8); } }
  .side .fill { animation:sweep .9s cubic-bezier(.2,.8,.2,1) both; }
  @keyframes sweep { from { width:0; } }
}
</style>"""

# These animations get a new name for every answer, so the browser plays them again each time.
RUN_CSS = """<style>
.res-UID .pct { --n:PCT; }
@media (prefers-reduced-motion: no-preference) {
  .res-UID .pix i { animation:fillUID .35s ease-out both; }
  @keyframes fillUID { from { opacity:0; transform:scale(.2); } }
  .res-UID .circle { animation:drawUID .9s .6s cubic-bezier(.5,.1,.3,1) both; }
  @keyframes drawUID { from { --p:0; } }
  .res-UID .digit { animation:inkUID .5s .4s ease-out both; }
  @keyframes inkUID { from { opacity:0; transform:scale(.6); } }
  .res-UID .pct { animation:countUID 1.1s .6s ease-out both; }
  @keyframes countUID { from { --n:0; } }
  .res-UID .fill { animation:sweepUID .8s cubic-bezier(.2,.8,.2,1) both; }
  @keyframes sweepUID { from { width:0; } }
  .res-UID .vote { animation:slideUID .5s ease-out both; }
  @keyframes slideUID { from { opacity:0; transform:translateX(-12px); } }
  .res-UID .grade { animation:stampUID .45s 1.3s cubic-bezier(.3,1.6,.5,1) both; }
  @keyframes stampUID { from { opacity:0; transform:rotate(-4deg) scale(2); } }
}
</style>"""

HERO = (
    '<div class="paper"><div class="title">Handwritten digit recognition</div>'
    '<div class="note" style="margin-top:.7rem">Show me one digit from 0 to 9. '
    'Three small AI models look at it and vote.</div></div>'
)

EMPTY = (
    '<div class="paper empty"><div class="note">Nothing to read yet. Upload a picture of one digit, '
    'take a photo, or pick a random test digit.</div><div class="row">'
    + "".join(f'<span style="animation-delay:{i * 90}ms">{i}</span>' for i in range(10))
    + "</div></div>"
)


def bar_html(label, value, delay, top=False, wide=False):
    cls = "bar" + (" top" if top else "") + (" wide" if wide else "")
    width = max(value * 100, 1.5)
    return (
        f'<div class="{cls}"><span class="d">{label}</span>'
        f'<div class="track"><div class="fill" style="width:{width:.1f}%;animation-delay:{delay}ms"></div></div>'
        f'<span class="v">{value:.1%}</span></div>'
    )


def pixel_html(uid, grid):
    cells = "".join(
        f'<i style="background:rgba(31,63,191,{grid[r, c]:.2f});animation-delay:{(r + c) * 45}ms"></i>'
        for r in range(8) for c in range(8)
    )
    return f'<div class="pixwrap res-{uid}"><div class="pix">{cells}</div></div>'


def result_html(uid, probs, experts, truth):
    order = np.argsort(probs)[::-1]
    best, second, conf = int(order[0]), int(order[1]), float(probs[order[0]])
    article = lambda d: "an" if d == 8 else "a"
    if conf >= 0.85:
        level, line = "sure", f'This is {article(best)} <span class="mark">{best}</span>.'
    elif conf >= 0.55:
        level, line = "think", f'I think this is {article(best)} <span class="mark">{best}</span>.'
    else:
        level = "unsure"
        line = (f'Not sure: {article(best)} <span class="mark">{best}</span> '
                f'or {article(second)} <span class="mark">{second}</span>?')

    grade = ""
    if truth is not None:
        ok = truth == best
        grade = f'<div class="grade {"ok" if ok else "bad"}">Real answer: {truth} {"✔" if ok else "✘"}</div>'

    bars = "".join(bar_html(int(d), probs[d], 500 + i * 90, top=(i == 0)) for i, d in enumerate(order[:5]))
    votes = "".join(
        f'<div class="vote {"agree" if guess == best else "differ"}" style="animation-delay:{900 + i * 140}ms">'
        f'<span>{EXPERTS[name]}<small>{c:.0%} sure</small></span><span class="g">{guess}</span></div>'
        for i, (name, (guess, c)) in enumerate(experts.items())
    )
    agree = sum(g == best for g, _ in experts.values())
    summary = ("All three models agree." if agree == len(experts)
               else f"{agree} of {len(experts)} models agree. The combined vote decides.")

    pct = round(conf * 100)
    return (
        RUN_CSS.replace("UID", uid).replace("PCT", str(pct))
        + f'<div class="paper res-{uid} lv-{level}"><div class="answer">'
        f'<div class="digit-wrap"><div class="circle"></div><div class="digit">{best}</div></div>'
        f'<div><div class="verdict">{line}</div>'
        f'<div class="conf"><span class="pct" title="{pct}% sure"></span><span>sure</span></div>{grade}</div></div>'
        f'<div class="sec">How sure am I about each digit?</div>{bars}'
        f'<div class="sec">How each model voted</div>{votes}<div class="note">{summary}</div></div>'
    )


# ----------------------------------------------------------------------------
# 5. THE APP
# ----------------------------------------------------------------------------
def open_image(file):
    try:
        img = Image.open(file)
        img.load()
        return img
    except Exception:
        return None


def sample_image(grid8):
    """A test digit as a black-on-white picture, so it goes through the same pipeline as uploads."""
    big = Image.fromarray((grid8 * 255).astype(np.uint8)).resize((240, 240), BICUBIC)
    return ImageOps.invert(big).convert("RGB")


def sidebar(scores, n_test):
    sb = st.sidebar
    sb.header("For a good reading")
    sb.markdown(
        "- One digit per picture\n"
        "- Dark pen on light paper works best (the opposite works too)\n"
        "- Thick strokes, no shadows\n"
        "- No need to centre or resize it. I crop and centre it for you."
    )
    sb.header("Settings")
    bg = sb.selectbox("Background", ["Detect automatically", "Light background, dark digit",
                                      "Dark background, light digit"])
    tta = sb.toggle("Look from several angles", value=True,
                    help="Averages the answer over slightly shifted and tilted copies of the digit.")
    steps = sb.toggle("Show the thinking steps", value=True)
    party = sb.toggle("Balloons for sure answers", value=True)
    sb.header("About the AI")
    rows = [bar_html("All three together", scores["ensemble"], 100, top=True, wide=True)]
    rows += [bar_html(EXPERTS[k], scores[k], 200 + 100 * i, wide=True) for i, k in enumerate(EXPERTS)]
    sb.markdown(
        f'<div class="paper side"><div class="note">Accuracy on {n_test} digits the models never saw '
        f'while training.</div>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )
    return {"bg": {"Detect automatically": "auto", "Light background, dark digit": "light",
                   "Dark background, light digit": "dark"}[bg],
            "tta": tta, "steps": steps, "party": party}


def main():
    st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)

    try:
        model, scores, X_test, y_test = load_model()
    except Exception as e:
        st.error(f"Could not train the AI models: {e}")
        st.stop()
    cfg = sidebar(scores, len(y_test))

    st.write("")
    mode = st.radio("Show me a digit", ["Upload", "Camera", "Random test digit"], horizontal=True)
    image = truth = key = None
    if mode == "Upload":
        file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
        if file is not None:
            image, key = open_image(file), f"up-{zlib.crc32(file.getvalue())}"
            if image is None:
                st.error("I couldn't open that file. Please upload a PNG or JPG picture.")
    elif mode == "Camera":
        file = st.camera_input("Take a picture of one digit")
        if file is not None:
            image, key = open_image(file), f"cam-{zlib.crc32(file.getvalue())}"
    else:
        picked = st.button("Pick another digit")
        if picked or "sample" not in st.session_state:
            st.session_state.sample = (int(np.random.randint(len(X_test))), time.time())
        idx, stamp = st.session_state.sample
        image, truth, key = sample_image(X_test[idx]), int(y_test[idx]), f"sample-{idx}-{stamp}"

    if image is None:
        st.markdown(EMPTY, unsafe_allow_html=True)
        return

    # run the AI (with a short "thinking" animation the first time we see a picture)
    first_time = st.session_state.get("last_key") != key
    steps = st.status("Thinking...", expanded=True) if (first_time and cfg["steps"]) else None

    def say(text):
        if steps is not None:
            steps.write(text)
            time.sleep(0.4)

    say("Finding the digit, centring it and shrinking it to 8×8 pixels...")
    grid = prepare_digit(image, cfg["bg"])
    result = None
    if grid is not None:
        say("Asking the three AI models to vote...")
        result = predict(model, grid, cfg["tta"])
    if steps is not None:
        steps.update(label="Done" if result else "No digit found",
                     state="complete" if result else "error", expanded=False)
    st.session_state["last_key"] = key

    if result is None:
        st.warning("I can't find a digit in this picture. Try a darker pen, a plainer background, "
                   "and make sure the digit isn't tiny.")
        return
    probs, experts = result
    uid = "r" + str(zlib.crc32(repr((key, cfg["bg"], cfg["tta"])).encode()))

    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.markdown("**Your picture**")
        st.image(image, width=230)
        st.markdown("**What the AI sees**")
        st.markdown(pixel_html(uid, grid), unsafe_allow_html=True)
        st.caption("Only these 64 numbers, from 0 (blank) to 1 (full ink). Everything else is thrown away.")
    with right:
        st.markdown(result_html(uid, probs, experts, truth), unsafe_allow_html=True)

    if first_time:
        best, conf = int(probs.argmax()), float(probs.max())
        if truth is not None and best != truth:
            st.toast(f"Missed that one. It was a {truth}.", icon="😅")
        elif conf >= 0.85:
            st.toast(f"I'm {conf:.0%} sure it's {best}.", icon="🎯")
            if cfg["party"]:
                st.balloons()


if __name__ == "__main__":
    main()
