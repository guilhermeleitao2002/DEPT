#!/usr/bin/env python3
"""Generate the Chapter 5 evaluation figures as house-style SVGs.

Reads measured metrics from data/models/ and writes:
  eval_streaming_auroc.svg     - AUROC@t curves (RF / TCN / meta-learner)
  eval_persource_strip.svg     - per-source AUROC@max strip plot, by model & mode
  eval_purification_scatter.svg- purification gain vs perturbation footprint
  eval_evasion_cost.svg        - ASR vs attacker spend, one line per target (RQ1)
  eval_baseline.svg            - deployment baseline: NIDS ASR before vs after DEPT

House palette (dissertation/Images conventions):
  blue #2c5f8a (fill #d7e4f2) | red #c44536 (#fbeae8) | teal #2a9d8f (#dff0ed)
  greys #5a6673 / #eef1f4 | axis #3a444e | Helvetica
Model glyphs follow dept_cadence.svg: TCN=circle, RF=square, meta=diamond.
Mode glyphs: both=circle, iat=triangle, size=square.
"""
import json, math, os

ROOT = "/mnt/nas/inesc/gleitao/DEPT"
OUT = os.path.join(ROOT, "dissertation", "Images")

BLUE, BLUE_D = "#2c5f8a", "#16344c"
RED, RED_D = "#c44536", "#7d2b21"
TEAL, TEAL_D = "#2a9d8f", "#14544c"
GREY, GREY_D = "#5a6673", "#3a444e"
GRID = "#e3e7ec"
MUT = "#8a949e"
INK = "#1a1a1a"

def load(p):
    return json.load(open(os.path.join(ROOT, p)))

def write_placeholder(name, note):
    """Emit a minimal house-style 'pending run' panel so the .tex reference resolves
    and the build never breaks; regenerating from real data overwrites it."""
    W, H = 1240, 300
    p = [f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">']
    p.append(f'  <rect x="8" y="8" width="{W-16}" height="{H-16}" fill="#f6f8fa" stroke="{GRID}" stroke-width="2" stroke-dasharray="8,6"/>')
    p.append(f'  <text x="{W//2}" y="{H//2-8}" text-anchor="middle" font-size="26" fill="{GREY}">Figure pending the full-scale run</text>')
    p.append(f'  <text x="{W//2}" y="{H//2+28}" text-anchor="middle" font-size="19" fill="{MUT}">{note}</text>')
    p.append('</svg>')
    open(os.path.join(OUT, name), "w").write("\n".join(p) + "\n")
    print(f"wrote {name} (placeholder)")

def fmt(x):
    return f"{x:.1f}".rstrip("0").rstrip(".")

# ---------------------------------------------------------------- glyphs ----
def circle(x, y, r, fill, stroke, sw=1.5, ring=True):
    ringat = f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r+sw)}" fill="#ffffff"/>' if ring else ""
    return f'{ringat}<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def square(x, y, s, fill, stroke, sw=1.5, ring=True):
    h = s / 2
    ringat = f'<rect x="{fmt(x-h-sw)}" y="{fmt(y-h-sw)}" width="{fmt(s+2*sw)}" height="{fmt(s+2*sw)}" fill="#ffffff"/>' if ring else ""
    return f'{ringat}<rect x="{fmt(x-h)}" y="{fmt(y-h)}" width="{fmt(s)}" height="{fmt(s)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def diamond(x, y, r, fill, stroke, sw=1.5, ring=True):
    def pts(rr):
        return f"{fmt(x)},{fmt(y-rr)} {fmt(x+rr)},{fmt(y)} {fmt(x)},{fmt(y+rr)} {fmt(x-rr)},{fmt(y)}"
    ringat = f'<polygon points="{pts(r+1.8)}" fill="#ffffff"/>' if ring else ""
    return f'{ringat}<polygon points="{pts(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def triangle(x, y, r, fill, stroke, sw=1.5, ring=True):
    def pts(rr):
        return (f"{fmt(x)},{fmt(y-rr)} {fmt(x+rr*0.866)},{fmt(y+rr*0.5)} "
                f"{fmt(x-rr*0.866)},{fmt(y+rr*0.5)}")
    ringat = f'<polygon points="{pts(r+2.2)}" fill="#ffffff"/>' if ring else ""
    return f'{ringat}<polygon points="{pts(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

MODE_GLYPH = {  # glyph, colors, nominal radius
    "both": (circle, BLUE, BLUE_D, 8.5),
    "iat": (triangle, RED, RED_D, 10.5),
    "size": (square, TEAL, TEAL_D, 15.0),
}

def mode_mark(mode, x, y, ring=True):
    fn, fill, stroke, r = MODE_GLYPH[mode]
    return fn(x, y, r, fill, stroke, ring=ring)

# ------------------------------------------------------- fig A: AUROC@t ----
def fig_streaming():
    ts = ["1", "2", "5", "10", "20", "50", "100", "max"]
    series = {}
    for name, path in [("rf", "data/models/rf/streaming_metrics.json"),
                       ("tcn", "data/models/tcn/streaming_metrics.json"),
                       ("meta", "data/models/metalearner/streaming_metrics.json")]:
        a = load(path)["overall"]["auroc_at"]
        series[name] = [a[t]["auroc"] for t in ts]

    W, H = 1240, 640
    x0, x1, y0, y1 = 150, 1060, 560, 70  # plot box (y grows downward)
    ymin, ymax = 0.80, 1.00
    ystep = 0.05
    xs = [x0 + i * (x1 - x0) / (len(ts) - 1) for i in range(len(ts))]
    def Y(v):
        return y0 + (v - ymin) * (y1 - y0) / (ymax - ymin)

    p = []
    p.append(f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')

    # gridlines + y ticks
    p.append('  <!-- grid and axes -->')
    v = ymin
    while v <= ymax + 1e-9:
        yy = Y(v)
        p.append(f'  <line x1="{x0}" y1="{fmt(yy)}" x2="{x1}" y2="{fmt(yy)}" stroke="{GRID}" stroke-width="1.5"/>')
        p.append(f'  <text x="{x0-14}" y="{fmt(yy+6)}" text-anchor="end" font-size="19" fill="{GREY}">{v:.2f}</text>')
        v = round(v + ystep, 4)
    # x ticks
    for i, t in enumerate(ts):
        lbl = t if t != "max" else "max"
        p.append(f'  <line x1="{fmt(xs[i])}" y1="{y0}" x2="{fmt(xs[i])}" y2="{y0+8}" stroke="{GREY_D}" stroke-width="1.5"/>')
        p.append(f'  <text x="{fmt(xs[i])}" y="{y0+34}" text-anchor="middle" font-size="19" fill="{GREY}">{lbl}</text>')
    # axes
    p.append(f'  <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{GREY_D}" stroke-width="2"/>')
    p.append(f'  <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1-10}" stroke="{GREY_D}" stroke-width="2"/>')
    p.append(f'  <text x="{(x0+x1)//2}" y="{y0+72}" text-anchor="middle" font-size="21" fill="{INK}">packets seen (t)</text>')
    p.append(f'  <text x="46" y="{(y0+y1)//2}" text-anchor="middle" font-size="21" fill="{INK}" transform="rotate(-90 46 {(y0+y1)//2})">flow-level AUROC@t</text>')

    def polyline(vals, color, start=0):
        pts = " ".join(f"{fmt(xs[i])},{fmt(Y(v))}" for i, v in enumerate(vals) if v is not None)
        return f'  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>'

    # RF (grey squares, starts at t=2)
    p.append('  <!-- Random Forest: grey squares, first verdict at packet 2 -->')
    p.append(polyline(series["rf"], GREY))
    for i, v in enumerate(series["rf"]):
        if v is not None:
            p.append("  " + square(xs[i], Y(v), 15, GREY, GREY_D))
    p.append(f'  <rect x="{fmt(xs[0]-9)}" y="{fmt(Y(series["rf"][1])-9)}" width="18" height="18" fill="none" stroke="{MUT}" stroke-width="2" stroke-dasharray="4,3"/>')
    p.append(f'  <text x="{fmt(xs[0]+18)}" y="{fmt(Y(series["rf"][1])+6)}" text-anchor="start" font-size="17" fill="{MUT}">no RF verdict yet</text>')

    # TCN (blue circles)
    p.append('  <!-- TCN: blue circles, dense from packet 1 -->')
    p.append(polyline(series["tcn"], BLUE))
    for i, v in enumerate(series["tcn"]):
        p.append("  " + circle(xs[i], Y(v), 8.5, BLUE, BLUE_D))

    # meta (teal diamonds)
    p.append('  <!-- meta-learner: teal diamonds -->')
    p.append(polyline(series["meta"], TEAL))
    for i, v in enumerate(series["meta"]):
        p.append("  " + diamond(xs[i], Y(v), 10.5, TEAL, TEAL_D))

    # direct labels at the right edge, ordered by final value so they never collide
    p.append('  <!-- direct series labels -->')
    lx = x1 + 18
    finals = sorted((("rf", GREY), ("meta", TEAL), ("tcn", BLUE)),
                    key=lambda kc: -series[kc[0]][-1])
    for key, color in finals:
        name = {"rf": "RF", "meta": "meta", "tcn": "TCN"}[key]
        p.append(f'  <text x="{lx}" y="{fmt(Y(series[key][-1])+6)}" font-size="21" '
                 f'font-weight="bold" fill="{color}">{name}  {series[key][-1]:.4f}</text>')

    # the packet-1 story: only the TCN and the meta-learner have a verdict there
    p.append(f'  <line x1="{fmt(xs[0])}" y1="{fmt(Y(series["meta"][0])+16)}" x2="{fmt(xs[0])}" y2="{fmt(Y(0.876))}" stroke="{MUT}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    p.append(f'  <text x="{fmt(xs[0]+12)}" y="{fmt(Y(0.874))}" text-anchor="start" font-size="18" fill="{GREY}">only the TCN and the meta-learner</text>')
    p.append(f'  <text x="{fmt(xs[0]+12)}" y="{fmt(Y(0.8625))}" text-anchor="start" font-size="18" fill="{GREY}">have a verdict at packet 1</text>')

    p.append('</svg>')
    open(os.path.join(OUT, "eval_streaming_auroc.svg"), "w").write("\n".join(p) + "\n")
    print("wrote eval_streaming_auroc.svg")

# --------------------------------------------- fig B: per-source strip ----
def fig_strip():
    models = [("Random Forest", "rf", "data/models/rf/streaming_metrics.json"),
              ("TCN", "tcn", "data/models/tcn/streaming_metrics.json"),
              ("Meta-learner", "meta", "data/models/metalearner/streaming_metrics.json")]
    # vs_same_base_benign is the axis that matches DEPT's actual task: separate a
    # TANTGen-perturbed flow of source X from an *unperturbed flow of the same
    # source*, rather than from generic benign traffic of the other nine sources.
    data = {}
    for _, key, path in models:
        s = load(path)["per_adversarial_source"]
        rows = []
        for src, d in s.items():
            base, mode = src.rsplit("_adv_", 1)
            a = d["vs_same_base_benign"]["auroc_at"]
            v = (a.get("max") or list(a.values())[-1])["auroc"]
            rows.append((base, mode, v))
        data[key] = rows

    W, H = 1240, 660
    x0, x1 = 240, 1150
    vmin, vmax = 0.14, 1.03
    def X(v):
        return x0 + (v - vmin) * (x1 - x0) / (vmax - vmin)
    rows_y = {"rf": 175, "tcn": 340, "meta": 505}
    lane = {"both": -30, "iat": 0, "size": 30}
    band_h = 118

    p = []
    p.append(f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')

    # alternating row bands
    for i, (_, key, _) in enumerate(models):
        if i % 2 == 0:
            p.append(f'  <rect x="{x0-12}" y="{rows_y[key]-band_h//2}" width="{x1-x0+24}" height="{band_h}" fill="#f6f8fa"/>')

    # x grid + ticks
    v = 0.2
    while v <= 1.0 + 1e-9:
        xx = X(v)
        p.append(f'  <line x1="{fmt(xx)}" y1="90" x2="{fmt(xx)}" y2="575" stroke="{GRID}" stroke-width="1.5"/>')
        p.append(f'  <text x="{fmt(xx)}" y="602" text-anchor="middle" font-size="19" fill="{GREY}">{v:.1f}</text>')
        v = round(v + 0.1, 3)
    # chance line
    xx = X(0.5)
    p.append(f'  <line x1="{fmt(xx)}" y1="82" x2="{fmt(xx)}" y2="575" stroke="{RED}" stroke-width="2" stroke-dasharray="7,5" opacity="0.75"/>')
    p.append(f'  <text x="{fmt(xx)}" y="70" text-anchor="middle" font-size="18" fill="{RED}">chance (0.5)</text>')
    p.append(f'  <text x="{(x0+x1)//2}" y="640" text-anchor="middle" font-size="21" fill="{INK}">end-of-flow streaming AUROC@max against same-source benign flows, one mark per source and mode</text>')

    # row labels
    for label, key, _ in models:
        p.append(f'  <text x="{x0-26}" y="{rows_y[key]-8}" text-anchor="end" font-size="22" font-weight="bold" fill="{INK}">{label}</text>')

    # marks, with vertical fan-out for near-ties inside a lane
    ann = []
    for label, key, _ in models:
        p.append(f'  <!-- {label} -->')
        lanes = {}
        for base, mode, v in sorted(data[key], key=lambda r: r[2]):
            lanes.setdefault(mode, []).append((base, v))
        for mode, pts in lanes.items():
            yb = rows_y[key] + lane[mode]
            taken = []
            for base, v in pts:
                xx = X(v)
                bump, step = 0, 0
                while any(abs(xx - tx) < 17 and abs(yb + bump - ty) < 15 for tx, ty in taken):
                    step += 1
                    bump = (step + 1) // 2 * 13 * (1 if step % 2 else -1)
                yy = yb + bump
                taken.append((xx, yy))
                p.append("  " + mode_mark(mode, xx, yy))
                if key == "rf" and base == "ntp_ddos" and mode == "size":
                    ann.append((xx, yy, "ntp ddos (size)  0.21", -1))
                if key == "tcn" and base == "portscan" and mode == "iat":
                    ann.append((xx, yy, "portscan (iat)  0.65", 1))
                if key == "meta" and base == "dns_ddos" and mode == "size":
                    ann.append((xx, yy, "dns ddos (size)  0.44", -1))

    for xx, yy, text, side in ann:
        ty = yy + 46 * side
        p.append(f'  <line x1="{fmt(xx)}" y1="{fmt(yy+14*side)}" x2="{fmt(xx)}" y2="{fmt(ty-12*side)}" stroke="{MUT}" stroke-width="1.5"/>')
        p.append(f'  <text x="{fmt(xx)}" y="{fmt(ty + (14 if side>0 else -2))}" text-anchor="middle" font-size="18" fill="{GREY}">{text}</text>')

    # the sub-chance cluster: the reflection floods' size variants
    p.append(f'  <text x="{fmt(X(0.36))}" y="104" text-anchor="middle" font-size="18" fill="{GREY}">reflection floods, size variants: below chance</text>')
    p.append(f'  <line x1="{fmt(X(0.36))}" y1="110" x2="{fmt(X(0.36))}" y2="126" stroke="{MUT}" stroke-width="1.5"/>')

    # legend (fixed order: both, iat, size)
    lx, ly = x0 + 6, 38
    p.append('  <!-- legend -->')
    p.append("  " + mode_mark("both", lx, ly))
    p.append(f'  <text x="{lx+20}" y="{ly+7}" font-size="20" fill="{INK}">both</text>')
    p.append("  " + mode_mark("iat", lx + 130, ly))
    p.append(f'  <text x="{lx+150}" y="{ly+7}" font-size="20" fill="{INK}">iat</text>')
    p.append("  " + mode_mark("size", lx + 240, ly))
    p.append(f'  <text x="{lx+260}" y="{ly+7}" font-size="20" fill="{INK}">size</text>')

    p.append('</svg>')
    open(os.path.join(OUT, "eval_persource_strip.svg"), "w").write("\n".join(p) + "\n")
    print("wrote eval_persource_strip.svg")

# ------------------------------------- fig C: purification scatter ----
def fig_scatter():
    m = load("data/models/reconstructor/metrics.json")["per_source"]
    pts = []
    for src, d in sorted(m.items()):
        base, mode = src.rsplit("_adv_", 1)
        mse = d["reconstruction"]["perturbed_mse_before"]
        gain = d["purification"]["gain"]
        if mse is None or gain is None:
            continue  # syn_ddos_both: no perturbed-dim footprint recorded
        pts.append((base, mode, mse, gain))

    W, H = 1240, 700
    x0, x1, y0, y1 = 170, 1130, 590, 70
    xmin, xmax = -0.03, 1.05
    ymin, ymax = -0.06, 1.0
    def X(v): return x0 + (v - xmin) * (x1 - x0) / (xmax - xmin)
    def Y(v): return y0 + (v - ymin) * (y1 - y0) / (ymax - ymin)

    p = []
    p.append(f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')

    # grid
    for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        p.append(f'  <line x1="{fmt(X(v))}" y1="{fmt(Y(ymin))}" x2="{fmt(X(v))}" y2="{y1}" stroke="{GRID}" stroke-width="1.5"/>')
        p.append(f'  <text x="{fmt(X(v))}" y="{fmt(Y(ymin)+30)}" text-anchor="middle" font-size="19" fill="{GREY}">{v:g}</text>')
        p.append(f'  <line x1="{x0}" y1="{fmt(Y(v))}" x2="{x1}" y2="{fmt(Y(v))}" stroke="{GRID}" stroke-width="1.5"/>')
        p.append(f'  <text x="{x0-14}" y="{fmt(Y(v)+6)}" text-anchor="end" font-size="19" fill="{GREY}">{v:g}</text>')
    # zero gain line slightly darker
    p.append(f'  <line x1="{x0}" y1="{fmt(Y(0))}" x2="{x1}" y2="{fmt(Y(0))}" stroke="{MUT}" stroke-width="1.5"/>')
    # axes
    p.append(f'  <line x1="{x0}" y1="{fmt(Y(ymin))}" x2="{x1}" y2="{fmt(Y(ymin))}" stroke="{GREY_D}" stroke-width="2"/>')
    p.append(f'  <line x1="{x0}" y1="{fmt(Y(ymin))}" x2="{x0}" y2="{y1-8}" stroke="{GREY_D}" stroke-width="2"/>')
    p.append(f'  <text x="{(x0+x1)//2}" y="{fmt(Y(ymin)+66)}" text-anchor="middle" font-size="21" fill="{INK}">perturbation footprint: reconstruction MSE on perturbed dimensions, before purification</text>')
    p.append(f'  <text x="52" y="{(y0+y1)//2}" text-anchor="middle" font-size="21" fill="{INK}" transform="rotate(-90 52 {(y0+y1)//2})">purification gain  &#916;P(adv)</text>')

    # regime annotations (drawn under the marks)
    p.append(f'  <text x="{fmt(X(0.05))}" y="{fmt(Y(0.17))}" font-size="19" font-style="italic" fill="{MUT}">no footprint:</text>')
    p.append(f'  <text x="{fmt(X(0.05))}" y="{fmt(Y(0.135))}" font-size="19" font-style="italic" fill="{MUT}">nothing to undo</text>')
    p.append(f'  <text x="{fmt(X(0.56))}" y="{fmt(Y(0.90))}" font-size="19" font-style="italic" fill="{MUT}">footprint erased,</text>')
    p.append(f'  <text x="{fmt(X(0.56))}" y="{fmt(Y(0.865))}" font-size="19" font-style="italic" fill="{MUT}">detection gap closed</text>')
    p.append(f'  <text x="{fmt(X(0.60))}" y="{fmt(Y(0.185))}" font-size="19" font-style="italic" fill="{MUT}">footprint, but detector near chance:</text>')
    p.append(f'  <text x="{fmt(X(0.60))}" y="{fmt(Y(0.15))}" font-size="19" font-style="italic" fill="{MUT}">no detection gap to close</text>')

    # marks at exact positions; coincident piles are annotated with their counts
    ann = []
    n_origin = sum(1 for _, _, mse, gain in pts if mse < 0.05 and abs(gain) < 0.02)
    n_right = sum(1 for _, _, mse, gain in pts if mse > 0.95 and abs(gain) < 0.02)
    for base, mode, mse, gain in sorted(pts, key=lambda r: (r[2], r[3])):
        xx, yy = X(mse), Y(gain)
        p.append("  " + mode_mark(mode, xx, yy))
        if base == "mirai" and mode == "iat":
            ann.append((xx, yy, "mirai (iat)", "left"))
        if base == "portscan" and mode == "both":
            ann.append((xx, yy, "portscan (both)", "left"))
        if base == "syn_ddos" and mode == "size":
            ann.append((xx, yy, "syn ddos (size)", "left"))

    for xx, yy, text, side in ann:
        p.append(f'  <line x1="{fmt(xx-14)}" y1="{fmt(yy-8)}" x2="{fmt(xx-36)}" y2="{fmt(yy-26)}" stroke="{MUT}" stroke-width="1.5"/>')
        p.append(f'  <text x="{fmt(xx-40)}" y="{fmt(yy-32)}" text-anchor="end" font-size="18" fill="{GREY}">{text}</text>')

    # pile counts (marks coincide exactly at these two spots); kept inside the plot
    # box so they never collide with the x tick labels below the axis
    p.append(f'  <text x="{fmt(X(0.05))}" y="{fmt(Y(0)-30)}" font-size="18" fill="{GREY}">{n_origin} source-mode pairs</text>')
    p.append(f'  <text x="{fmt(X(0.985))}" y="{fmt(Y(0)-30)}" text-anchor="end" font-size="18" fill="{GREY}">{n_right} pairs (reflection floods, size)</text>')

    # legend
    lx, ly = x0 + 20, y1 + 18
    p.append("  " + mode_mark("both", lx, ly))
    p.append(f'  <text x="{lx+20}" y="{ly+7}" font-size="20" fill="{INK}">both</text>')
    p.append("  " + mode_mark("iat", lx + 130, ly))
    p.append(f'  <text x="{lx+150}" y="{ly+7}" font-size="20" fill="{INK}">iat</text>')
    p.append("  " + mode_mark("size", lx + 240, ly))
    p.append(f'  <text x="{lx+260}" y="{ly+7}" font-size="20" fill="{INK}">size</text>')

    p.append('</svg>')
    open(os.path.join(OUT, "eval_purification_scatter.svg"), "w").write("\n".join(p) + "\n")
    print("wrote eval_purification_scatter.svg")

# --------------------------------------- fig D: evasion cost (RQ1) ----
def fig_evasion_cost():
    """ASR vs attacker spend for RF / TCN / stack, under one shared perturbation.

    Reads data/models/gradient_ablation/gradient_ablation_metrics.json (written by
    scripts/gradient_ablation/run_gradient_ablation.py). Skipped with a notice if the
    study has not been run, so the other three figures still build.
    """
    path = "data/models/gradient_ablation/gradient_ablation_metrics.json"
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        print(f"skipped eval_evasion_cost.svg - {path} not found (run "
              f"scripts/gradient_ablation/run_gradient_ablation.py)")
        return
    m = load(path)
    policy = m.get("report_policy", "fpr_0.05")
    budgets = m["asr"]["budgets"]
    over = m["asr"]["overall"].get("best") or m["asr"]["overall"]["pgd"]

    # x-axis: what the attacker actually paid, not the budget cap it was allowed.
    xvals = [m["spend"][b]["median_total_pad_bytes_per_flow"] for b in budgets]
    series = {t: [over[t][policy][b]["asr"] for b in budgets]
              for t in ("rf", "tcn", "stack") if t in over}

    W, H = 1240, 670
    x0, x1, y0, y1 = 150, 1010, 560, 70
    xmax = max(xvals) or 1.0
    def X(v):
        return x0 + (v / xmax) * (x1 - x0)
    def Y(v):
        return y0 + (v - 0.0) * (y1 - y0) / 1.0

    p = []
    p.append(f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')

    p.append('  <!-- grid and axes -->')
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        yy = Y(v)
        p.append(f'  <line x1="{x0}" y1="{fmt(yy)}" x2="{x1}" y2="{fmt(yy)}" stroke="{GRID}" stroke-width="1.5"/>')
        p.append(f'  <text x="{x0-14}" y="{fmt(yy+6)}" text-anchor="end" font-size="19" fill="{GREY}">{int(v*100)}%</text>')
    # x ticks: the first three budgets crowd the origin, so alternate their labels
    # onto a second row rather than letting them overprint each other
    for i, b in enumerate(budgets):
        xx = X(xvals[i])
        row = 34 if (i % 2 == 0 or xvals[i] > 200) else 56
        p.append(f'  <line x1="{fmt(xx)}" y1="{y0}" x2="{fmt(xx)}" y2="{y0+8}" stroke="{GREY_D}" stroke-width="1.5"/>')
        p.append(f'  <text x="{fmt(xx)}" y="{y0+row}" text-anchor="middle" font-size="18" fill="{GREY}">{int(round(xvals[i]))}</text>')
    p.append(f'  <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{GREY_D}" stroke-width="2"/>')
    p.append(f'  <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1-10}" stroke="{GREY_D}" stroke-width="2"/>')
    p.append(f'  <text x="{(x0+x1)//2}" y="{y0+92}" text-anchor="middle" font-size="21" fill="{INK}">attacker spend - median padding per flow (bytes)</text>')
    p.append(f'  <text x="46" y="{(y0+y1)//2}" text-anchor="middle" font-size="21" fill="{INK}" transform="rotate(-90 46 {(y0+y1)//2})">attack success rate</text>')

    def poly(vals, color):
        pts = " ".join(f"{fmt(X(xvals[i]))},{fmt(Y(v))}"
                       for i, v in enumerate(vals) if v is not None)
        return f'  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>'

    # Same glyph vocabulary as fig_streaming: TCN=circle, RF=square, meta/stack=diamond.
    p.append('  <!-- TCN: the attacker\'s own target (white-box) -->')
    p.append(poly(series["tcn"], BLUE))
    for i, v in enumerate(series["tcn"]):
        if v is not None:
            p.append("  " + circle(X(xvals[i]), Y(v), 8.5, BLUE, BLUE_D))

    p.append('  <!-- Random Forest: reached only by transfer, no gradient path -->')
    p.append(poly(series["rf"], GREY))
    for i, v in enumerate(series["rf"]):
        if v is not None:
            p.append("  " + square(X(xvals[i]), Y(v), 15, GREY, GREY_D))

    p.append('  <!-- fused stack -->')
    p.append(poly(series["stack"], TEAL))
    for i, v in enumerate(series["stack"]):
        if v is not None:
            p.append("  " + diamond(X(xvals[i]), Y(v), 10.5, TEAL, TEAL_D))

    lx = x1 + 18
    for name, key, color in (("TCN", "tcn", BLUE), ("RF", "rf", GREY),
                             ("stack", "stack", TEAL)):
        v = series[key][-1]
        if v is None:
            continue
        p.append(f'  <text x="{lx}" y="{fmt(Y(v)+6)}" font-size="21" font-weight="bold" fill="{color}">{name}  {v*100:.0f}%</text>')

    p.append(f'  <text x="{x0+16}" y="{y1+4}" font-size="18" fill="{GREY}">one perturbation, crafted on the TCN, scored on all three</text>')
    p.append(f'  <text x="{x0+16}" y="{y1+28}" font-size="18" fill="{MUT}">matched {policy}; denominator = flows the target detects unperturbed</text>')

    p.append('</svg>')
    open(os.path.join(OUT, "eval_evasion_cost.svg"), "w").write("\n".join(p) + "\n")
    print("wrote eval_evasion_cost.svg")

# ------------------------------------ fig E: deployment baseline (RQ2 e2e) ----
def fig_baseline():
    """NIDS attack-success rate before vs after DEPT, one row per scored source.

    Reads data/models/baseline/baseline_metrics.json (written by
    scripts/baseline/run_baseline.py). Each row is a dumbbell: the adversarial
    (before-DEPT) ASR in red, the DEPT-defended (after-DEPT) ASR in blue, joined by
    a line whose length is the attack success the pipeline removes; the clean-oracle
    ASR floor is a faint teal tick. Skipped with a notice if the study has not run.
    """
    path = "data/models/baseline/baseline_metrics.json"
    if not os.path.exists(os.path.join(ROOT, path)):
        write_placeholder("eval_baseline.svg",
                          "run scripts/baseline/run_baseline.py, then regenerate")
        return
    m = load(path)
    per = m.get("per_source", {})
    rows = []
    for src in sorted(per):
        a = per[src].get("asr", {})
        if a.get("adversarial") is None or a.get("defended") is None:
            continue
        rows.append((src, a.get("clean"), a["adversarial"], a["defended"]))
    if not rows:
        write_placeholder("eval_baseline.svg",
                          "baseline ran but no source had a defended ASR")
        return

    W = 1240
    top, bot, rh = 96, 70, 46
    H = top + bot + rh * len(rows)
    x0, x1 = 300, 1120
    def X(v):
        return x0 + max(0.0, min(1.0, v)) * (x1 - x0)

    p = []
    p.append(f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')

    # x grid + ticks (ASR 0..1)
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        xx = X(v)
        p.append(f'  <line x1="{fmt(xx)}" y1="{top-16}" x2="{fmt(xx)}" y2="{H-bot+8}" stroke="{GRID}" stroke-width="1.5"/>')
        p.append(f'  <text x="{fmt(xx)}" y="{H-bot+32}" text-anchor="middle" font-size="19" fill="{GREY}">{v:g}</text>')
    p.append(f'  <text x="{(x0+x1)//2}" y="{H-bot+62}" text-anchor="middle" font-size="21" fill="{INK}">NIDS attack success rate (1 - detection), per source, before vs after DEPT</text>')

    # alternating row bands + dumbbells
    for i, (src, clean, adv, defe) in enumerate(rows):
        yc = top + rh * i + rh // 2
        if i % 2 == 0:
            p.append(f'  <rect x="{x0-288}" y="{yc-rh//2}" width="{x1-x0+300}" height="{rh}" fill="#f6f8fa"/>')
        p.append(f'  <text x="{x0-20}" y="{yc+6}" text-anchor="end" font-size="19" fill="{INK}">{src.replace("_adv_", " ").replace("_", " ")}</text>')
        # connector: the ASR the pipeline removes
        p.append(f'  <line x1="{fmt(X(defe))}" y1="{yc}" x2="{fmt(X(adv))}" y2="{yc}" stroke="{MUT}" stroke-width="3"/>')
        if clean is not None:  # oracle floor tick
            p.append(f'  <line x1="{fmt(X(clean))}" y1="{yc-11}" x2="{fmt(X(clean))}" y2="{yc+11}" stroke="{TEAL_D}" stroke-width="2.5"/>')
        p.append("  " + circle(X(adv), yc, 8.5, RED, RED_D))     # before DEPT
        p.append("  " + circle(X(defe), yc, 8.5, BLUE, BLUE_D))  # after DEPT

    # legend
    lx, ly = x0, 44
    p.append("  " + circle(lx, ly, 8.5, RED, RED_D))
    p.append(f'  <text x="{lx+18}" y="{ly+7}" font-size="20" fill="{INK}">before DEPT (adversarial)</text>')
    p.append("  " + circle(lx + 340, ly, 8.5, BLUE, BLUE_D))
    p.append(f'  <text x="{lx+358}" y="{ly+7}" font-size="20" fill="{INK}">after DEPT (defended)</text>')
    p.append(f'  <line x1="{lx+640}" y1="{ly-11}" x2="{lx+640}" y2="{ly+11}" stroke="{TEAL_D}" stroke-width="2.5"/>')
    p.append(f'  <text x="{lx+652}" y="{ly+7}" font-size="20" fill="{INK}">clean oracle floor</text>')

    p.append('</svg>')
    open(os.path.join(OUT, "eval_baseline.svg"), "w").write("\n".join(p) + "\n")
    print("wrote eval_baseline.svg")

if __name__ == "__main__":
    fig_streaming()
    fig_strip()
    fig_scatter()
    fig_evasion_cost()
    fig_baseline()
