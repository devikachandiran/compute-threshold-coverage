"""
Extensions to coverage_model.py (ideas from D. Chandiran's notes, 27 Oct 2026):
  1. Sensitivity to the speed of algorithmic progress (halving time tau)
  2. False negatives / false positives across many thresholds (a GW-style
     "detection efficiency vs false alarm" curve)
  3. How often a threshold must be updated to keep catching capable models
  4. Open-weight vs closed models
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coverage_model import load, E_STAR, T_REF, CONF_SIGMA_DEX

RNG = np.random.default_rng(7)
N = 1000
LOG_E = np.log10(E_STAR)
lines = []


def draw(d, tau_months):
    logC = d["logC"].values + d["sigma"].values * RNG.standard_normal(len(d))
    alg = d["years_since_ref"].values / (tau_months / 12) * np.log10(2)
    return logC, logC + alg


def tau_sample():
    return np.exp(np.log(8) + np.log(14 / 8) / 1.96 * RNG.standard_normal())


d = load()
y25 = (d["year"] == 2025).values

# ---------- 1. sensitivity to halving time ----------
lines.append("1. Coverage of the strict rule (1e25) in 2025, by assumed halving time")
for tau in [5, 8, 14, 24]:
    cov = []
    for _ in range(N):
        logC, logE = draw(d, tau)
        tier = y25 & (logE >= LOG_E)
        if tier.sum():
            cov.append((tier & (logC >= 25)).sum() / tier.sum())
    p = np.percentile(cov, [5, 50, 95])
    lines.append(f"   halving time {tau:>2} months: coverage {p[1]:.2f} [{p[0]:.2f}-{p[2]:.2f}]")

# ---------- 2. false negatives / false positives across thresholds (2025) ----------
grid = np.arange(23.0, 25.51, 0.25)
eff, fa, n_caught = [], [], []
for lt in grid:
    e, f, c = [], [], []
    for _ in range(N):
        logC, logE = draw(d, tau_sample())
        tier = y25 & (logE >= LOG_E)
        caught = y25 & (logC >= lt)
        e.append((tier & caught).sum() / max(tier.sum(), 1))           # detection efficiency
        f.append((caught & ~tier).sum() / max(caught.sum(), 1))         # share of caught that are false alarms
        c.append(caught.sum())
    eff.append(np.percentile(e, [5, 50, 95])); fa.append(np.percentile(f, [5, 50, 95]))
    n_caught.append(np.median(c))
eff, fa = np.array(eff), np.array(fa)
lines.append("\n2. 2025, by threshold: detection (share of capable models caught) | "
             "false alarms (share of caught models that are below the capability tier) | models caught")
for i, lt in enumerate(grid):
    lines.append(f"   1e{lt:.2f}: detection {eff[i,1]:.2f} | false alarms {fa[i,1]:.2f} | caught {n_caught[i]:.0f}")

fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(grid, eff[:, 1] * 100, "o-", color="#2c3e50", lw=2.2, label="Capable models caught (detection)")
ax.fill_between(grid, eff[:, 0] * 100, eff[:, 2] * 100, color="#2c3e50", alpha=0.15)
ax.plot(grid, fa[:, 1] * 100, "s-", color="#c0392b", lw=2.2, label="Caught models that are not capable (false alarms)")
ax.fill_between(grid, fa[:, 0] * 100, fa[:, 2] * 100, color="#c0392b", alpha=0.15)
ax.set_xticks([23, 23.5, 24, 24.5, 25, 25.5])
ax.set_xticklabels(["10²³", "3×10²³", "10²⁴", "3×10²⁴",
                    "10²⁵", "3×10²⁵"])
ax.set_xlabel("Compute threshold (FLOP)"); ax.set_ylabel("Percent")
ax.set_ylim(0, 105); ax.grid(alpha=0.25)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.suptitle("Choosing a threshold: catching capable models vs false alarms (2025)",
             fontsize=12.5, fontweight="bold", x=0.02, ha="left")
ax.set_title("Like a detector: lower the threshold to catch more real signals, but accept more false alarms.",
             fontsize=9, color="#555", loc="left")
ax.legend(fontsize=9, loc="upper right", frameon=False)
fig.text(0.02, 0.01, "Data: Epoch AI (CC BY). 1,000 simulations per threshold.", fontsize=7.5, color="#777")
fig.tight_layout(rect=(0, 0.03, 1, 0.97)); fig.savefig("fig4_detection_vs_false_alarm.png", dpi=160)

# ---------- 3. how often must the threshold be updated? ----------
# At each reset date r, set the threshold so it exactly matches the capability tier at r:
#   T_r = E_STAR / A(r). Then measure coverage of tier models released in the next L months.
resets = pd.date_range("2023-01-01", "2025-07-01", freq="QS")
lags = [3, 6, 12, 18, 24]
res = {L: [] for L in lags}
dates = d["date"].values
for _ in range(400):
    tau = tau_sample()
    logC, logE = draw(d, tau)
    for r in resets:
        rr = (r - T_REF).days / 365.25
        logT = LOG_E - rr / (tau / 12) * np.log10(2)
        for L in lags:
            end = r + pd.DateOffset(months=L)
            if end > pd.Timestamp("2026-10-01"):
                continue
            win = (dates >= np.datetime64(r)) & (dates < np.datetime64(end))
            tier = win & (logE >= LOG_E)
            if tier.sum() >= 3:
                res[L].append((tier & (logC >= logT)).sum() / tier.sum())
lines.append("\n3. If the threshold is reset to match the capability tier, then left unchanged for L months:")
upd = []
for L in lags:
    p = np.percentile(res[L], [5, 50, 95]); upd.append(p)
    lines.append(f"   update every {L:>2} months: average coverage {p[1]:.2f} [{p[0]:.2f}-{p[2]:.2f}]")
upd = np.array(upd)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lags, upd[:, 1] * 100, "o-", color="#27ae60", lw=2.5, ms=8)
ax.fill_between(lags, upd[:, 0] * 100, upd[:, 2] * 100, color="#27ae60", alpha=0.15)
ax.axhline(90, color="grey", ls=":", lw=1); ax.text(lags[-1], 91.5, "90% coverage", ha="right", fontsize=8, color="grey")
ax.set_xticks(lags); ax.set_xlabel("Months between threshold updates")
ax.set_ylabel("Capable models caught"); ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="y", alpha=0.25)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.suptitle("How often does a compute threshold need updating?", fontsize=12.5, fontweight="bold", x=0.02, ha="left")
ax.set_title("Threshold reset to match the capability tier, then left unchanged. Shading = 5–95% range.",
             fontsize=9, color="#555", loc="left")
fig.text(0.02, 0.01, "Data: Epoch AI (CC BY). Resets every quarter, 2023–2025; 400 simulations.", fontsize=7.5, color="#777")
fig.tight_layout(rect=(0, 0.03, 1, 0.97)); fig.savefig("fig5_update_interval.png", dpi=160)

# ---------- 4. open-weight vs closed ----------
lines.append("\n4. Strict rule (1e25) coverage, open-weight vs closed models")
ow = d["Open model weights?"].astype(str).str.lower()
groups = {"open weights": ow.eq("yes").values, "closed": ow.eq("no").values}
for name, g in groups.items():
    for yr in [2024, 2025]:
        m = g & (d["year"] == yr).values
        cov, nt = [], []
        for _ in range(N):
            logC, logE = draw(d, tau_sample())
            tier = m & (logE >= LOG_E)
            nt.append(tier.sum())
            if tier.sum():
                cov.append((tier & (logC >= 25)).sum() / tier.sum())
        p = np.percentile(cov, [5, 50, 95]) if cov else [np.nan] * 3
        lines.append(f"   {name:<12} {yr}: capable models {np.median(nt):.0f} | coverage {p[1]:.2f} [{p[0]:.2f}-{p[2]:.2f}]")

txt = "\n".join(lines)
print(txt)
open("results_extensions.txt", "w").write(txt + "\n")
