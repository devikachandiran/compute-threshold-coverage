"""
How well does a fixed training-compute threshold keep covering capable AI models
as algorithms get more efficient?

Data: Epoch AI, "All AI Models" dataset (CC BY), downloaded 25 Oct 2026.

Idea
----
A compute threshold (e.g. 1e25 FLOP) regulates models by *physical* training compute.
But algorithmic progress means the same capability needs less compute over time.
We convert each model's physical compute into *effective compute* (compute measured
in 2023-01 "algorithm units"):

    C_eff = C_phys * 2 ** ((t - t_ref) / tau)

where tau is the halving time of compute needed for a fixed capability.

A "capability tier" is the set of models with C_eff >= E_star.
Coverage(year) = share of tier models whose *physical* compute is above the threshold.

Uncertainty (Monte Carlo):
  * tau ~ lognormal, median 8 months, 95% interval ~5-14 months (Ho et al. 2024)
  * each model's compute ~ lognormal around Epoch's estimate, width set by Epoch's
    "Confidence" label (an assumption; see README)

Rules compared: fixed thresholds of 1e24, 3e24 and 1e25 FLOP.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
N_DRAWS = 2000
T_REF = pd.Timestamp("2023-01-01")
THRESHOLD = 1e25          # physical FLOP threshold under study
E_STAR = 1e25             # capability tier, in 2023-01 effective FLOP
YEARS = list(range(2023, 2027))
THRESHOLDS = [1e24, 3e24, 1e25]   # fixed physical-FLOP thresholds compared

# --- assumptions (illustrative; documented in README) ---
TAU_MEDIAN_MONTHS = 8.0
TAU_LOG_SIGMA = np.log(14 / 8) / 1.96          # 95% interval ~5-14 months (Ho et al. 2024)
CONF_SIGMA_DEX = {"Confident": 0.15, "Likely": 0.30, "Speculative": 0.50}
DEFAULT_SIGMA_DEX = 0.40                          # "Unknown" or missing label


def load(path="data/epoch_all_ai_models.csv"):
    d = pd.read_csv(path, low_memory=False)
    d["date"] = pd.to_datetime(d["Publication date"], errors="coerce")
    d = d[d["Training compute (FLOP)"].notna() & (d["date"] >= "2018-01-01")]
    d = d[d["Domain"].str.contains("Language|Multimodal", na=False)].copy()
    d["logC"] = np.log10(d["Training compute (FLOP)"].astype(float))
    d["years_since_ref"] = (d["date"] - T_REF).dt.days / 365.25
    d["year"] = d["date"].dt.year
    d["sigma"] = d["Confidence"].map(CONF_SIGMA_DEX).fillna(DEFAULT_SIGMA_DEX)
    return d.reset_index(drop=True)


def frontier_trend(d):
    """Growth of frontier compute: fit log10(C) vs time on Epoch's 'Frontier model' flag."""
    f = d[d["Frontier model"] == True]
    x, y = f["years_since_ref"].values, f["logC"].values
    slopes = []
    for _ in range(2000):
        i = RNG.integers(0, len(x), len(x))
        slopes.append(np.polyfit(x[i], y[i], 1)[0])
    s = np.array(slopes)
    m, b = np.polyfit(x, y, 1)
    return f, m, b, np.percentile(10 ** s, [5, 50, 95])


def simulate(d):
    """Per-draw coverage of the capability tier, and number of models caught, for each fixed threshold."""
    n = len(d)
    cov = {T: np.full((N_DRAWS, len(YEARS)), np.nan) for T in THRESHOLDS}
    caught = {T: np.zeros((N_DRAWS, len(YEARS))) for T in THRESHOLDS}
    tier_n = np.zeros((N_DRAWS, len(YEARS)))
    yrs = d["year"].values
    t = d["years_since_ref"].values
    for k in range(N_DRAWS):
        tau_yr = np.exp(np.log(TAU_MEDIAN_MONTHS) + TAU_LOG_SIGMA * RNG.standard_normal()) / 12
        logC = d["logC"].values + d["sigma"].values * RNG.standard_normal(n)
        log_eff = logC + t / tau_yr * np.log10(2)
        in_tier = log_eff >= np.log10(E_STAR)
        for j, y in enumerate(YEARS):
            m = yrs == y
            tier = m & in_tier
            tier_n[k, j] = tier.sum()
            for T in THRESHOLDS:
                above = logC >= np.log10(T)
                caught[T][k, j] = (m & above).sum()
                if tier.sum() > 0:
                    cov[T][k, j] = (tier & above).sum() / tier.sum()
    return cov, caught, tier_n


def summarise(a):
    return np.nanpercentile(a, [5, 50, 95], axis=0)


def main():
    d = load()
    f, m, b, growth = frontier_trend(d)
    cov, caught, tier_n = simulate(d)
    tn = summarise(tier_n)
    lines = [f"Models used (language/multimodal, 2018+, with compute): {len(d)}",
             f"Frontier-flagged models used for trend: {len(f)}",
             f"Frontier compute growth per year: {growth[1]:.1f}x "
             f"(90% bootstrap: {growth[0]:.1f}x to {growth[2]:.1f}x)",
             "Capability tier: effective compute >= 1e25 FLOP in 2023-01 algorithm units",
             "2026 is a partial year (data to 23 Oct 2026).", ""]
    for T in THRESHOLDS:
        c, g = summarise(cov[T]), summarise(caught[T])
        lines.append(f"Fixed threshold {T:.0e} FLOP")
        for j, y in enumerate(YEARS):
            lines.append(f"  {y}: tier models {tn[1,j]:.0f} | coverage {c[1,j]:.2f} "
                         f"[{c[0,j]:.2f}-{c[2,j]:.2f}] | models caught {g[1,j]:.0f}")
    txt = "\n".join(lines)
    print(txt)
    open("results_summary.txt", "w").write(txt + "\n")

    # --- Figure 1: compute over time with threshold and frontier trend ---
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.scatter(d["date"], d["Training compute (FLOP)"], s=9, alpha=0.35, color="#7a8ca5",
               label="Language and multimodal models")
    ax.scatter(f["date"], f["Training compute (FLOP)"], s=22, color="#c0392b",
               label="Frontier models (as flagged by Epoch)")
    xs = np.linspace(d["years_since_ref"].min(), d["years_since_ref"].max(), 50)
    ax.plot(T_REF + pd.to_timedelta(xs * 365.25, unit="D"), 10 ** (m * xs + b),
            color="#c0392b", lw=2, label=f"Frontier trend: about {growth[1]:.0f}\u00d7 per year")
    ax.axhline(THRESHOLD, color="black", ls="--", lw=1.2)
    ax.text(pd.Timestamp("2018-02-01"), THRESHOLD * 1.6, "EU AI Act threshold: 10\u00b2\u2075 FLOP",
            fontsize=9, fontweight="bold")
    ax.set_yscale("log"); ax.set_ylabel("Training compute (FLOP, log scale)")
    ax.grid(axis="y", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.suptitle("Frontier training compute has grown about 5\u00d7 per year",
                 fontsize=13, fontweight="bold", x=0.02, y=0.985, ha="left")
    ax.set_title("Each dot is one AI model. Red dots are the most compute-intensive models of their time.",
                 fontsize=9, color="#555", loc="left")
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    fig.text(0.02, 0.01, "Data: Epoch AI, All AI Models dataset (CC BY). Language/multimodal models, 2018 onwards, with compute estimates.",
             fontsize=7.5, color="#777")
    fig.tight_layout(rect=(0, 0.03, 1, 0.995))
    fig.savefig("fig1_compute_trend.png", dpi=160)

    # --- Figure 2: coverage, plain-language version ---
    cols = {1e24: "#27ae60", 3e24: "#e67e22", 1e25: "#2c3e50"}
    names = {1e25: "Strict rule: 10\u00b2\u2075 FLOP",
             3e24: "Middle rule: 3\u00d710\u00b2\u2074 FLOP",
             1e24: "Loose rule: 10\u00b2\u2074 FLOP"}
    full = [y for y in YEARS if y <= 2025]
    jf = [YEARS.index(y) for y in full]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for T in [1e24, 3e24, 1e25]:
        c = summarise(cov[T]) * 100
        ax.fill_between(full, c[0, jf], c[2, jf], color=cols[T], alpha=0.13, lw=0)
        ax.plot(full, c[1, jf], marker="o", ms=8, lw=2.5, color=cols[T])
        # 2026 (partial year) shown faintly and dashed
        ax.plot([2025, 2026], c[1, [YEARS.index(2025), YEARS.index(2026)]],
                ls=":", lw=1.5, color=cols[T], alpha=0.5)
        ax.plot(2026, c[1, YEARS.index(2026)], "o", ms=6, mfc="white", mec=cols[T], alpha=0.7)
        ax.text(2026.1, c[1, YEARS.index(2026)], names[T], color=cols[T],
                fontsize=10, fontweight="bold", va="center")
    j25 = YEARS.index(2025)
    n25 = summarise(tier_n)[1, j25]
    for T, dy in [(1e25, -9), (3e24, -9), (1e24, -9)]:
        c25 = summarise(cov[T])[1, j25]
        ax.annotate(f"\u2248{c25*n25:.0f} of {n25:.0f}", (2025, c25 * 100),
                    xytext=(-6, dy), textcoords="offset points", ha="right",
                    fontsize=9, color=cols[T])
    ax.set_ylim(0, 105); ax.set_xlim(2022.8, 2027.75)
    ax.set_xticks(YEARS); ax.set_xticklabels(["2023", "2024", "2025", "2026\n(partial year)"])
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Capable AI models caught by the rule")
    ax.grid(axis="y", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.suptitle("A fixed compute rule catches fewer capable AI models each year",
                 fontsize=13, fontweight="bold", x=0.02, y=0.985, ha="left")
    ax.set_title("'Capable' = at least as capable as a top 10\u00b2\u2075 FLOP model from 2023. "
                 "Shading = range across 2,000 simulations.",
                 fontsize=9, color="#555", loc="left")
    fig.text(0.02, 0.01, "Data: Epoch AI, All AI Models dataset (CC BY). "
             "Language/multimodal models with compute estimates.", fontsize=7.5, color="#777")
    fig.tight_layout(rect=(0, 0.03, 1, 0.995))
    fig.savefig("fig2_coverage.png", dpi=160)

if __name__ == "__main__":
    main()
