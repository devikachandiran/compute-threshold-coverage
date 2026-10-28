# How quickly do fixed compute thresholds lose track of capable AI models?

Code and data for my short research note of the same name (October 2026). The write-up is in [`writeup.pdf`](writeup.pdf).

Regulators use a fixed training-compute threshold (for example 10²⁵ FLOP in Article 51 of the EU AI Act) to decide which AI models get extra scrutiny. Algorithms keep getting more efficient, so later models can match earlier capability with less compute. This code estimates what share of capable models a fixed threshold catches each year, and how often the threshold would need updating to keep catching them.

## Main results (2025, central estimates)

- Frontier training compute grew about 5× per year (90% range 4.6–5.6×).
- The 10²⁵ FLOP threshold caught about 15% of capable models (5–95% range 10–24%), down from about 67% in 2023.
- Even if algorithmic progress were much slower (24-month halving time), it caught only about 42%.
- A threshold updated every 6 months kept about 89% coverage; one updated every 24 months kept about 51%.

## Method

1. **Data.** Epoch AI's *All AI Models* dataset, downloaded 25 October 2026. I keep language and multimodal models from 2018 onwards that have a training-compute estimate (822 models).
2. **Effective compute.** Each model's compute is converted to January-2023 "algorithm units":
   `C_eff = C_phys × 2^((t − 2023-01-01) / τ)`,
   where τ is the halving time of compute needed for fixed capability (median 8 months, 95% interval about 5–14 months; Ho et al. 2024).
3. **Capable models.** A model counts as capable if `C_eff ≥ 10²⁵ FLOP`, meaning at least as capable as a 10²⁵ FLOP model from January 2023.
4. **Coverage.** This is the share of capable models released in a given year whose *physical* compute is above the threshold.
5. **Uncertainty.** I run 2,000 Monte Carlo draws. Each draw samples τ from a lognormal distribution and adds noise to each model's compute based on Epoch's confidence label: Confident 0.15 dex, Likely 0.30 dex, Speculative 0.50 dex, other/missing 0.40 dex. These noise widths are my own assumption.

## Files

| File | What it does |
|---|---|
| `coverage_model.py` | Main analysis. Fits the frontier compute trend, simulates coverage for 10²⁴, 3×10²⁴ and 10²⁵ FLOP thresholds, and writes `results_summary.txt`, `fig1_compute_trend.png` and `fig2_coverage.png`. |
| `extensions.py` | Sensitivity to τ (Table 1), the detection vs false-alarm curve (`fig4_detection_vs_false_alarm.png`), the update-interval analysis (`fig5_update_interval.png`) and open-weight vs closed models. Writes `results_extensions.txt`. |
| `data/epoch_all_ai_models.csv` | Epoch AI dataset (CC BY), see below. |
| `writeup.pdf` | The research note. |

## How to run

```bash
pip install -r requirements.txt
python coverage_model.py
python extensions.py
```

Run both scripts from the repository root. Each takes about a minute. Random seeds are fixed, so the results should reproduce. Small differences across library versions are possible.

## Limitations

- "Capable" is defined by effective compute, not by benchmark or capability-evaluation scores.
- I assume algorithmic progress affects all models equally.
- Many compute figures in the dataset are Epoch's estimates, not developer-reported numbers.
- 2026 is a partial year, and unpublished models are missing from the dataset.

## Data and credit

Data: Epoch AI, *Data on AI Models*, https://epoch.ai/data/ai-models, accessed 25 October 2026. Licensed under CC BY. The data is redistributed here unchanged, with attribution. For the latest version, download it from Epoch AI.

## References

1. Ho, A., Besiroglu, T., Erdil, E., et al. (2024). *Algorithmic progress in language models*. NeurIPS 2024. arXiv:2403.05812.
2. European Union (2024). *Regulation (EU) 2024/1689 (Artificial Intelligence Act)*, Article 51.
3. Heim, L. & Koessler, L. (2024). *Training Compute Thresholds: Features and Functions in AI Regulation*. arXiv:2405.10799.
4. Hooker, S. (2024). *On the Limitations of Compute Thresholds as a Governance Strategy*. arXiv:2407.05694.

## Author

Devika Chandiran, independent researcher, Liverpool, UK.
