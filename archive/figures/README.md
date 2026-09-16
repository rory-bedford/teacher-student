# Figures

Paper figure SVGs, the minimal data behind them, and the scripts that rebuild them.

```bash
uv run python figures/export_data.py    # runs/ caches -> figures/data/<name>.csv (runs/ is only read)
uv run python figures/make_figures.py   # figures/data/<name>.csv -> figures/<name>.svg
uv run python figures/make_figures.py raster --out-dir /tmp/check   # subset / elsewhere
```

## Naming scheme

```
ff-<known|learnt>__obs-<pct>__recur-<pct>__wn-<pct>__<panel>.<ext>
```

| Factor | Meaning |
|---|---|
| `ff-known` / `ff-learnt` | feedforward connections given to the student / inferred |
| `obs-` | % of recurrent neurons observed (activity in the loss) |
| `recur-` | % of recurrent connections reconstructed |
| `wn-` | weight-noise % (`noise_frac`) |
| panel | `scatter`, `raster`, `bar`, `curve`, `schematic` |

Prefixes: `controls__` means a comparison across control conditions. `sweep-<factor>__` means the figure varies
that factor, so it is left out of the rest of the name.

Unless stated otherwise, `obs` and `recur` are percentages of the 5000-neuron teacher. Removing a fraction
*f* of neurons from the student removes every connection to or from them, so
`recur` = (1 − *f*)², e.g. 10% removed → 81%. That figure was checked against the saved masks.

## Figures → run configs

Run configs are the `parameters.toml` / `experiment.toml` snapshots in each run directory under `runs/`.
The run directory names still use the old "hidden" / "no-hidden" / "full-inference" terms.

| Figure | Run config(s) | Key parameters |
|---|---|---|
| `ff-learnt__obs-45__recur-81__wn-30__scatter` | `full-inference/hidden/surrgrad-10/` | `ff_rank = 1` (FF learnt), `missing_unit_fraction = 0.1`, `hidden_fraction = 0.5` (2250 of 5000 observed), `noise_frac = 0.3` |
| `controls__ff-learnt__obs-90__recur-81__wn-mixed__bar` | `full-inference/no-hidden/{baseline, no-connectome-control, shuffle-weights-control, shuffle-control}/` | `ff_rank = 10`, `missing_unit_fraction = 0.1`, all 4500 student neurons observed; `noise_frac` = 0.2 (Full Connectome, Shuffle-Connections) or 0.3 (Learnt-Recurrence, Shuffle-Weights) |
| `ff-known__obs-10__recur-100__wn-0__scatter` | `hidden-activity/increasing-hidden-fraction/hidden-frac-0.9/` | `hidden_cell_fraction = 0.9`, `optimisable = "scaling_factors"` |
| `ff-known__obs-10__recur-100__wn-0__raster` | same | unobserved neurons 2433, 2246, 1316 (top to bottom), 0–2 s |
| `sweep-recur__ff-known__obs-100__wn-0__curve` | `hidden-units/increasing-hidden-fraction/hidden-{0.05..0.50}/` | `hidden_unit_fraction` = removed neuron fraction (x-axis). `obs-100` is relative to the student: every remaining neuron is in the loss |
| `sweep-wn__ff-known__obs-100__recur-100__curve` | `noisy-weights/varying-noise/noise-{0.05..0.50}/` | `noise_frac` (x-axis) |
| `panel-{a..f}`, `title-*` | — | text only |

## Provenance notes

- The original SVGs (2026-07-04) came from notebook cells that were never saved. The sources above were confirmed by
  recomputing the R² values printed on each figure, and for the raster by matching every spike time. The styling is a close
  reconstruction.
- **Shuffle controls are swapped in the run directory names.** `shuffle-control/` has `shuffle_weights = true`
  (wiring kept, weight values permuted); `shuffle-weights-control/` has `shuffle_connectome = true` (wiring permuted).
  The bar labels follow the parameters, so they are the reverse of the original figure's labels.
- The two sweeps were a single figure with twin x-axes (`increasing-hidden-units-and-weight-noise-r2.svg`). They are now
  split into one file per sweep. Both keep the y range 0.5–1.0 so they can be compared.
- `noisy-weights/varying-noise/experiment.toml` has `output_dir` set to `noisy-weights/test-sgd/`, but the grid results
  are in `noisy-weights/varying-noise/`. That's why `export_data.py` lists the run dirs directly.
- Colours come from `FIGURE_*` in `connectome_snns.visualization`.
