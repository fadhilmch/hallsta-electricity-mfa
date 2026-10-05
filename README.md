# hallsta-mfa

Electricity Material Flow Analysis (energy + exergy) of the Holmen Hallsta paper mill, 2023, with Monte Carlo uncertainty. Made for a Material Flow Analysis course.

Start with `notebooks/hallsta_electricity_mfa.ipynb`. It explains every step: intuition first, then the formula, then the code, with a glossary at the top. GitHub renders it with all outputs, so you can read it without running anything.

## Run it

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if you do not have uv
uv sync                                          # installs Python 3.12 and every dependency
uv run jupyter lab notebooks/hallsta_electricity_mfa.ipynb
uv run pytest                                    # tests for the model
```

Re-execute and save the notebook from the command line:

```bash
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace hallsta_electricity_mfa.ipynb
```

The static Sankey picture needs Chrome or Chromium installed (used by kaleido). Without it, comment out the `write_image` lines and use `fig_E.show()`.

## Layout

| Path | What |
|---|---|
| `notebooks/hallsta_electricity_mfa.ipynb` | The analysis, step by step |
| `src/hallsta_mfa/model.py` | The model: base inputs, uncertainty ranges, exergy maths, `run()` |
| `tests/test_model.py` | Balance checks, exergy checks, headline numbers |
| `notebooks/outputs/` | CSV and PNG results written by the notebook |

## Sources and data

Numbers come from Holmen Paper AB, *Miljorapport 2023 Hallsta pappersbruk* (MR2023), and Holmen's answers to student questions (2019, 2025). Only the numbers needed for the calculation are included, each with a reference. The reports themselves are not in this repo.
