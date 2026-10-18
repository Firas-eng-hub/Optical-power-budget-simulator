# Optical Power Budget Simulator

Interactive fiber-optic link simulator for optical transmission coursework. Configure a transmitter, fiber, connections and receiver to explore power loss, pulse broadening, available bandwidth and estimated maximum reach.

**[Open the live simulator](https://optical-power-budget-simulator.vercel.app)** · [Source code](https://github.com/Firas-eng-hub/Optical-power-budget-simulator)

## Features

- French-language interface with instant calculations and responsive layout.
- LED / LASER sources and 850, 1300, 1550 nm or custom wavelengths.
- Step-index multimode, graded-index multimode and single-mode fibers.
- Editable attenuation, spectral width, chromatic dispersion, coupler and connection losses.
- Manual splice counts, splice spacing or reel-based splice counts.
- PIN, PIIPN or manual receiver sensitivity.
- Power budget, chromatic/modal dispersion, bandwidth and maximum-length estimates.
- Link diagram, power-vs-distance curves with optional connection drops, and bandwidth-vs-distance chart.

## Run locally

### Original Python / Streamlit application

Use Python 3.10 or newer:

```sh
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

### Browser application

The Vercel version runs entirely in the browser and requires no Python server, API keys or environment variables.

```sh
python -m http.server 8000 --directory site
```

Open `http://localhost:8000`. The browser model is a JavaScript port of the Python calculations; the original Streamlit app remains available independently.

## Model and units

- Received power: `Pr = Pe − αL − NcLc − NsLs − source coupling − receiver coupling − other losses`.
- Power margin: `Pr − receiver sensitivity`.
- Chromatic broadening: `|Dch| × Δλ × L`, in ps.
- Total broadening: root-sum-square of chromatic and modal broadening.
- Model bandwidth: `700 / total broadening (ps)`, in GHz.
- Available bandwidth: minimum of model and catalog bandwidth.
- Estimated maximum reach: minimum of power, chromatic dispersion and catalog bandwidth-distance limits.

Inputs use km, nm, dB/dBm, ps/nm/km and GHz. A bandwidth of `0.002 GHz` equals `2 MHz`.

### Assumptions and limitations

This is a teaching model. PIN and PIIPN sensitivities are fixed at −52 dBm and −64 dBm. Catalog bandwidth-distance products are 10 MHz·km for step-index fiber and 100 MHz·km for graded-index fiber. Modal calculations use `n₁ = 1.468` and `Δ = 0.01`. Wavelength presets set chromatic dispersion; attenuation comes from the selected fiber preset and remains editable.

The maximum-length estimator preserves the Python model: it includes chromatic dispersion and catalog limits, but **does not include modal broadening**. It also holds connector/splice losses fixed at the configured link's counts when estimating reach. Check the computed bandwidth and margin at the actual proposed length. The browser's overall status requires both nonnegative power margin and bandwidth at least equal to the target; the original Streamlit power badge checks power only.

Connection positions in the stepped plot are uniformly distributed for illustration. The browser bandwidth chart caps high values for readability. Ideal zero-dispersion single-mode inputs can produce an infinite model bandwidth; this represents a model limit rather than a physical prediction.

## Project structure

```text
app.py                    Original Streamlit interface
physics/                  Python power, dispersion and reach models
utils/                    Unit helpers
site/index.html           Browser interface
site/styles.css           Responsive styling
site/model.js             Browser calculation model
site/app.js               Controls, results and SVG charts
tests/test_browser_parity.py  Cross-language calculation checks
vercel.json               Static deployment configuration
requirements.txt          Python application dependencies
```

## Verification

With Python dependencies and Node.js installed:

```sh
python tests/test_browser_parity.py
```

This compares browser results against the Python models across source, fiber, receiver and splice-count modes, including zero-dispersion cases.

## Deploy to Vercel

Import this GitHub repository into Vercel. Choose the **Other** framework preset, leave the root directory at the repository root, set the output directory to **site**, and leave build/install commands empty. These settings are supplied by `vercel.json`.

The hosted application is static: Vercel serves `site/`, while all simulation calculations happen locally in the browser. The Streamlit process is not deployed to Vercel. With Git integration connected, pushes to `main` trigger production deployments.
