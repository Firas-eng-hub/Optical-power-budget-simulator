# Project Summary

This project is a Streamlit-based optical link simulator for coursework on fiber-optic transmission.

## Current Scope

- Power budget simulation
- Dispersion simulation
- Maximum link length estimation
- Link schema visualization
- Power-vs-distance plot
- Dispersion-vs-distance plot

## Main Files

- `app.py`: Streamlit interface and plots
- `physics/power_budget.py`: power-loss chain, receiver sensitivity, received power, margin
- `physics/dispersion.py`: chromatic/modal dispersion and bandwidth computation
- `physics/link_length.py`: maximum link length from power and dispersion constraints
- `utils/units.py`: unit helpers

## What Was Added

- Support for source presets:
  - `LED`
  - `LASER`
- Support for fiber presets:
  - multimode step-index
  - multimode graded-index
  - single-mode
- Support for detector presets:
  - `PIN`
  - `PIIPN`
  - manual sensitivity entry
- Reel-based splice counting for fibers delivered by rolls
- Vendor bandwidth-distance constraint for fibers
- Extra lumped losses through `Autres pertes (dB)`
- Editable attenuation `α` for all fiber types
- Editable source-fiber and fiber-detector coupler losses
- Editable transmitted power `Pe` and spectral width `Δλ` for both `LED` and `LASER`

## What Was Removed

- Sellmeier equation display
- Refractive-index tab
- Refractive-index summary table
- User inputs for `n1` and `n2`
- The `Utiliser le catalogue de l'exercice` toggle

## Physics Assumptions Used

- Receiver sensitivity:
  - `PIN = -52 dBm`
  - `PIIPN = -64 dBm`
- Fiber vendor bandwidth-distance products:
  - step-index: `100 MHz @ 100 m` -> `10 MHz·km`
  - graded-index: `100 MHz @ 1 km` -> `100 MHz·km`
- Internal modal-dispersion constants are still kept in code:
  - `DEFAULT_N1 = 1.468`
  - `DEFAULT_DELTA = 0.01`

These are internal only and are no longer exposed in the UI.

## Exercise Result Already Verified

For the 12 km, 2 Mbit/s case with:

- 2 connectors at `1 dB` each
- 11 splices at `0.3 dB` each

the acceptable combinations found were:

- `LED + graded-index + PIIPN`
- `LASER + graded-index + PIN`
- `LASER + graded-index + PIIPN`

The strongest recommended solution was:

- `LASER + fibre à gradient d'indice + PIIPN`

## Verification Done

- `py_compile` checks were run on the edited Python files successfully.
