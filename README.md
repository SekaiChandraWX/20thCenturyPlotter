# 20CR Global Plotter — Dewpoint/MSLP & 500/850 mb Winds

A simple, fast Streamlit app that fetches **NOAA/PSL 20th Century Reanalysis (20CR)** hourly composites,
downloads the generated **NetCDF** files, and renders:
- **Dewpoint (°F) fill + MSLP (isobars) + 10 m wind barbs**
- **500 mb** wind speed (kt) & height (dam) with barbs
- **850 mb** wind speed (kt) & height (dam) with barbs

The app is **global & dateline-safe**, with presets for common basins (N. Atlantic, WPAC, etc.) and regions.

## Features
- NARR-matched aesthetics (line widths, labeling, ~1.5° barb density target).
- IDL-robust **PlateCarree(central_longitude=...)** projection.
- Automatic stride/tick/contour density based on zoom level.
- On-the-fly cleanup of temporary NetCDFs.

## How it works
The app mirrors NOAA/PSL’s composite workflow:
1. Hits the PSL composite HTML endpoint for 20CR variables (e.g., `Vector Wind`, `Geopotential Height`).
2. Parses the page to find the **downloadable NetCDF** link.
3. Reads the data with **netCDF4** and plots with **matplotlib + cartopy**.

> No CDS key is required; this app does **not** use ERA5. It uses only PSL 20CR composite pages.

## Quickstart

### Run locally
```bash
# (Recommended) In a fresh environment:
pip install -r requirements.txt

streamlit run app.py
