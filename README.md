# FermiSunPy
Classes and methods relative to the Solar analysis with Fermi data

## sunpos.py
Knowing the position of the Sun (J2000) at a Given Misson elapsed Time.
Signature:
```bash
usage: sunpos.py [-h] [met]

Compute the Sun position for a given MET time.

positional arguments:
  met         Mission Elapsed Time (MET) in seconds (default: 376358403.000)

optional arguments:
  -h, --help  show this help message and exit
```

## suncenter.py

This routine converts a FT1 and FT2 file congtaining LAT events and pointing history into a "sun center" list of events and into a "sun center" reference frame of the LAT satellite.
FT1 anbd FT2 data can be obtained bhy the [Fermi LAT Science Support Center](https://fermi.gsfc.nasa.gov/ssc/data/]). Note bthat the selection must be large enough to inlude all the Sun trajectory. This implies that, if you want to convert large interval of time, you should consider using the provided [weekly all-sky files](https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3table.pl?tablehead=name=fermilweek&Action=More+Options}).

```bash
usage: suncenter.py [-h] [-ft1 ft1] [-ft2 ft2]

Convert FT1/FT2 files to Sun-centered coordinates.

optional arguments:
  -h, --help  show this help message and exit
  -ft1 ft1    Input FT1 file to convert.
  -ft2 ft2    Input FT2 file to convert.
  --legacy    Use the original event-by-event loop (slow).
  ```
The default conversion is vectorized (`convertFT1_fast` / `convertFT2_fast`); it agrees with the original loop to
better than 0.1 arcsec.

To visualize the newly genberated FT1 file (it is saved with `_sun.fit` extension), you can use the standard fermitools, using the center of the ROI as 0,0 in CEL coordinate, For example:
```bash
>gtbin                                                                                          
This is gtbin version HEAD
Type of output file (CCUBE|CMAP|LC|PHA1|PHA2|HEALPIX) [] CMAP 
Event data file name[] NAME_OF_THE_FT1_SUN_CENTERD_FILE
Output file name[] OUTPUT_NAME_FOR_CMAP 
Spacecraft data file name[] NAME_OF_THE_FT1_SUN_CENTERD_FILE
Size of the X axis in pixels[] 101
Size of the Y axis in pixels[] 101
Image scale (in degrees/pixel)[] 0.2
Coordinate system (CEL - celestial, GAL -galactic) (CEL|GAL) [] CEL
First coordinate of image center in degrees (RA or galactic l)[] 0
Second coordinate of image center in degrees (DEC or galactic b)[] 0
Rotation angle of image axis, in degrees[0] 0
Projection method e.g. AIT|ARC|CAR|GLS|MER|NCP|SIN|STG|TAN:[] CAR
```
the output file can be visualized with standard visualization tools, such as fv, ds9 or with python code (astropy,...)

## Solar-flare analysis console (GUI) and pipeline

A local web console that sets up and runs a complete Fermi analysis of a solar flare (GOES + GBM + LLE + LAT)
and produces an HTML report. It generalizes the scripts of the `solarflares` repository
(`SunModel.py`, `SolarTemplates.py`, `timeresolved_analysis.py`, `spectral_analysis.py`, ...).

```bash
conda activate FermiSunPy        # needs fermitools, threeML, astromodels and sunpy (pip install "sunpy[all]")
./gui/start_gui.sh               # serves http://localhost:8766
```

1. **Flare & data**: pick a day, *Find flares* lists the GOES flares (HEK, via sunpy) and the GBM triggers;
   *Use* fills T0 (GOES start), end, peak and class. *Download data* fetches into `flares/SOLYYYYMMDD_HHMM/`:
   GOES XRS (sunpy `Fido`), GBM daily CSPEC for the 14 detectors with the **Sun-pointed RSP2** responses from
   hesperia (merged over the window), LLE files of GBM triggers in the window (if any), LAT FT1/FT2 within 15°
   of the Sun, the LAT Sun visibility windows, and optionally **Sun-centred** FT1/FT2 (`suncenter.py`).
2. **GBM detectors**: sorted by the Sun angle; the two NaI and the BGO facing the Sun are pre-selected.
3. **Light curves & intervals**: GOES, GBM NaI/BGO bands, LLE, LAT photons and the Sun off-axis/zenith angles
   on a common UTC axis. Drag to define GBM/LLE backgrounds, joint spectral-fit intervals, the LAT localization
   interval, the LAT time-integrated interval and custom LAT bins. *Bayesian blocks* (with p0) fills the custom LAT
   bins from the Sun photons of the time-integrated interval (same selection as the LAT step, split at the
   visibility windows) and draws each bin's photon rate on the LAT panel; the bins can then be edited.
4. **LAT analysis**: event class, ROI, zenith/θ cuts, energy range and flux band, Sun-centred frame, localization
   (gtfindsrc + TS map), time-resolved binning (visibility windows, Bayesian blocks, N photons, custom) and LAT
   models (pion-decay template, power law, cutoff power law).
5. **Spectral models**: build models from components — `pow_le`, `exp_co_le`, `pow_he`, `exp_co_he`
   (bremsstrahlung continua), `nar`, `brd` (narrow/broad nuclear de-excitation line templates), `g_2p2MeV`,
   `g_511keV`, `g_1p7MeV` (lines), `pion` (pion-decay template) — e.g. `exp_co_le+nar+brd+g_2p2MeV+pion`.
   *Parameters* (on each model) opens a table of its parameters: start value, min, max and fixed, for every model with
   that component or for this model only (this model wins). The LAT card has the same table for the LAT models.
   Template indices cannot leave their grids (pion s 2–6, nar/brd s 2.6–4.2). *Start each interval from the best fit of
   the previous one* chains the joint-fit intervals (per model) and the LAT time-resolved bins (from the previous bin
   with TS above the detection threshold); otherwise every interval starts from the configured values.
6. **Run & report**: writes `config_<flare>.json`, runs the steps and builds `results/report.html`.

**Reproducing a setup:** `config_<flare>.json` holds every setting, interval, model and the chosen run steps, plus
the flare definition used for the download (T0, end, peak, class, window, Sun-centred files, local FT1/FT2/LLE
paths) and the software versions. *Download configuration* saves it; *Load configuration…* restores it: for a flare
already on disk it applies the settings (review, then Save or Run); otherwise it fills the New-flare form and applies
the configuration once *Download data* finishes. `python -m sunflare.run_analysis <config>` runs the saved steps.

Pipeline steps (also usable without the GUI):

| step | module | output (`flares/<id>/results/`) |
|---|---|---|
| download | `python -m sunflare.fetch_data --t0 "2025-11-11 09:49:00" --tend ... --peak ... [--sun-centered] [--lat-ft1 F --lat-ft2 F] [--lle-dir D]` | `flare_info.json`, data |
| light curves | `python -m sunflare.lightcurves <config>` | `lightcurves/overview.png, zoom.png` |
| localization | `python -m sunflare.localization <config>` | `localization/tsmap.png, localization.json` (offset from the Sun centre, helioprojective) |
| LAT likelihood | `python -m sunflare.lat_analysis <config>` | `lat/lat_summary.json, lat_lightcurve.png` (flux, proton index, TS, ULs, source-probability events) |
| joint fits | `python -m sunflare.spectral <config>` | `spectrum_<t0>-<t1>/summary.json, vfv.png, counts_<model>.png` (BIC ranking, TS of added components) |
| report | `python -m sunflare.make_report <config>` | `report.html` |
| all | `python -m sunflare.run_analysis <config> --steps lightcurves,localization,lat,spectral,report` | |

`--lat-ft1/--lat-ft2` and `--lle-dir` import files already on disk (e.g. weekly files, or when the LAT data server
is down). The pion/NAR/BRD astromodels templates are built on first use from `solarflares/Sun/data`
(override with `SUNFLARE_TEMPLATES`); flare data go to `flares/` (override with `SUNFLARE_DATA`).

---
For question and comments: [nicola.omodei@stanford.edu](mailto:nicola.omodei@stanford.edu)