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

The solar-flare analysis console (GOES + GBM + LLE + LAT pipeline, GUI and interactive report) now lives in the
[fermiconsole](https://github.com/omodei/fermiconsole) repository, as its `sun` flavor. It uses `suncenter.py` and
`sunpos.py` from this repository and keeps reading and writing the flare folders in `flares/` here.

```bash
./gui/start_gui.sh               # starts the fermiconsole solar-flare console (FermiSunPy env), http://localhost:8766
```

The wrapper looks for fermiconsole in `~/gitrepository/fermiconsole` (set `FERMICONSOLE_PATH` otherwise). The console
code that was in `sunflare/` and `gui/` is in this repository's history (branch `solar-console`, cc5976d).
