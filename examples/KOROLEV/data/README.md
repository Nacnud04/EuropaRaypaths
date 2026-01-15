# KOROLEV DATA

Some of the following data files are not found in the repository but can be found through
NASA's PDS system.

## MOLA
DEM of Mars' surface as captured by the Mars Orbiter Laser Altimeter. I am using 
the `0.00781` degree resolution data product with ID: `MGS-M-MOLA-5-MEGDR-L3-V1.0`.
On the PDS this has the filename: `megr88n090hb.img`.

**From this I used QGIS to crop the DEM** via `Extent.gpkg` to the relevant region
 producing the geotiff `RCropped.tif` which is provided.


## OBSERVATION

### GEOMETRY
This is the path which SHARAD took over KOROLEV. It is provided as the 
`s_00554201_geom.tab` file and supporting `.lbl.txt` and `.xml` files.

### RADAR DATA
This is the actual data captured by SHARAD. It is provided as `s_00554201_tiff.tif`
and supporting files with the `.lbl.txt` and `.xml` extensions.

