# REASON & RHYME Raypath Simulator

---

## Index
1. [Source](README.md#Source)
2. [Surface](README.md#Surface)
3. [Model](README.md#Model)

---

## Source

#### Initializing a source
Sources are initalized with a **location**, **sampling rate**, **duration** and **power**. The duration of the signal does not need to span the duration for the data being received. It just needs to be long enough to contain the entire wavelet.
#### Defining a source
Once initalized sources can be defined through calling `gauss_sin` or `ricker` which generates either a gaussian sin wavelet or a ricker wavelet correspondingly. Each are defined with a center frequency and an optional time offset. The default time offset is 0 s causing only half the wavelet to be realized.
#### Example
This shows a rickerwavelet with a center frequency of 9 MHz with a time offset of 250 ns.  
![RickerWavelet](images/RickerSource.png)

---

## Surface

#### Initializing a surface
The surface is a very simple class just containing a surface mesh composed of xyz locations as well as surface normals for each facet.  It is defined by a **facet size**, **origin** and **dimension**. The facet size is the side length of each facet, defaulting to 10 m, but being set at 100 m in the example notebook. The origin is the minimum x and y coordinate of the surface, and the dimensions are how many facets are there in each direction, defined as (x, y). So, with a facet size of 100 m and dimensions of 50 m by 50 m, the surface is 5 km by 5 km.
#### Defining a surface
Surfaces can be defined with the `gen_flat` function, which generates a surface of a constant elevation (z) and custom normal vectors. By default the normal vectors point upwards, but you can have a flat surface, where all the facets are pointing in a different direction, resulting in discontinuities on the surface.
#### Visualizing a surface
`show_surf` will produce a 3D plot of the surface. This does not account for the surface normal directions and instead just uses the Z value. So a "flat" surface with varying normal directions will appear the same, even though the normals are changing.

---

## Model

#### Initializing a model
The model is the class which actually simulates the propagation of the wavefront. It is simply defined with a predefined surface and source. Then a point target location is set, right now this is just a simple (x, y, z) tuple which defines the location and nothing else.
#### Generating raypaths
Generating raypaths produces a value for each facet which acts as the percentage of energy returned back to the source after reflecting off the target and passing through the facet twice. To do this the following steps occur:
1. Raypath instances are created. Raypaths start at the source and go to the facet. Then they are forced to go through the facet and towards the point target. Is is important to note that this is not the direction of the naturally refracted raypath.
2. The angle between the facet surface and the naturally refracted raypath is computed.
3. The angle between the facet surface and the forced ray from the surface to the target is computed.
4. The difference in angle between the refracted ray and the forced ray (ray to target) is computed.  
*Difference in angle between the forced ray and the refracted ray*  
![ForcedRefractedDiff](images/DTh-Forced-Refracted.png)  
5. The difference in angle is combined with the reradiation pattern from the facet to result in the returned energy from each facet.  
*2D reradiation pattern for a single facet*  
![facetrad](images/ReradiationFacet.png)  
***Reradiation amount for the entire footprint***  
![reradiated](images/reradiation.png)  
#### Timeseries generation
Timeseries generation is performed by the `gen_timeseries` function which generates the returned signal from the source &rarr; facet &rarr; target &rarr; facet &rarr; source raypaths. **This is the final simulated signal**. It is created by the following steps:
1. Antenna gain is defined.
2. Using the travel time for each raypath an offset is created in terms of indicies based off of the `dt` value defined by the source.
3. The source wavelet is multiplied by the trasmitted energy value defined as the returned energy from each facet created when generating raypaths.
4. The source wavelet is divided by $r^2$
5. The source wavelet is 0-padded so the predefined source timeseries starts at the index defined previously.
6. All modified and padded source wavelets are stacked to get the final timeseries.   
***This is an example timeseries and spectrum for a flat surface***
![FinalTimeseries](images/FinalTimeseries.png)