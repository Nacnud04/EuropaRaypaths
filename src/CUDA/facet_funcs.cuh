#pragma once

__host__ void generateFacetNormals(float* h_fx, float* h_fy, float* h_fz,
                                   float* h_fnx, float* h_fny, float* h_fnz,
                                   float* h_fux, float* h_fuy, float* h_fuz,
                                   float* h_fvx, float* h_fvy, float* h_fvz,
                                   int nxfacets, int nyfacets);

__host__ void generateFlatSurface(float* f_x, float* f_y, float* f_z,
                                  int nxfacets, int nyfacets, 
                                  float x0, float y0, float z0,
                                  float fs);