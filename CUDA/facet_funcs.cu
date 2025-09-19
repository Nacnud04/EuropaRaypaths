#define IDX_FCT(i, j, nxfacets) ((i) + (j) * (nxfacets))

__host__ void generateFacetNormals(float* h_fx, float* h_fy, float* h_fz,
                                   float* h_fnx, float* h_fny, float* h_fnz,
                                   float* h_fux, float* h_fuy, float* h_fuz,
                                   float* h_fvx, float* h_fvy, float* h_fvz,
                                   int nxfacets, int nyfacets) {

    // NOTE: This will only function with facets in a grid

    float dz_dy, dz_dx;

    // loop over each axis of facets
    for (int j = 0; j < nyfacets - 1; j++) {
        for (int i = 0; i < nxfacets - 1; i++) {

            // if facet on edge skip
            if (i == nxfacets - 1 || j == nyfacets - 1) {
                continue;
            }

            int idx = IDX_FCT(i, j, nxfacets);

            // compute dz/dy
            dz_dy = (h_fz[IDX_FCT(i, j + 1, nxfacets)] - h_fz[IDX_FCT(i, j, nxfacets)]) /
                    (h_fy[IDX_FCT(i, j + 1, nxfacets)] - h_fy[IDX_FCT(i, j, nxfacets)]);

            // compute dz/dx
            dz_dx = (h_fz[IDX_FCT(i + 1, j, nxfacets)] - h_fz[IDX_FCT(i, j, nxfacets)]) /
                    (h_fx[IDX_FCT(i + 1, j, nxfacets)] - h_fx[IDX_FCT(i, j, nxfacets)]);

            // compute normal vector components
            h_fnx[idx] = -dz_dx;
            h_fny[idx] = -dz_dy;
            h_fnz[idx] = 1.0f;

            // normalize
            float norm = sqrt(h_fnx[idx] * h_fnx[idx] +
                              h_fny[idx] * h_fny[idx] +
                              h_fnz[idx] * h_fnz[idx]);

            h_fnx[idx] /= norm;
            h_fny[idx] /= norm;
            h_fnz[idx] /= norm;

            // compute tangent vecotrs in the plane

            // tangent in x direction
            h_fux[idx] = 1.0f;
            h_fuy[idx] = 0.0f;
            h_fuz[idx] = dz_dx;
            norm = sqrt(h_fux[idx] * h_fux[idx] +
                        h_fuy[idx] * h_fuy[idx] +
                        h_fuz[idx] * h_fuz[idx]);
            h_fux[idx] /= norm; h_fuy[idx] /= norm; h_fuz[idx] /= norm;

            // tangent in y direction
            h_fvx[idx] = 0.0f;
            h_fvy[idx] = 1.0f;
            h_fvz[idx] = dz_dy;
            norm = sqrt(h_fvx[idx] * h_fvx[idx] +
                        h_fvy[idx] * h_fvy[idx] +
                        h_fvz[idx] * h_fvz[idx]);
            h_fvx[idx] /= norm; h_fvy[idx] /= norm; h_fvz[idx] /= norm;

        }
    }

    // Handle edges by copying normals from adjacent facets
    for (int j = 0; j < nyfacets; j++) {
        for (int i = 0; i < nxfacets; i++) {
            int idx = IDX_FCT(i, j, nxfacets);  
            if (i == nxfacets - 1 && i > 0) {  
                h_fnx[idx] = h_fnx[IDX_FCT(i - 1, j, nxfacets)];
                h_fny[idx] = h_fny[IDX_FCT(i - 1, j, nxfacets)];
                h_fnz[idx] = h_fnz[IDX_FCT(i - 1, j, nxfacets)];
            }
            if (j == nyfacets - 1 && j > 0) {  
                h_fnx[idx] = h_fnx[IDX_FCT(i, j - 1, nxfacets)];
                h_fny[idx] = h_fny[IDX_FCT(i, j - 1, nxfacets)];
                h_fnz[idx] = h_fnz[IDX_FCT(i, j - 1, nxfacets)];
            }
        }
    }

}

__host__ void generateFlatSurface(float* f_x, float* f_y, float* f_z,
                                  int nxfacets, int nyfacets, 
                                  float x0, float y0, float z0,
                                  float fs)
{
    // loop over facets
    for (int j = 0; j < nyfacets; j++)
    {
        for (int i = 0; i < nxfacets; i++)
        {
            int idx = i + j * nxfacets;

            f_x[idx] = x0 + i * fs;
            f_y[idx] = y0 + j * fs;
            f_z[idx] = z0;
        }
    }
}
