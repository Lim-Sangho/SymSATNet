typedef struct mix_t {
    int b, n, m, k;
    int32_t *is_input;  // b*n
    int32_t *index;     // b*n
    int32_t *niter;     // b
    float *C, *dC;      // n*m
    float *z, *dz;      // b*n
    float *V, *U;       // b*n*k
    float *W, *Phi;     // b*m*k
    float *gnrm, *Cdiags;// b*n
    float *cache;
} mix_t ;
