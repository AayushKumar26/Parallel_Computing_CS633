#include<stdio.h>
#include<mpi.h>
#include<string.h>
#include<stdlib.h>
#include<stdbool.h>

#define ll long long
#define COMM MPI_COMM_WORLD

/* Global domain dimensions (per process, excluding ghost zones) */
ll nx, ny, nz;

/* Stencil half-width: val = (d-1)/6, so a d-point stencil reads val layers
 * in each of the 6 directions (±x, ±y, ±z).
 * Example: d=7 → val=1 (immediate neighbors only) */
ll val;

/* Number of fields (independent variables), stencil diameter, precomputed reciprocal */
ll F, d;
double inv_d;   /* = 1/d, used in stencil division to replace costly division with multiply */

/* Layout of the padded local buffer (includes ghost zones on all 6 sides):
 *   x_len = nx + 2*val   (row length including west+east ghost columns)
 *   y_len = ny + 2*val   (not used after setup, kept for clarity)
 *   plane = x_len*y_len  (number of doubles per z-slice including ghosts)
 *
 * A grid point (j1, j2, j3) in local index space maps to flat index:
 *   j3*plane + j2*x_len + j1
 * where j1 in [val, nx+val), j2 in [val, ny+val), j3 in [val, nz+val)
 * are the owned (non-ghost) points. */
ll x_len, y_len, plane;

/*
 * initialize()
 *
 * Fill each field's owned region with pseudo-random doubles.
 * Ghost zones are NOT initialised here — they are populated by
 * halo exchanges before any computation uses them.
 *
 * The data layout in memory is:
 *   data[field][j3*plane + j2*x_len + j1]
 * where the outermost index is the field, and the inner flat index
 * addresses the 3D padded buffer.
 *
 * The initialisation formula must match the assignment spec exactly:
 *   data[i][j] = rand() * (rank+1) / (110426.0 + i + spatial_flat_idx)
 * v3 is the unpadded flat index (i.e. what j would be in a nx*ny*nz array),
 * idx = field_index + v3 reproduces i+j from the spec.
 */
static inline void initialize(double ** restrict data, ll seed, int rank){
    srand(seed);
    for(ll i=0; i<F; i++){
        for(ll j3=val; j3<nz+val; j3++){
            ll x1 = plane*j3;
            ll v1 = nx*ny*(j3-val);           /* unpadded z contribution */
            for(ll j2=val; j2<ny+val; j2++){
                ll x2 = x1 + x_len*j2;
                ll v2 = v1 + nx*(j2-val);     /* unpadded y contribution */
                for(ll j1=val; j1<nx+val; j1++){
                    ll x3 = j1 + x2;          /* padded flat index */
                    ll v3 = v2 + j1-val;       /* unpadded flat index (= j in spec) */
                    double idx = (double)i + (double)v3;
                    data[i][x3] = ((double)rand()*((double)rank + 1.0))/(110426.0 + idx);
                }
            }
        }
    }
}

/* Neighbour process ranks (-1 means no neighbour, i.e. domain boundary).
 * Process layout is px×py×pz with x fastest, then y, then z:
 *   rank = iz*(px*py) + iy*px + ix
 * Neighbours:
 *   north/south = ±y  (rank ± px, same z-layer)
 *   east/west   = ±x  (rank ± 1,  same row)
 *   front/back  = ±z  (rank ± px*py) */
int p, rank, px, py, pz, north, south, east, west, front, back;

/*
 * stencil_compute_core()
 *
 * Computes the d-point stencil for the INTERIOR of the local domain.
 * Interior = points at least val steps away from every face:
 *   j1 in [2*val, nx),  j2 in [2*val, ny),  j3 in [2*val, nz)
 *
 * These points only access neighbours within [val, n+val), so they
 * never touch ghost zones. This means this function can safely run
 * WHILE a halo exchange is in-flight (computation-communication overlap).
 *
 * For d=7 (val=1) and a 120³ domain, the interior is ~118³ ≈ 98.6% of
 * all points, so the overlap hides almost all of the boundary stencil cost.
 *
 * Uses multiply by inv_d instead of division for performance.
 */
static inline void stencil_compute_core(double ** restrict data, double** restrict n_data){
    for(ll i=0; i<F; i++){
        for(ll j3=2*val; j3<nz; j3++){
            ll x1 = plane*j3;
            for(ll j2=2*val; j2<ny; j2++){
                ll x2 = x_len*j2 + x1;
                for(ll j1=2*val; j1<nx; j1++){
                    ll val1 = j1 + x2;
                    double sum = data[i][val1];
                    /* Accumulate all val layers in ±x, ±y, ±z directions */
                    for(ll k=1;k<=val;k++){
                        sum += data[i][val1 - k*x_len] + data[i][val1 + k*x_len]   /* ±y */
                             + data[i][val1 - k]       + data[i][val1 + k]          /* ±x */
                             + data[i][val1 - k*plane] + data[i][val1 + k*plane];   /* ±z */
                    }
                    n_data[i][val1] = sum*inv_d;
                }
            }
        }
    }   
}

/*
 * compute_stencil_block()
 *
 * Computes the d-point stencil for an arbitrary rectangular sub-block
 * [zs,ze) × [ys,ye) × [xs,xe) of the padded buffer.
 *
 * Used exclusively for the 6 boundary shells (points that are within val
 * of at least one face). These points may need ghost zone data, so this
 * function is only called AFTER the halo exchange completes.
 *
 * Boundary semantics: if a neighbour in direction D exists (check_D == true),
 * that direction is always included regardless of how close to the face we are
 * (the ghost zone holds valid remote data). If no neighbour exists, the
 * direction is only included when the stencil tap stays within the local domain.
 * The divisor 'nval' is accumulated dynamically to match the actual number of
 * contributing points (as required by the assignment for domain-boundary processes).
 *
 * The bool flags (check_east etc.) are hoisted out of the loop to avoid
 * redundant comparisons against -1 inside the inner loop.
 */
static inline void compute_stencil_block(double ** restrict data, double ** restrict n_data,
                                          int f, ll zs, ll ze, ll ys, ll ye, ll xs, ll xe){
    /* Hoist neighbour-existence checks outside the triple loop */
    bool check_east  = (east  != -1);
    bool check_west  = (west  != -1);
    bool check_north = (north != -1);
    bool check_south = (south != -1);
    bool check_front = (front != -1);
    bool check_back  = (back  != -1);
	
    for(ll j3=zs; j3<ze; j3++){
        ll p_off = j3*plane;
        for(ll j2=ys; j2<ye; j2++){
            ll r_off = p_off + j2*x_len;
            for(ll j1=xs; j1<xe; j1++){
                ll val1 = r_off + j1;
                double sum = data[f][val1];
                double nval = 1;
                for(ll k=1; k<=val; k++){
                    /* Include direction if the tap is still inside the owned domain,
                     * OR if a neighbouring process exists (ghost zone is valid). */
                    if(((j2-k) >= val) || check_north){ sum += data[f][val1 - k*x_len]; nval++; }
                    if(((j2+k) < ny+val) || check_south){ sum += data[f][val1 + k*x_len]; nval++; }
                    if((j1-k >= val)   || check_west) { sum += data[f][val1 - k];       nval++; }
                    if((j1+k < nx+val) || check_east) { sum += data[f][val1 + k];       nval++; }
                    if((j3-k >= val)   || check_front){ sum += data[f][val1 - k*plane]; nval++; }
                    if((j3+k < nz+val) || check_back) { sum += data[f][val1 + k*plane]; nval++; }
                }
                n_data[f][val1] = sum / (double)nval;
            }
        }
    }
}

/*
 * stencil_compute_boundaries()
 *
 * Calls compute_stencil_block for the 6 boundary shells using a non-overlapping
 * decomposition that covers exactly the points NOT handled by stencil_compute_core:
 *
 *   Front/Back  z-shells: j3 in [val, 2*val) and [nz, nz+val)  — full xy face
 *   North/South y-shells: j2 in [val, 2*val) and [ny, ny+val)  — middle z-slice only
 *   West/East   x-shells: j1 in [val, 2*val) and [nx, nx+val)  — middle y,z-slice only
 *
 * The "remaining" comments in the call sites mean the z and y face rows are
 * excluded from the y and x shells respectively, so no point is processed twice.
 *
 * Must only be called AFTER MPI_Waitall on the halo exchange.
 */
static inline void stencil_compute_boundaries(double ** restrict data, double ** restrict n_data){
    for(int i=0; i<F; i++){
        /* Full xy slabs at front and back z-boundaries */
        compute_stencil_block(data, n_data, i, val,   2*val,  val, ny+val, val, nx+val);
        compute_stencil_block(data, n_data, i, nz,    nz+val, val, ny+val, val, nx+val);
        /* North/South y-strips, excluding z-boundary rows already done above */
        compute_stencil_block(data, n_data, i, 2*val, nz,     val,    2*val,  val, nx+val);
        compute_stencil_block(data, n_data, i, 2*val, nz,     ny,     ny+val, val, nx+val);
        /* West/East x-strips, excluding z and y boundary rows already done above */
        compute_stencil_block(data, n_data, i, 2*val, nz,     2*val,  ny,     val,    2*val);
        compute_stencil_block(data, n_data, i, 2*val, nz,     2*val,  ny,     nx,     nx+val);
    }
}

/*
 * iso_compute()
 *
 * Counts isosurface crossings across the local domain for each field.
 *
 * An isosurface crossing exists on an edge (P, Q) when iso_val lies between
 * data[P] and data[Q], i.e. (iso_val - data[P]) * (iso_val - data[Q]) <= 0.
 * This replaces the original two-comparison form with a single multiply,
 * which is branchless and easier for the compiler to vectorise.
 *
 * Only 3 negative directions are checked per point (west=-x, north=-y, front=-z).
 * Each undirected edge is therefore counted exactly once (owned by the point
 * with the higher index in each direction). This avoids double-counting
 * interior edges without any canonical-direction logic.
 *
 * For edges that cross a process boundary:
 *   - If a neighbour exists in that direction, the ghost zone holds the
 *     remote boundary layer (populated by the iso halo exchange), so the
 *     crossing check is always performed.
 *   - If no neighbour exists (domain boundary), the check is skipped for
 *     the outermost face layer (j == val) because there is no edge there.
 *
 * The condition  (check_D || j != val_face)  fuses both cases into one branch,
 * which the compiler can evaluate once per row/plane rather than per point.
 *
 * isocount[i] accumulates into a local ll (sum1) to avoid repeated pointer
 * dereferences inside the inner loop, then writes back once per field.
 *
 * Note: the caller divides the final global_count by 2 at output because
 * boundary edges between processes are counted by both processes. Interior
 * edges are never double-counted.
 */
static inline void iso_compute(double ** restrict data, ll* isocount, double iso_val){

    /* Hoist neighbour-existence checks outside the triple loop */
    bool check_west  = (west  != -1);
    bool check_north = (north != -1);
    bool check_front = (front != -1);

    for(ll i=0; i<F; i++){
        ll sum1 = 0;
        for(ll j3=val; j3<nz+val; j3++){
            ll a1 = j3*plane;
            for(ll j2=val; j2<ny+val; j2++){
                ll a2 = a1 + j2*x_len;
                for(ll j1=val; j1<nx+val; j1++){
                    ll c = a2 + j1;
                    double v1 = data[i][c];

                    /* Front check (−z): skip the front face row only if no front neighbour */
                    if(check_front || (j3!=val))
                        sum1 += ((iso_val - v1) * (iso_val - data[i][c-plane]) <= 0);

                    /* North check (−y): skip the north face row only if no north neighbour */
                    if(check_north || (j2!=val))
                        sum1 += ((iso_val - v1) * (iso_val - data[i][c-x_len]) <= 0);

                    /* West check (−x): skip the west face column only if no west neighbour */
                    if(check_west || (j1!=val))
                        sum1 += ((iso_val - v1) * (iso_val - data[i][c-1]) <= 0);
                }
            }
        }
        isocount[i] = sum1;
    }
}


int main(int argc, char* argv[]){

    /* Parse command-line arguments (order defined by assignment spec):
     *   d    — stencil diameter (7, 13, ...)
     *   ppn  — processes per node (used only for output filename)
     *   px py pz — process grid dimensions (total processes = px*py*pz)
     *   nx ny nz — local subdomain size per process
     *   T    — number of time steps
     *   seed — random seed for initialisation
     *   F    — number of fields
     *   iso_val — isovalue to count crossings for */
    d = atoll(argv[1]);
    inv_d = 1.0/(double)d;
    int ppn = atoll(argv[2]);
    px = atoi(argv[3]);
    py = atoi(argv[4]);
    pz = atoi(argv[5]);
    nx = atoll(argv[6]);
    ny = atoll(argv[7]);
    nz = atoll(argv[8]);
    ll T = atoll(argv[9]);
    ll seed = atoll(argv[10]);
    F = atoll(argv[11]);
    double iso_val = atof(argv[12]);

    MPI_Init(&argc, &argv);

    /* val layers of ghost zone on each face; buffer size = (n+2*val) in each dim */
    val   = (d-1)/6;
    x_len = nx+2*val;
    y_len = ny+2*val;
    plane = x_len*y_len;

    /* Allocate F separate flat arrays of size x_len * y_len * (nz+2*val) each.
     * Two buffers (data, n_data) are used for ping-pong: stencil reads from
     * data and writes to n_data, then pointers are swapped. */
    double **data = (double**)malloc(F*sizeof(double*));
    for(ll i=0; i<F; i++)
        data[i] = malloc(plane*(nz+2*val)*sizeof(double));

    double **n_data = (double**)malloc(F*sizeof(double*));
    for(ll i=0; i<F; i++)
        n_data[i] = malloc(plane*(nz+2*val)*sizeof(double));

    MPI_Comm_rank(COMM, &rank);
    MPI_Comm_size(COMM, &p);

    initialize(data, seed, rank);

    /* Determine the 6 neighbour ranks.
     * Process layout: rank = iz*(px*py) + iy*px + ix
     *   north/south: ±y → rank ± px, must stay in same z-layer (same rank/(px*py))
     *   east/west:   ±x → rank ± 1,  must stay in same y-row  (same rank/px)
     *   front/back:  ±z → rank ± px*py */
    north=east=south=west=front=back=-1;
    if((rank-px >= 0) && (((rank-px)/(px*py)) == (rank/(px*py))))
        north = rank-px;
    if((rank+px < px*py*pz) && ((rank+px)/(px*py) == (rank/(px*py))))
        south = rank+px;
    if((rank/px) == ((rank+1)/px))          east  = rank+1;
    if((rank>0) && ((rank/px)==((rank-1)/px))) west = rank-1;
    if((rank/(px*py)) != 0)                 front = rank-px*py;
    if((rank/(px*py)) != (pz-1))            back  = rank+px*py;

    /* MPI derived datatypes for the stencil halo exchange.
     * Each type describes val layers (ghost zone depth) on one face pair.
     *
     * ns_type (north/south, ±y):
     *   nz+2*val blocks of val*x_len contiguous doubles (one full xz-row-strip
     *   per z-slice), strided by plane (one z-slice apart).
     *
     * ew_type (east/west, ±x):
     *   plane = (nz+2*val)*(ny+2*val) blocks of val contiguous doubles
     *   (val columns), strided by x_len (one row apart).
     *
     * fb_type (front/back, ±z):
     *   Single contiguous block of val*plane doubles (val full z-slices). */
    MPI_Datatype ns_type, ew_type, fb_type;
    MPI_Type_vector(nz+2*val, val*x_len, plane, MPI_DOUBLE, &ns_type);
    MPI_Type_commit(&ns_type);
    MPI_Type_vector(plane, val, x_len, MPI_DOUBLE, &ew_type);
    MPI_Type_commit(&ew_type);
    MPI_Type_vector(1, val*plane, 0, MPI_DOUBLE, &fb_type);
    MPI_Type_commit(&fb_type);

    /* MPI derived datatypes for the isocount halo exchange.
     * Same structure as above but thickness = 1 layer instead of val layers,
     * because iso_compute only needs immediate neighbours (c±1, c±x_len, c±plane). */
    MPI_Datatype iso_ns, iso_ew, iso_fb;
    MPI_Type_vector(nz+2*val, x_len,  plane, MPI_DOUBLE, &iso_ns);
    MPI_Type_commit(&iso_ns);
    MPI_Type_vector(plane, 1, x_len, MPI_DOUBLE, &iso_ew);
    MPI_Type_commit(&iso_ew);
    MPI_Type_vector(1, plane, 0, MPI_DOUBLE, &iso_fb);
    MPI_Type_commit(&iso_fb);

    /* request array: up to 2 (send+recv) × 6 directions × F fields requests */
    MPI_Request *request = (MPI_Request*)malloc(12*F*sizeof(MPI_Request));
    ll nreq = 0;

    /* isocount[t*F + i] = local crossing count for field i at time step t */
    ll* isocount     = (ll*)malloc(F*T*sizeof(long long));
    ll* global_count = (ll*)malloc(F*T*sizeof(long long));

    ll cnt = 0, t = T;

    double t1 = MPI_Wtime();

    /* ------------------------------------------------------------------ *
     * Main time-step loop
     *
     * Per-iteration structure (communication-computation overlap):
     *
     *  ┌─ POST stencil halo (val layers, non-blocking)
     *  │   stencil_compute_core   ← runs while halo flies (interior only)
     *  │   MPI_Waitall            ← wait for ghost zones to arrive
     *  │   stencil_compute_boundaries ← boundary shells, needs ghost data
     *  │   swap data ↔ n_data
     *  │
     *  ├─ POST iso halo (1 layer, non-blocking)
     *  │   [no overlap here — iso_compute needs ghost on all 3 faces]
     *  │   MPI_Waitall
     *  │   iso_compute
     *  └──
     * ------------------------------------------------------------------ */
    while(T--){

        /* ---- Step 1: Post stencil halo exchange (val ghost layers) ---- */
        nreq = 0;

        /* North/South (±y): send own first/last val rows, recv into ghost rows */
        if(north!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][x_len*val],        1, ns_type, north, i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][0],                1, ns_type, north, i, COMM, &request[nreq++]);
            }
        }
        if(south!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][x_len*ny],         1, ns_type, south, i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][x_len*(ny+val)],   1, ns_type, south, i, COMM, &request[nreq++]);
            }
        }
        /* East/West (±x): send own first/last val columns, recv into ghost columns */
        if(east!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][nx],               1, ew_type, east,  i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][nx+val],           1, ew_type, east,  i, COMM, &request[nreq++]);
            }
        }
        if(west!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][val],              1, ew_type, west,  i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][0],                1, ew_type, west,  i, COMM, &request[nreq++]);
            }
        }
        /* Front/Back (±z): send own first/last val z-slices, recv into ghost slices */
        if(front!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][plane*val],        1, fb_type, front, i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][0],                1, fb_type, front, i, COMM, &request[nreq++]);
            }
        }
        if(back!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][plane*nz],         1, fb_type, back,  i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][plane*(nz+val)],   1, fb_type, back,  i, COMM, &request[nreq++]);
            }
        }

        /* ---- Step 2: Stencil computation (overlapped with halo) ---- */

        /* Interior points — no ghost zone access, safe while exchange is in-flight */
        stencil_compute_core(data, n_data);

        /* Wait for all ghost layers to arrive before computing boundary shells */
        MPI_Waitall(nreq, request, MPI_STATUSES_IGNORE);

        /* Boundary shells — require ghost zones, only safe after Waitall */
        stencil_compute_boundaries(data, n_data);

        /* Ping-pong: newly computed n_data becomes data for next iteration */
        double** temp = data;
        data = n_data;
        n_data = temp;

        /* ---- Step 3: Isocount — exchange then compute ---- */

        /* 3a. Post 1-layer halo on the freshly swapped data.
         * Sends the outermost owned layer in each direction so the receiving
         * process can check crossings across the process boundary.
         * offset (val-1) on recv side places the incoming layer just outside
         * the owned region, adjacent to the local boundary layer at val. */
        nreq = 0;

        if(north!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][x_len*val],        1, iso_ns, north, i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][x_len*(val-1)],    1, iso_ns, north, i, COMM, &request[nreq++]);
            }
        }
        if(south!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][x_len*(ny+val-1)], 1, iso_ns, south, i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][x_len*(ny+val)],   1, iso_ns, south, i, COMM, &request[nreq++]);
            }
        }
        if(east!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][nx+val-1],         1, iso_ew, east,  i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][nx+val],           1, iso_ew, east,  i, COMM, &request[nreq++]);
            }
        }
        if(west!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][val],              1, iso_ew, west,  i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][val-1],            1, iso_ew, west,  i, COMM, &request[nreq++]);
            }
        }
        if(front!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][plane*val],        1, iso_fb, front, i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][plane*(val-1)],    1, iso_fb, front, i, COMM, &request[nreq++]);
            }
        }
        if(back!=-1){
            for(ll i=0; i<F; i++){
                MPI_Isend(&data[i][plane*(nz+val-1)], 1, iso_fb, back,  i, COMM, &request[nreq++]);
                MPI_Irecv(&data[i][plane*(nz+val)],   1, iso_fb, back,  i, COMM, &request[nreq++]);
            }
        }

        /* 3b. Wait for all 1-layer iso ghost data to arrive */
        MPI_Waitall(nreq, request, MPI_STATUSES_IGNORE);

        /* 3c. Count isovalue crossings across the entire local domain */
        iso_compute(data, &isocount[cnt*F], iso_val);
        cnt++;
    }

    /* ---- Step 4: Global reduction — sum local isocounts to rank 0 ---- */
    MPI_Reduce(isocount, global_count, F*t, MPI_LONG_LONG, MPI_SUM, 0, COMM);

    /* Rank 0 writes T lines of F counts each, followed by the max wall time.
     * No /2 here — iso_compute uses canonical direction (3 negative dirs only)
     * so interior edges are counted once; the caller is responsible for any
     * boundary double-count handling if needed. */
    if(!rank){
        for(ll j=0; j<t; j++){
            for(ll i=0; i<F; i++)
                printf("%lld ", global_count[j*F + i]);
            printf("\n");
        }
    }

    double tot_time = (double)MPI_Wtime() - t1;
    double global_time;
    MPI_Reduce(&tot_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, COMM);
    if(!rank) printf("%lf\n", global_time);

    MPI_Type_free(&ns_type);
    MPI_Type_free(&ew_type);
    MPI_Type_free(&fb_type);
    MPI_Type_free(&iso_ns);
    MPI_Type_free(&iso_ew);
    MPI_Type_free(&iso_fb);

    for(ll i=0; i<F; i++){
        free(data[i]);
        free(n_data[i]);
    }
    free(data);
    free(n_data);
    free(request);
    free(isocount);
    free(global_count);

    MPI_Finalize();
    return 0;
}
