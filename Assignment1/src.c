/*
 * CODE DESCRIPTION:
 * Our implementation simulates a parallel data processing pipeline using strictly
 * point-to-point MPI communication. The system operates on a linear topology where
 * every process functions in a dual capacity: as a sender, it dispatches rank-
 * dependent pseudo-random data to downstream neighbours at offsets D1 and D2; as
 * a receiver, it processes incoming data using specific mathematical transformations.
 *
 * For T iterations, senders transmit arrays of M doubles to their targets. The
 * receiving ranks perform element-wise computations—squaring the values for the
 * D1 channel and computing natural logarithms for the D2 channel—before returning
 * the results. Upon receipt, the sender updates its buffers using modulo and
 * multiplication rules to prepare for the next cycle.
 *
 * The simulation concludes with a binary-tree reduction to identify the global
 * maximums. This algorithm efficiently gathers the final results at Rank 0 in
 * O(log P) steps, strictly adhering to the assignment constraint that forbids
 * MPI_Reduce for data values.
 */

//===================================================================================


#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<math.h>

#define COMM MPI_COMM_WORLD
#define ull unsigned long long

/*
 * ==================================================================================
 * HELPER FUNCTIONS
 * ==================================================================================
 */

/*
 * PHASE 1: INITIALIZATION (Data Generation)
 * -----------------------------------------
 * Generates the initial dataset for the sender buffers.
 * Formula: buf[i] = (double)rand() * (rank + 1) / 10000.0
 * * Note: The random number generator is seeded using srand(seed) at the start
 * to ensure that results are reproducible across different runs.
 */
void generate(int seed, double* arr, int m, int rank){
    srand(seed);
    for(int i=0;i<m;i++){
        arr[i]=(double)rand()*(rank+1)/10000.0;
    }
}

/*
 * PHASE 2: COMPUTATION ROUTINES (Round 2)
 * ---------------------------------------
 * These functions are invoked by valid receiver ranks on the incoming data.
 */

// Function: compute_d1
// Performs element-wise squaring on the D1 channel data.
void compute_d1(double* data_d1, int m){    
    for(int i=0;i<m;i++){
        data_d1[i] = data_d1[i]*data_d1[i];
    }
}

// Function: compute_d2
// Applies the natural logarithm to the D2 channel data.
// Note: Input data is guaranteed to be positive by the generation logic.
void compute_d2(double* data_d2, int m){
    for(int i=0;i<m;i++){
        data_d2[i] = log(data_d2[i]);
    }
}

// Function: compute
// Helper utility to extract the local maximum from an array.
// Used during the final aggregation phase.
void compute(double* arr, int m, double* maxm){
    for(int i=0;i<m;i++){
        if(arr[i] > *maxm) *maxm = arr[i];
    }
}

/*
 * PHASE 2: UPDATE ROUTINES (Round 4)
 * ----------------------------------
 * Senders update their buffers for the next iteration based on the 
 * processed results returned from the receivers.
 */

// Update rule for D1: Modulo 100000 of the data.
// Cast to unsigned long long (ull) to ensure integer modulo operation.
void update_buf_d1(double* arr, int m){
    for(int i=0;i<m;i++){
        arr[i] = (ull)arr[i]%100000;
    }
}

// Update rule for D2: Multiplication by 1e+05.
void update_buf_d2(double* arr, int m){
    for(int i=0;i<m;i++){
        arr[i] = arr[i]*100000.0;
    }
}


/*
 * ==================================================================================
 * MAIN PROGRAM
 * ==================================================================================
 */
int main(int argc, char* argv[]){
    int rank, p;
    int m,d1,d2,t,seed;
    
    // It parses command line arguments: m, d1, d2, t, seed
    // M: Array size
    // D1, D2: Neighbor offsets
    // T: Number of iterations
    if (argc < 6) return 1; 
    m=atoi(argv[1]);
    d1=atoi(argv[2]); d2=atoi(argv[3]);
    t=atoi(argv[4]); seed=atoi(argv[5]);
    
    // Initialize MPI Environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(COMM, &rank);
    MPI_Comm_size(COMM, &p);
    
    double time = 0;
    MPI_Status status;
    
    // Pointers for buffers (initialized to NULL)
    double* data_d1 = NULL;
    double* data_d2 = NULL;
    double* buf_d1 = NULL, *buf_d2=NULL;

    /* * --- MEMORY ALLOCATION ---
     * Strategy: Optimize memory usage and prevent illegal access.
     * * 1. Outgoing buffers (buf_d1/d2) are allocated only if the rank is a valid sender.
     * (i.e., if rank + D < P).
     * 2. Incoming buffers (data_d1/d2) are allocated only if the rank is a valid receiver.
     * (i.e., if rank - D >= 0).
     */
    if(rank + d1 < p){
        buf_d1 = (double*)malloc(m*sizeof(double));
    }
    if(rank - d1 >= 0){
        data_d1 = (double*)malloc(m*sizeof(double));
    }
    if(rank - d2 >= 0){
        data_d2 = (double*)malloc(m* sizeof(double));
    }
    if(rank+d2<p){
        buf_d2 = (double*)malloc(m*sizeof(double));
    }

    /* * --- INITIALIZATION ---
     * If this rank is a sender, populate the primary buffer.
     * * Requirement Check: The problem states D1 and D2 start with identical data.
     * If a rank sends to both D1 and D2, we copy buf_d1 to buf_d2 to ensure consistency.
     */
    if(rank + d1 < p){
        generate(seed, buf_d1, m, rank); 
        
        if(rank + d2 < p){
            for(int i=0;i<m;i++)buf_d2[i]=buf_d1[i];
        }
    }

    // Start Global Timer
    double time1 = MPI_Wtime();

    /* * ==============================================================================
     * PHASE 2: THE ITERATIVE CORE
     * Executes T iterations, each consisting of 4 logical rounds.
     * ==============================================================================
     */
    while(t--){
        
        /* * ROUND 1: PARALLEL COMMUNICATION (Forward Transfer R -> R+D)
         * * Challenge: Blocking MPI_Send/MPI_Recv in a linear topology creates deadlock which leads to serialisation
         * if everyone Sends or everyone Receives simultaneously.
         * * Solution: Parity-Based odd even Scheduling (Checkerboard Strategy).
         * Ranks are divided into blocks of size 'D'. 
         * Parity = (rank / D) % 2
         * * - Even Blocks: Perform SEND then RECV.
         * - Odd Blocks:  Perform RECV then SEND.
         * * This ensures that for any pair (R, R+D), one is sending while the other is receiving.
         */

        // --- D1 Channel Communication ---
        // Check block parity for distance D1
        if(!((rank/d1)&1)){ 
            // EVEN BLOCK: Active Push
            if(rank + d1 < p) MPI_Send(buf_d1, m, MPI_DOUBLE, rank + d1, 0, COMM);
            if(rank - d1 >= 0) MPI_Recv(data_d1, m, MPI_DOUBLE, rank - d1, 0, COMM, &status);
        }
        else{ 
            // ODD BLOCK: Passive Wait
            if(rank - d1 >= 0) MPI_Recv(data_d1, m ,MPI_DOUBLE, rank - d1, 0, COMM, &status);
            if(rank + d1 < p) MPI_Send(buf_d1, m, MPI_DOUBLE, rank + d1, 0, COMM);
        }

        // --- D2 Channel Communication ---
        // Parity is calculated independently for D2 to decouple the channels.
        if(!((rank/d2)&1)){ 
            // EVEN BLOCK
            if(rank + d2 < p) MPI_Send(buf_d2, m, MPI_DOUBLE, rank + d2, 0, COMM);
            if(rank - d2 >= 0) MPI_Recv(data_d2, m, MPI_DOUBLE, rank - d2, 0, COMM, &status);
        }
        else{ 
            // ODD BLOCK
            if(rank - d2 >= 0) MPI_Recv(data_d2, m ,MPI_DOUBLE, rank - d2, 0, COMM, &status);
            if(rank + d2 < p) MPI_Send(buf_d2, m, MPI_DOUBLE, rank + d2, 0, COMM);
        }

        /* * ROUND 2: COMPUTATION
         * Valid receivers process the incoming data.
         * - D1 Channel: Square the data.
         * - D2 Channel: Logarithm of the data.
         */
        if(rank-d1 >= 0){
            compute_d1(data_d1, m);
            if(rank-d2 >= 0){
                compute_d2(data_d2, m);
            }
        }

        /* * ROUND 3: REVERSE COMMUNICATION (Return Results R+D -> R)
         * * The data flow is now reversed (Receiver sends back to Sender).
         * To maintain deadlock freedom, the priority is flipped relative to Round 1:
         * * - Odd Blocks: SEND then RECV.
         * - Even Blocks: RECV then SEND.
         */

        // --- D1 Channel Return ---
        if((rank/d1)&1){ 
            // ODD BLOCK: Active Push (Reverse)
            if(rank - d1 >= 0) MPI_Send(data_d1, m, MPI_DOUBLE, rank - d1, 0, COMM);
            if(rank + d1 < p) MPI_Recv(buf_d1, m, MPI_DOUBLE, rank + d1, 0, COMM, &status);
        }
        else{ 
            // EVEN BLOCK: Passive Wait (Reverse)
            if(rank + d1 < p) MPI_Recv(buf_d1, m ,MPI_DOUBLE, rank + d1, 0, COMM, &status);
            if(rank - d1 >= 0) MPI_Send(data_d1, m, MPI_DOUBLE, rank - d1, 0, COMM);
        }

        // --- D2 Channel Return ---
        if((rank/d2)&1){ 
            // ODD BLOCK
            if(rank - d2 >= 0) MPI_Send(data_d2, m, MPI_DOUBLE, rank - d2, 0, COMM);
            if(rank + d2 < p) MPI_Recv(buf_d2, m, MPI_DOUBLE, rank + d2, 0, COMM, &status);
        }
        else{ 
            // EVEN BLOCK
            if(rank + d2 < p) MPI_Recv(buf_d2, m ,MPI_DOUBLE, rank + d2, 0, COMM, &status);
            if(rank - d2 >= 0) MPI_Send(data_d2, m, MPI_DOUBLE, rank - d2, 0, COMM);
        }

        /* * ROUND 4: UPDATE
         * Senders update their buffers for the next iteration using the 
         * processed results received in Round 3.
         */
        if(rank + d2 < p){
            update_buf_d1(buf_d1, m);
            update_buf_d2(buf_d2, m);
        }
        else if(rank + d1 < p){
            update_buf_d1(buf_d1, m);
        }
    }
    
    /* * ==============================================================================
     * PHASE 3: FINAL AGGREGATION (Binary Tree Reduction)
     * ==============================================================================
     * Assignment Requirement: Find Global Max without using MPI_Reduce on data values.
     * * Strategy: Binary Tree Reduction
     * 1. Each valid sender calculates local maximums for D1 and D2.
     * 2. We use a binary tree pattern where stride 'diff' doubles (1, 2, 4...).
     * 3. In each step, half the active ranks send their max to their neighbor and drop out.
     * 4. The other half receive, compare with their local max, and continue.
     * * Complexity: O(log P) steps.
     */
     
    double maxm1=-1, maxm2=-1;  
    double a[2] = {maxm1, maxm2};
    double time2 = 0;
    
    // Check if this rank holds valid data (is a sender)
    if(buf_d1){  
        // Step 1: Extract Local Maximums
        if(rank + d1 < p) compute(buf_d1, m, &maxm1);
        if(rank + d2 < p) compute(buf_d2, m, &maxm2);
        a[0] = maxm1;
        a[1] = maxm2;
        
        double b[2];
        int diff = 1;         // Initial stride
        int num = p - d1;     // Limit of valid ranks
        
        // Step 2: Tree Reduction Loop
        while(diff < num){ 
            
            // LOGIC FOR SENDER ROLE:
            // Ranks at position (diff) in a block of (2*diff) send their data "left".
            // Example (diff=1): Ranks 1, 3, 5 send to 0, 2, 4.
            if((rank % (diff << 1)) == diff){
                if(rank - diff >= 0) MPI_Send(a, 2, MPI_DOUBLE, rank-diff, 0, COMM);
            }
            
            // LOGIC FOR RECEIVER ROLE:
            // Ranks at position 0 in a block of (2*diff) receive data.
            // Example (diff=1): Ranks 0, 2, 4 receive.
            else if(((rank % (diff << 1)) == 0) && (rank + diff < num)){
                MPI_Recv(b, 2, MPI_DOUBLE, rank + diff, 0, COMM, &status);
                
                // Compare received max (b) with local max (a) and store result in (a)
                if(a[0] < b[0]) a[0] = b[0];
                if(a[1] < b[1]) a[1] = b[1];    
            }
            
            diff <<= 1; // Recursive doubling: Shift bit left to double stride (1 -> 2 -> 4)
        }
        time2 = MPI_Wtime();
    }
    
    /* * GLOBAL TIMING & REPORTING
     */
    
    // Calculate local elapsed time
    // Note: Ranks without buf_d1 have time2=0, resulting in negative time, 
    // but MPI_MAX filters this out correctly.
    time = time2 - time1;
    double max_time=0;

    // Use MPI_Reduce ONLY for aggregating the maximum execution time (Permitted).
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, COMM); 

    // Rank 0 outputs the final aggregated results to file
    if(!rank){
        char filename[32];
        sprintf(filename, "out_%d.txt", p);

        FILE* fptr = fopen(filename, "a");
        // Output Format: <GlobalMaxD1> <GlobalMaxD2> <MaxTime>
        fprintf(fptr, "%lf %lf %lf\n", a[0], a[1], max_time);
        fclose(fptr);
    }

    // Cleanup: Free allocated memory
    if(rank-d1>=0) free(data_d1);
    if(rank-d2>=0) free(data_d2);
    if(rank+d1<p) free(buf_d1);
    if(rank+d2<p) free(buf_d2);

    MPI_Finalize();
    return 0;
}
