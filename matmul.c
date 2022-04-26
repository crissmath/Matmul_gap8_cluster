#include "pmsis.h"
#include "Gap.h"


#if defined(__PULP_OS__)
#include "pulp.h"
#define pi_pmu_set_voltage(x, y)      ( rt_voltage_force(RT_VOLTAGE_DOMAIN_MAIN, x, NULL) )
#endif


#define MATRIX_SIZE 64
#define min(a,b)    (((a)<(b))?(a):(b))

/* Variables used. */
PI_L2 char * matA;
PI_L2 char * matB;
PI_L2 int  * matC;


/* 
    Matrix Multiplication with sdotp intrisics    
*/
void matmul_sdotp(void *arg)
{
    // core ID 
    uint32_t core_id    =  pi_core_id();
    // cluster ID 
    uint32_t cluster_id =  pi_cluster_id();

    // Matrices
    v4s * MatVecA, * MatVecB;
    
    //block size for core 
    int blockSize = MATRIX_SIZE / pi_cl_cluster_nb_cores();

    // core indentifier init
    int start = rt_core_id()*blockSize;
    // stop max core
    int stop = min(start+blockSize, MATRIX_SIZE);


    // in-place transpose B matrix
    for (int i = start; i < stop; i++) {
        for (int j = 0; j <= i; j++) {
            char temp = matB[i*MATRIX_SIZE+j];
            
            matB[i*MATRIX_SIZE+j] = matB[j*MATRIX_SIZE+i];
            matB[j*MATRIX_SIZE+i] = temp;
        }
    }
    pi_cl_team_barrier();

    // MatMul
    for (int i=start; i < stop; i++) {
        // load A matrix 
        MatVecA = (v4s*)(&matA[i*MATRIX_SIZE]);
        for (int j = 0; j < MATRIX_SIZE; j++) {
            // temp variable sum 
            int S = 0;
            // load B matrix 
            MatVecB = (v4s*)(&matB[j*MATRIX_SIZE]);
            // Manual unrolling factor 4
            for (int k = 0; k < MATRIX_SIZE>>2; k++) {
                S = gap_sumdotp4(MatVecA[k], MatVecB[k],S);
            }
            matC[i*MATRIX_SIZE+j] = S;
        }
    }
    pi_cl_team_barrier();

}


/* 
    Matrix Multiplication 
*/
void matmul(void *arg)
{
    uint32_t core_id = pi_core_id();
    uint32_t cluster_id = pi_cluster_id();

    int blockSize = (MATRIX_SIZE+pi_cl_cluster_nb_cores()-1) / pi_cl_cluster_nb_cores();
    int start = rt_core_id()*blockSize;
    int stop = min(start+blockSize, MATRIX_SIZE);
    for (int i=start; i < stop; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matC[i*MATRIX_SIZE+j] = 0;
            // Manual unrolling
            for (int k = 0; k < MATRIX_SIZE; k+=4) {
                matC[i*MATRIX_SIZE+j] += matA[i*MATRIX_SIZE+k] *   matB[k*MATRIX_SIZE+j];
                matC[i*MATRIX_SIZE+j] += matA[i*MATRIX_SIZE+k+1] * matB[k*MATRIX_SIZE+j+MATRIX_SIZE];
                matC[i*MATRIX_SIZE+j] += matA[i*MATRIX_SIZE+k+2] * matB[k*MATRIX_SIZE+j+2*MATRIX_SIZE];
                matC[i*MATRIX_SIZE+j] += matA[i*MATRIX_SIZE+k+3] * matB[k*MATRIX_SIZE+j+3*MATRIX_SIZE];
            }
        }
    }
    pi_cl_team_barrier();

}


/* Cluster main entry, executed by core 0. */
void cluster_delegate(void *arg)
{
    unsigned int Ti;

    printf("Cluster master core entry\n");

    /*
        Matrix allocation and initialization
    */
    matA = (char *) pi_l1_malloc(0, (MATRIX_SIZE*MATRIX_SIZE) * 2 * sizeof(char));
    matB = (char *) pi_l1_malloc(0, (MATRIX_SIZE*MATRIX_SIZE) * 2 * sizeof(char));
    matC = (int *)  pi_l1_malloc(0, (MATRIX_SIZE*MATRIX_SIZE) * 2 * sizeof(int));
    
    // init matrix 
    for(int i=0;i<MATRIX_SIZE*MATRIX_SIZE;i++){
        matA[i] = 1;
        matB[i] = 1;
    }


    /* 
        Matrix Multiplication with unrolling
    */
    printf("Run a Parallel Matrix Multiplication\n");
    pi_perf_conf(1 << PI_PERF_ACTIVE_CYCLES);
    pi_perf_reset(); 
    pi_perf_start();
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), matmul, arg);
    pi_perf_stop();

    Ti = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
    printf("Computation done in %d cycles at %.2f operations per cycle....\n", 
            Ti, ((float) (MATRIX_SIZE*MATRIX_SIZE*MATRIX_SIZE)/Ti));


    /* 
        Matrix Multiplication with transpose + intrisics
    */
    printf("Run a Parallel Matrix Multiplication w/ intrinsics\n");
    pi_perf_conf(1 << PI_PERF_ACTIVE_CYCLES);
    pi_perf_reset();
    pi_perf_start();
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), matmul_sdotp, arg);
    pi_perf_stop();
    Ti = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
    printf("Computation done in %d cycles at %.2f operations per cycle....\n", 
        Ti, ((float) (MATRIX_SIZE*MATRIX_SIZE*MATRIX_SIZE)/Ti));

}

//The chip wakes up at 50MhZ for Fabric Controller and Cluster
void fc_main(void)
{
    printf("Entering main controller\n");    
    uint32_t errors = 0;
    uint32_t core_id = pi_core_id();
    uint32_t cluster_id = pi_cluster_id();

    printf("[%d %d] Hello World!\n", (int) cluster_id, (int) core_id);

    /* 
        Set SoC Voltage to 1.2V
    */
    uint32_t voltage_in_mV = 1200;
    if(pi_pmu_set_voltage(voltage_in_mV, 0)==-1){
        printf("Frequency set failed!\n");
        pmsis_exit(-1);
    }

    /* 
        Set Cluster Freq at 250 MHz 
    */
    uint32_t fc_freq_in_Hz = 250 * 1000 * 1000;
    pi_freq_set(PI_FREQ_DOMAIN_FC, fc_freq_in_Hz);
    printf("Fabric Controller Frequency %d Hz\n", (int) pi_freq_get(PI_FREQ_DOMAIN_FC));

    /* 
        Configure & open cluster. 
    */
    struct pi_device cluster_dev   = {0};
    struct pi_cluster_conf cl_conf = {0};

    // Init cluster configuration structure.
    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;                             /* Set cluster ID. */
    pi_open_from_conf(&cluster_dev, &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-1);
    }

    /* 
        Set the max freq for the cluster @1.2V
    */
    uint32_t cl_freq_in_Hz = 175 * 1000 * 1000;
    pi_freq_set(PI_FREQ_DOMAIN_CL, cl_freq_in_Hz);
    printf("Cluster Frequency %d Hz\n", (int) pi_freq_get(PI_FREQ_DOMAIN_CL));


    /* 
        Prepare cluster task and send it to cluster. 
    */
    struct pi_cluster_task cl_task = {0};
    cl_task.entry = cluster_delegate;
    cl_task.arg = NULL;
    // send task to cluster someone need odentifier 
    pi_cluster_send_task_to_cl(&cluster_dev, &cl_task);
    pi_cluster_close(&cluster_dev);


    // Terminate and exit the test
    printf("Test success !\n");
    pmsis_exit(errors);
}

/* Program Entry. */
int main(void)
{
    printf("\n\n\t *** PMSIS MatMul & Frequency Test ***\n\n");
    return pmsis_kickoff((void *) fc_main);
}

