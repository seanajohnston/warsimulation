
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>


 class Nation {

 public:
    int
        attack_rate,
        tcount = 0,
        score = 0;

    double
         //Specialization multipliers. Specified by paramater f1
         mspec, //Increases "per unit" power
         tspec, //Decreases cost of technology

         //Technologies used as multipliers 
         ptech = 1, //Increases rate of resource production
         mtech = 1, //Decreases cost of military units

         //Resource investment strategies. Specified by parameter f2
         mfocus, //Percentage of budget for military purchase
         tfocus, //Percentage of budge for technology investment

         //Research strategies. Specified by parameter f3
         ptech_ratio, //Rate of production reasearch
         mtech_ratio, //Rate of military reasearch

         //Military size
         msize = 500,

         //Initial resources
         resources = 100,

        //Resource gain per turn
        rpt = 100,

         //Budget for current turn
         budget;

public:
     
    __device__ Nation(double r, double f1, double f2, double f3) {
        attack_rate = r*10;

        //Assign specialization
        if (f1 < 0.5) {
            mspec = 1.25; tspec = 1;
        }
        else {
            mspec = 1; tspec = 1.25;
        }
        
        //Assign resource investment strategy
        mfocus = f2;
        tfocus = 1 - f2;

        //Assign research strategy
        ptech_ratio = f3;
        mtech_ratio = 1 - f3;

    }

    __device__ Nation() {}

    __device__ void purchase_military() {
        //Military budget is calculated as a percentage of this turns budget
        //Percentage given by strategy in variable mfocus
        double mbudget = mfocus * budget;

        //Military tech increases amount of military purchased
        msize += mbudget * mtech;
    }

    __device__ void perform_research() {
        //Reaserch budget is a percentage of the turn's budget
        //Percentage is determined by strategy given by tfocus
        double rbudget = tfocus * budget;
        ptech += (rbudget * ptech_ratio) / 1000;
        mtech += (rbudget * mtech_ratio) / 1000;
    }

    //Returns 1 on turn to attack. Otherwise, returns 0
    __device__ int turn(int n) {

        //Increment resources
        resources += rpt * ptech;

        //Designate budget (%60 of current resources)
        budget = 0.6 * resources;
        resources -= budget;

        //Purchase military
        purchase_military();

        //Perform research
        perform_research();


        if (attack_rate == tcount) {
            tcount = 0;
            return 1;
        }
        else {
            tcount++;
        }
        return 0;

    }

    __device__ void attack(Nation defender) {
        //Each nation attacks/defends with %50 of their military
        double defense = defender.msize * 0.5 * defender.mspec, offense = msize * 0.5 * mspec,
            losses = offense - defense;

        if (losses > 0) {
            //This nation scores
            score += 15;
            //Nation gains resources from loser
            double plunder = defender.resources * 0.15;
            resources += plunder;
            defender.resources -= plunder;
            //Adjust militaries due to losses
            defender.msize -= defense;
            msize -= losses;
        }
        //This nation suffers loss. (Losses variable is negative)
        else {
            msize -= offense;
            defender.msize += losses;
        }
        
    }
     
};

__global__ void simulation(double *results, double *input)
{   
    Nation nations[5];
    int index = threadIdx.x * 20 + blockIdx.x * blockDim.x * 20;

    //Initalize 5 nations with input variables
    for (int i = index, j = 0; i < index + 20; i+=4, j++) {
        Nation temp(input[i], input[i + 1], input[i + 2], input[i + 3]);
        nations[j] = temp;
    }

    //Run simulation for 1000 turns

    int nextDefender = 0; //Follows round robin for defender
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 5; j++) {
            Nation temp = nations[j];
            if (temp.turn(i) == 1) {
                if (nextDefender != j) {
                    temp.attack(nations[nextDefender]);
                }
                else {
                    temp.attack(nations[nextDefender+1]);
                }
            }
            nextDefender = nextDefender < 4 ? nextDefender + 1 : 0;
        }
    }

    //Calculate victor
    int winner = 0, maxScore = 0;
    for (int i = 0; i < 5; i++) {
        int tempScore = 0;
        Nation temp = nations[i];
        tempScore = temp.score + temp.msize / 100 + temp.ptech + temp.mtech;
        if (tempScore > maxScore) {
            winner = i;
            maxScore = tempScore;
        }
    }

    //Store the parameters of the winner in results variable
    int i = threadIdx.x * 4 + blockIdx.x * blockDim.x * 4;
    Nation w = nations[winner];
    results[i] = w.attack_rate;
    results[i + 1] = w.mspec;
    results[i + 2] = w.mfocus;
    results[i + 3] = w.mtech_ratio;

    //Finished
}   

int main()
{
    const int
        BLOCK_COUNT = 10,
        THREADS_PER_BLOCK = 256,
        inputSize = BLOCK_COUNT * THREADS_PER_BLOCK * 20,
        resultSize = BLOCK_COUNT * THREADS_PER_BLOCK * 4;

        std::ofstream MyFile("results.txt");
        




    //Host variables
    double input[inputSize]; //Parameters to be given to countries
    double results[resultSize]; //Results will contain parameters of winning countries

    //Device variables
    double* dev_input;
    double* dev_results;

    //Device status
    cudaError_t cudaStatus;

    bool worked = true;

    for (int i = 0; i < inputSize; i++) {
        input[i] = (rand() % 9 + 1) * .1;
        //MyFile << input[i] << "\n";
    }
    
    

    
    

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_input, inputSize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_results, resultSize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_input, input, inputSize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    simulation<<<BLOCK_COUNT, THREADS_PER_BLOCK >>>(dev_results, dev_input);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(results, dev_results, resultSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
        return 1;
    }

    cudaFree(dev_results);
    cudaFree(dev_input);


    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        goto Error;
        return 1;
    }

    
    for (int i = 0; i < resultSize;) {
        for (int j = 0; j < 4 && i < resultSize; i++, j++) {
            MyFile << results[i] << " ";
        }
        MyFile << "\n";

    }


    

Error:
    cudaFree(dev_results);
    cudaFree(dev_input);
    
    return 0;
}
