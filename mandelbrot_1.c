#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

int size, rank;


// return 1 if in set, 0 otherwise
int inset(double real, double img, int maxiter){
    //Add a cardioid check
    double img2 = img*img;
    double check1 = (real-0.25)*(real-0.25) + img2;
    double check2 = 4*check1*(real-0.25+check1);
    double check3 = (real+1)*(real+1)+img2;
	if((check2<=img2)||(check3<0.0625))
		return 1;

    //If not, keep checking
	double z_real = real;
	double z_img = img;
	for(int iters = 0; iters < maxiter; iters++){
		double z2_real = z_real*z_real-z_img*z_img;
		double z2_img = 2.0*z_real*z_img;
		z_real = z2_real + real;
		z_img = z2_img + img;
		if(z_real*z_real + z_img*z_img > 4.0) return 0;
	}
	return 1;
}

// count the number of points in the set, within the region
int mandelbrotSetCount(double real_lower, double real_upper, double img_lower, double img_upper, int num, int maxiter){
	int count=0;
	double real_step = (real_upper-real_lower)/num;
	double img_step = (img_upper-img_lower)/num;
	#pragma omp parallel for schedule(dynamic)
	for(int real=0; real<num; real++){
		for(int img=0; img<num; img++){
			count+=inset(real_lower+real*real_step,img_lower+img*img_step,maxiter);
		}
	}
	return count;
}

// main
int main(int argc, char *argv[]){
	double real_lower;
	double real_upper;
	double img_lower;
	double img_upper;
	int num;
	int maxiter;
	int num_regions = (argc-1)/6;

    //Initialize communicator
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start = MPI_Wtime();
	for(int region=rank;region<num_regions;region+=size){
		// scan the arguments
		sscanf(argv[region*6+1],"%lf",&real_lower);
		sscanf(argv[region*6+2],"%lf",&real_upper);
		sscanf(argv[region*6+3],"%lf",&img_lower);
		sscanf(argv[region*6+4],"%lf",&img_upper);
		sscanf(argv[region*6+5],"%i",&num);
		sscanf(argv[region*6+6],"%i",&maxiter);

		int localCount = mandelbrotSetCount(real_lower,real_upper,img_lower,img_upper,num,maxiter);
		double end = MPI_Wtime();
		printf("Count is %d, run time is %f, by node %d", localCount, end-start, rank);
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
