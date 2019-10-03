#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define BILLION 1E9

struct timespec start, end;

double dp(long N, float *pA, float *pB) {
  double R = 0.0;
  int j;
  for (j=0;j<N;j++)
    R += pA[j]*pB[j];
  //printf("R is %f, N is %ld",R,N);
  return R;
}



int main(int argc, char* argv[])
{
	long N = atoi(argv[1]);
	int repetitions = atoi(argv[2]);

	long FLOP = 2L * N;

	float pa[N];
	float pb[N];
	double ans=0.0;
	int i = 0;
	double avg_time = 0.0;

	for (i=0; i<N; i++)
	{
		pa[i] = 1.0;
		pb[i] = 1.0;	
	}

	for (i=0; i<repetitions; i++)
	{
		clock_gettime(CLOCK_MONOTONIC, &start);
		ans = dp(N, pa, pb);
		
		assert(ans == (double)N);
		clock_gettime(CLOCK_MONOTONIC, &end);

		if (i > repetitions/2)
		{
			double time_1 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
			avg_time += time_1;
		}
	}

	avg_time = avg_time * 2 / (repetitions);

	double bw = (double)N * (double) sizeof(double) / (avg_time * BILLION);
	double flops = (double) FLOP / avg_time;
	
	printf("N: %ld\n<T>: %lf sec \nBw: %lf GB/s \nF: %lf FLOP/s\n",N,avg_time ,bw, flops);
	return 0;
}
