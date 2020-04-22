#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <mkl_cblas.h>

#define BILLION 1E9

struct timespec start, end;

float bdp(long N, float *pA, float *pB) {
	float R = cblas_sdot(N, pA, 1, pB, 1);
	return R;
}


int main(int argc, char* argv[])
{
	long N = atol(argv[1]);
	int repetitions = atoi(argv[2]);

	long FLOP = 2L * N;

	float pa[N];
	float pb[N];
	float ans=0.0;
	int i = 0;

	float total_time = 0.0;

	for (i=0; i<N; i++)
	{
		pa[i] = 1.0;
		pb[i] = 1.0;	
	}

	for (i=0; i<=repetitions; i++)
	{
		clock_gettime(CLOCK_MONOTONIC, &start);
		ans = bdp(N, pa, pb);
		clock_gettime(CLOCK_MONOTONIC, &end);

		float time_diff = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;

		total_time += time_diff;

		if (i == repetitions/2)
		{
			total_time = 0.0;
		}
	}

	printf("Dot product result is %lf\n",ans);

	int num_reps = repetitions / 2;

	float avg_time = total_time / num_reps;

	float bw = (float)N * (float) sizeof(float) * 2 / (avg_time * BILLION);
	float flops = (float) FLOP / avg_time;
	
	printf("N: %ld \n<T_avg>: %lf sec \nBw: %lf GB/s \nF: %lf FLOP/s\n",N,avg_time ,bw, flops);
	
	return ans;
}
