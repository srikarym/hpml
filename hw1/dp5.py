import sys
import numpy as np
import time

BILLION = 1E9

def dp(N,A,B):
	return np.dot(A,B)

def main():

	N = int(sys.argv[1])
	repetitions = int(sys.argv[2])

	FLOP = 2*N

	A = np.ones(N,dtype=np.float32)
	B = np.ones(N,dtype=np.float32)

	ans = 0.0
	total_time = 0
	j = 0
	for i in range(1,repetitions+1):
		start = time.monotonic()
		ans = dp(N,A,B)
		end = time.monotonic()
		total_time += end-start

		if (i == repetitions // 2 -1):
			total_time = 0
			j += 1

	avg_time = total_time / j

	print(f'Dot product result is {ans}')

	bw = N*8 / (avg_time * BILLION)
	flops = FLOP / avg_time

	print(f'N: {N}\n<T_avg>: {avg_time} sec\nBw: {bw} GB/sec\nF: {flops} FLOP/s')

if __name__ == "__main__":
	main()