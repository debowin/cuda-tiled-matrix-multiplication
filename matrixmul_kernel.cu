/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#define TILE_WIDTH 16

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	__shared__ float tileMs[TILE_WIDTH][TILE_WIDTH];
	__shared__ float tileNs[TILE_WIDTH][TILE_WIDTH];

	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;

	// target element coordinates
	int row = by * TILE_WIDTH + ty;
	int column = bx * TILE_WIDTH + tx;

	float pValue = 0;

	// compute target element value
	for(int i=0;i<ceilf(M.width/(float)TILE_WIDTH);i++){
		// move the tiles and update shared memory value for new tile positions
		if(row < M.height && (i*TILE_WIDTH + tx)<M.width)
			tileMs[ty][tx] = M.elements[row*M.width + i*TILE_WIDTH + tx];
		else
			tileMs[ty][tx] = 0;
		if(column < N.width && (i*TILE_WIDTH + ty)<N.height)
			tileNs[ty][tx] = N.elements[(i*TILE_WIDTH + ty)*N.width + column];
		else
			tileNs[ty][tx] = 0;

		// after the entire tile's values are available, proceed
		__syncthreads();

		for(int j=0;j<TILE_WIDTH;j++)
			pValue += tileMs[ty][j] * tileNs[j][tx];
		// after the entire tile's values have been used, proceed
		__syncthreads();
	}
	// boundary check
	if(row < P.height && column < P.width)
		P.elements[row*P.width+column] = pValue;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
