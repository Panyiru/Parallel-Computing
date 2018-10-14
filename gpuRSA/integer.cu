#include "integer.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void cuda_crackKeys(const integer *keys, uint16_t *block_noCoprime, int gridRow, int gridCol, int gridDim, int keyNum) {

  // In each block, we use two shared arrays to record the key pairs
  __shared__ volatile uint32_t keyOne[BLOCK_DIM][BLOCK_DIM][32];
  __shared__ volatile uint32_t keyTwo[BLOCK_DIM][BLOCK_DIM][32];

  // This will generate two keys for this block to compare.
  int keyX = gridCol * gridDim + blockIdx.x * BLOCK_DIM + threadIdx.x;
  int keyY = gridRow * gridDim + blockIdx.y * BLOCK_DIM + threadIdx.y;

  //We only need to compare each pair of key for one time
  if (keyX < keyNum && keyY < keyNum && keyX > keyY) {
    //Each thread will load its corresponding chunk
    keyOne[threadIdx.x][threadIdx.y][threadIdx.z] = keys[keyX].ints[threadIdx.z];
    keyTwo[threadIdx.x][threadIdx.y][threadIdx.z] = keys[keyY].ints[threadIdx.z];

    //Calculate gcd for each pair of keys
    gcd(keyOne[threadIdx.x][threadIdx.y], keyTwo[threadIdx.x][threadIdx.y]);

    if (threadIdx.x == 31) {
      // If gcd > 1, it means the pair is coPrime, and we need to record it.
      if ((keyTwo[threadIdx.x][threadIdx.y][threadIdx.z]) > 1) {
        int noCoprimeBlockId = blockIdx.y * gridDim.x + blockIdx.x;
        block_noCoprime[noCoprimeBlockId] |= 1 << threadIdx.y * BLOCK_DIM + threadIdx.x;
      }
    }
  }
}

/**
 * Binary GCD algo
 */
__device__ void gcd(volatile uint32_t *x, volatile uint32_t *y) {
  int tid = threadIdx.z;

  while (__any(x[tid])) {
    while ((x[31] & 1) == 0)
      shiftR1(x);

    while ((y[31] & 1) == 0)
      shiftR1(y);

    if (geq(x, y)) {
      cuSubtract(x, y, x);
      shiftR1(x);
    }
    else {
      cuSubtract(y, x, y);
      shiftR1(y);
    }
  }
}

__device__ void shiftR1(volatile uint32_t *x) {
  int tid = threadIdx.z;
  uint32_t prevX = tid ? x[tid-1] : 0;
  x[tid] = (x[tid] >> 1) | (prevX << 31);
}

__device__ void shiftL1(volatile uint32_t *x) {
  int tid = threadIdx.z;
  uint32_t nextX = tid != 31 ? x[tid+1] : 0;
  x[tid] = (x[tid] << 1) | (nextX >> 31);
}

__device__ int geq(volatile uint32_t *x, volatile uint32_t *y) {
  /* shared memory to hold the position at which the int of x >= int of y */
  __shared__ unsigned int pos[BLOCK_DIM][BLOCK_DIM];
  int tid = threadIdx.z;

  if (tid == 0)
    pos[threadIdx.x][threadIdx.y] = 31;

  if (x[tid] != y[tid])
    atomicMin(&pos[threadIdx.x][threadIdx.y], tid);

  return x[pos[threadIdx.x][threadIdx.y]] >= y[pos[threadIdx.x][threadIdx.y]];
}

__device__ void cuSubtract(volatile uint32_t *x, volatile uint32_t *y, volatile uint32_t *z) {
  /* shared memory to hold underflow flags */
  __shared__ unsigned char s_borrow[BLOCK_DIM][BLOCK_DIM][32];
  unsigned char *borrow = s_borrow[threadIdx.x][threadIdx.y];
  int tid = threadIdx.z;

  /* set LSB's borrow to 0 */
  if (tid == 0)
    borrow[31] = 0;

  uint32_t t;
  t = x[tid] - y[tid];

  /* set the previous int's underflow flag if the subtraction answer is bigger than the subtractee */
  if(tid)
    borrow[tid - 1] = (t > x[tid]);

  /* keep processing until there's no flags */
  while (__any(borrow[tid])) {
    if (borrow[tid])
      t--;

    /* have to set flag if the new sub answer is 0xFFFFFFFF becuase of an underflow */
    if (tid)
      borrow[tid - 1] = (t == 0xFFFFFFFFu && borrow[tid]);
  }

  z[tid] = t;
}
