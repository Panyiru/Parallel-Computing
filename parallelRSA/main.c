#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#include "rsa.h"
#include "mygmp.h"

static int *crackedKeys = NULL;
static int crackedLen = 0;
int size, rank;


int checkIfCrackedAlready(int n) {
  for (int i = 0; i < crackedLen; ++i) {
    if (n == cracked[i])
      return 1;
  }

  return 0;
}

int main(int argc, char** argv) {
    // When invoking the program, you need to type in the filename and keyNum
    if (argc < 3){
        printf("The user needs to input the filename and number of keys in this file\n")
        return 0;
    }

    int keyNum = atoi(argv[2]);
    mpz_t* keys = mpz_reads(argv[1], keyNum);

    FILE *stream = argc == 4 ? fopen(argv[3], "w") : stdout;

    crackedKeys = malloc(keyNum * sizeof(int));
    mpz_t gcd, q1, q2, d1, d2;
    mpz_inits(gcd, q1, q2, d1, d2, NULL);

    //Initialize communicator
	MPI_Init( &argc, &argv );
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel for schedule(dynamic)
    for (int i = rank; i < keyNum-1; i+=size*2) {
        for (int j = i+1; j < keyNum; j++) {
          mpz_gcd(gcd, keys[i], keys[j]);
          if (mpz_cmp_ui(gcd, 1) != 0) {
              int crackedN1 = checkIfCrackedAlready(i);
              int crackedN2 = checkIfCrackedAlready(j);

              if (!crackedN1 || !crackedN2) {
                  if (!crackedN1) {
                      mpz_divexact(q1, keys[i], gcd);
                      rsa_compute_d(d1, keys[i], gcd, q1);
                      mpz_out_str(stream, 10, keys[i]);
                      fputc(':', stream);
                      mpz_out_str(stream, 10, d1);
                      fputc('\n', stream);

                      crackedKeys[crackedLen++] = i;
                      MPI_Bcast(crackedKeys, 1, MPI_INT, rank, MPI_COMM_WORLD);
                      MPI_Bcast(&crackedLen, 1, MPI_INT, rank, MPI_COMM_WORLD);
                   }

                  if (!crackedN2) {
                      mpz_divexact(q2, keys[j], gcd);
                      rsa_compute_d(d2, keys[j], gcd, q2);
                      mpz_out_str(stream, 10, keys[j]);
                      fputc(':', stream);
                      mpz_out_str(stream, 10, d2);
                      fputc('\n', stream);

                      crackedKeys[crackedLen++] = j;
                      MPI_Bcast(crackedKeys, 1, MPI_INT, rank, MPI_COMM_WORLD);
                      MPI_Bcast(&crackedLen, 1, MPI_INT, rank, MPI_COMM_WORLD);
                  }
              }
            }
         }
      }

    // For even distribution, we need to divide the whole matrix into two even parts
    #pragma omp parallel for schedule(dynamic)
    for (int i = (2*size-rank-1); i < keyNum-1; i+=size*2) {
        for (int j = i+1; j < keyNum; j++) {
          mpz_gcd(gcd, keys[i], keys[j]);
          if (mpz_cmp_ui(gcd, 1) != 0) {
              int crackedN1 = checkIfCrackedAlready(i);
              int crackedN2 = checkIfCrackedAlready(j);

              if (!crackedN1 || !crackedN2) {
                  if (!crackedN1) {
                      mpz_divexact(q1, keys[i], gcd);
                      rsa_compute_d(d1, keys[i], gcd, q1);
                      mpz_out_str(stream, 10, keys[i]);
                      fputc(':', stream);
                      mpz_out_str(stream, 10, d1);
                      fputc('\n', stream);

                      crackedKeys[crackedLen++] = i;
                      MPI_Bcast(crackedKeys, 1, MPI_INT, rank, MPI_COMM_WORLD);
                      MPI_Bcast(&crackedLen, 1, MPI_INT, rank, MPI_COMM_WORLD);

                   }

                  if (!crackedN2) {
                      mpz_divexact(q2, keys[j], gcd);
                      rsa_compute_d(d2, keys[j], gcd, q2);
                      mpz_out_str(stream, 10, keys[j]);
                      fputc(':', stream);
                      mpz_out_str(stream, 10, d2);
                      fputc('\n', stream);

                      crackedKeys[crackedLen++] = j;
                      MPI_Bcast(crackedKeys, 1, MPI_INT, rank, MPI_COMM_WORLD);
                      MPI_Bcast(&crackedLen, 1, MPI_INT, rank, MPI_COMM_WORLD);

                  }
              }
            }
         }
      }

    free(keys);
    free(crackedKeys);

    if (argc == 4)
        fclose(stream);

    MPI_Finalize();
    return 0;
}

