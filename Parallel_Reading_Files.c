#include <stdio.h>
#include <omp.h>
#include <time.h>


#define MAX 10000
#define NOT_CONNECTED -1
#define FILE_PATH "Testing-n-500.txt"

int distance[MAX][MAX];

int nodesCount;

void Initialize(){
    int i,j;
    for (i=0;i<MAX;++i){
        for (j=0;j<MAX;++j){
            distance[i][j]=NOT_CONNECTED;

        }
        distance[i][i]=0;
    }
}

void readLine(int a, int b, int c){

}


int main(void){

    clock_t start=clock();


    FILE *in_file  = fopen(FILE_PATH, "r");
    if (in_file  == NULL)
    {
        printf("Can't open file for reading.\n");
        return -1;
    }



    Initialize();

    fscanf(in_file,"%d", &nodesCount);

    int a, b, c;
    int faultFlag;
    #pragma omp parallel firstprivate(a,b,c)
    {
        while(fscanf(in_file,"%d %d %d", &a, &b, &c)!= EOF){
        {
            if ( a > nodesCount || b > nodesCount){
            faultFlag = -1;
            }
            distance[a][b]=c;
        }
        }

    }

    if(faultFlag == -1){
        printf("Vertex index out of boundary.");
            return -1;
    }

    //Floyd-Warshall
    int k;
    for (k=1;k<=nodesCount;++k){
        int i;
        #pragma omp parallel for schedule(static)
        for (i=1;i<=nodesCount;++i){
            if (distance[i][k]!=NOT_CONNECTED){
                int j;
                for (j=1;j<=nodesCount;++j){
                    if (distance[k][j]!=NOT_CONNECTED && (distance[i][j]==NOT_CONNECTED || distance[i][k]+distance[k][j]<distance[i][j])){
                        distance[i][j]=distance[i][k]+distance[k][j];
                    }
                }
            }
        }
    }

    int diameter=-1;

    //look for the most distant pair
    int i;
    for (i=1;i<=nodesCount;++i){
        int j;
        for (j=1;j<=nodesCount;++j){
            if (diameter<distance[i][j]){
                diameter=distance[i][j];
                //printf("%d-%d-%d\n", i, diameter, j);
            }
        }
    }

    printf("%d\n", diameter);

    clock_t end=(clock()-start)/CLOCKS_PER_SEC;
    printf("%f\n", end);

    return 0;

}
