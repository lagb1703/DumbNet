#include "optimizadores.c"
#define learning_rate 0.01


typedef struct {
    unsigned int neurons;
    float (*activacion)(float);
    float (*derivada)(float);
    char *name;
} Layer;
typedef Layer * Sequential;



typedef struct{
    Sequential sequential;
    unsigned int length;
    float ***w;
    float **b;
} Model;


static void __w(float *w, unsigned int n){
    for (unsigned int l = 0; l < n; l++) {
        w[l] = ((float)rand()/ RAND_MAX)-0.5f;
        printf("[%i] = %f\n", l, w[l]);
    }
}

float neuron(float *x, 
                    float *w, 
                    float *b, 
                    unsigned int n,
                    float (*activacion)(float)){
    float result = 0;
    while(n--){
        result += activacion(x[n] * w[n] + b[n]);
    }
    return result;
}

Model *fit(float **x, 
                float **y, 
                unsigned int n,
                unsigned int epochs,
                Sequential sequential,
                float (*error)(float *, float *, int),
                float (*devError)(float, float),
                float InitB,
                unsigned int length){
    Layer layer;
    float ***w = (float ***) malloc(sizeof(float **)*(length-1));
    float ***deltas = (float ***) malloc(sizeof(float **)*(length-1));
    float **neu = (float **) malloc(sizeof(float *)*length);
    float **b = (float **) malloc(sizeof(float *)*length);
    printf("configuración inicial\n");
    for(int i = 0; i < length; i++){
        layer = sequential[i];
        neu[i] = (float *) malloc(sizeof(float)*layer.neurons);
        b[i] = (float *) malloc(sizeof(float)*layer.neurons);
        if(i < (length-1)){
            w[i] = (float **) malloc(sizeof(float *)*layer.neurons);
            deltas[i] = (float **) malloc(sizeof(float *)*layer.neurons);
            for(unsigned int j = 0; j < layer.neurons; j++){
                b[i][j] = InitB;
                w[i][j] = (float *) malloc(sizeof(float)*(sequential[i+1].neurons));
                deltas[i][j] = (float *) malloc(sizeof(float)*(sequential[i+1].neurons));
                //printf("%i %i\n", i, j);
                printf("w[%i][%i]", i, j);
                __w(w[i][j], sequential[i+1].neurons);
            }
        }
    }
    printf("entrenamiento:\n");
    while(epochs--){
        for(int a = 0; a < n; a++){
            printf("data  %i\n", a);
            printf("capa  0\n");
            for(unsigned int j = 0; j < sequential[0].neurons; j++){
                neu[0][j] = x[a][j];
                printf("neu[%i][%i] = %f\n", 0, j, neu[0][j]);
            }
            for(unsigned int i = 1; i < length; i++){
                printf("capa  %i\n", i);
                layer = sequential[i];
                //printf("sequential[%i] = %i, sequential[%i] = %i\n", i, layer.neurons, i-1, sequential[i-1].neurons);
                for(unsigned int j = 0; j < layer.neurons; j++){
                    float *sw = (float *)malloc(sizeof(float)*sequential[i-1].neurons);
                    for(unsigned int k = 0; k < sequential[i-1].neurons; k++){
                        sw[k] = w[i-1][k][j];
                    }
                    neu[i][j] = neuron(neu[i-1], sw, b[i-1], sequential[i-1].neurons, sequential[i-1].activacion);
                    free(sw);
                    printf("neu[%i][%i] = %f\n", i, j, neu[i][j]);
                }
                //printf("\n");
            }
            //printf("109 funciona");
            float err = error(neu[length-1], y[a], layer.neurons);
            if(layer.neurons == 1)
                printf("error: %f, prediccion: %f, y:%f\n", err, neu[length-1][0], y[a][0]);
            float delta = 1;
            layer = sequential[length - 1];
            Layer before = sequential[length-2];
            
            for(int j = 0; j < before.neurons; j++){
                float *sw = (float *)malloc(sizeof(float)*before.neurons);
                for(unsigned int k = 0; k < before.neurons; k++){
                    sw[k] = w[length-2][k][j];
                }
                delta = neuron(neu[length-2], sw, b[length-2], before.neurons, before.derivada);
            }
    //         for(int i = length - 2;i > 0; i--){
    //             layer = sequential[i];
    //             before = sequential[i-1];
    //             for(int j = 0; j < before.neurons; j++){
    //                 for(int k = 0; k < layer.neurons; k++){
    //                     delta = neuron(neu[i-1], w[i-1][j][k], b[i-1], before.neurons, before.derivada);
    //                     //printf("1 deltas[%i][%i][%i] = %f\n", i-1, j, k, deltas[i-1][j][k]);
    //                     deltas[i-1][j][k] = deltas[i][k][0] * delta;
    //                     delta = learning_rate * deltas[i-1][j][k] * delta;
    //                     w[i-1][j][k] -= delta*neu[i-1][j];
    //                     b[i-1][j] -= delta;
    //                 }
    //             }
    //         }
        }
    }
    Model *model = (Model *) malloc(sizeof(Model));
    model->b = b;
    model->w = w;
    model->sequential = sequential;
    free(deltas);
    free(neu);
    return model;
}

int main(){
    srand(time(NULL));
    int n = 3;
    float **x = (float **) malloc(sizeof(float *) * 4);
    x[0] = (float *) malloc(sizeof(float) * 2);
    x[0][0] = 0;
    x[0][1] = 0;
    x[1] = (float *) malloc(sizeof(float) * 2);
    x[1][0] = 0;
    x[1][1] = 1;
    x[2] = (float *) malloc(sizeof(float) * 2);
    x[2][0] = 1;
    x[2][1] = 0;
    x[3] = (float *) malloc(sizeof(float) * 2);
    x[3][0] = 1;
    x[3][1] = 1;
    float **y = (float **) malloc(sizeof(float *) * 4);
    y[0] = (float *) malloc(sizeof(float) * 1);
    y[0][0] = 0;
    y[1] = (float *) malloc(sizeof(float) * 1);
    y[1][0] = 1;
    y[2] = (float *) malloc(sizeof(float) * 1);
    y[2][0] = 1;
    y[3] = (float *) malloc(sizeof(float) * 1);
    y[3][0] = 0;
    Layer inicio = {2, relu, devRelu, "inicio"};
    Layer medio = {1, relu, devRelu, "medio"};
    Layer final = {1, relu, devRelu, "final"};
    Sequential se = (Sequential) malloc(sizeof(Layer)*n);
    se[0] = inicio;
    se[1] = medio;
    se[2] = final;
    Model m = *(fit(x, 
                    y, 
                    4, 
                    1, 
                    se, 
                    mse,
                    devMse,
                    0, 
                    n));
    //printf("%f", cal(m, 1));
    free(m.sequential);
    free(m.w);
    free(m.b);
    return 0;
}
