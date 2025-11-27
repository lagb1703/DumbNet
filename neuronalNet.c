#include "optimizadores.c"

typedef struct
{
    unsigned int neurons;
    float (*activacion)(float);
    float (*derivada)(float);
    char *name;
} Layer;
typedef Layer *Sequential;

typedef struct
{
    Sequential sequential;
    unsigned int length;
    float ***w;
    float **b;
} Model;

static void __w(float *w, unsigned int n)
{
    for (unsigned int l = 0; l < n; l++)
    {
        // w[l] = ((float)rand() / RAND_MAX) - 0.5f;
        w[l] = 0.5;
        printf("[%i] = %f\n", l, w[l]);
    }
}

float neuron(float *x,
             float *w,
             float *b,
             unsigned int n,
             float (*activacion)(float))
{
    float result = 0;
    while (n--)
    {
        result += x[n] * w[n] + b[n];
    }
    return activacion(result);
}

Model *fit(float **x,
           float **y,
           unsigned int n,
           unsigned int epochs,
           float learningRate,
           Sequential sequential,
           float (*error)(float *, float *, int),
           float (*devError)(float *, float *, int),
           float InitB,
           unsigned int length)
{
    Layer layer;
    float ***w = (float ***)malloc(sizeof(float **) * (length - 1));
    float ***deltas = (float ***)malloc(sizeof(float **) * (length - 1));
    float **neu = (float **)malloc(sizeof(float *) * length);
    float **b = (float **)malloc(sizeof(float *) * length);
    printf("configuración inicial\n");
    for (int i = 0; i < length; i++)
    {
        layer = sequential[i];
        neu[i] = (float *)malloc(sizeof(float) * layer.neurons);
        b[i] = (float *)malloc(sizeof(float) * layer.neurons);
        if (i < (length - 1))
        {
            w[i] = (float **)malloc(sizeof(float *) * layer.neurons);
            deltas[i] = (float **)malloc(sizeof(float *) * layer.neurons);
            for (unsigned int j = 0; j < layer.neurons; j++)
            {
                b[i][j] = InitB;
                w[i][j] = (float *)malloc(sizeof(float) * (sequential[i + 1].neurons));
                deltas[i][j] = (float *)malloc(sizeof(float) * (sequential[i + 1].neurons));
                printf("w[%i][%i]", i, j);
                __w(w[i][j], sequential[i + 1].neurons);
            }
        }
    }
    printf("entrenamiento:\n");
    while (epochs--)
    {
        for (int a = 0; a < n; a++)
        {
            printf("data  %i\n", a);
            printf("capa  0\n");
            for (unsigned int j = 0; j < sequential[0].neurons; j++)
            {
                neu[0][j] = x[a][j];
                printf("neu[%i][%i] = %f\n", 0, j, neu[0][j]);
            }
            for (unsigned int i = 1; i < length; i++)
            {
                printf("capa  %i\n", i);
                layer = sequential[i];
                for (unsigned int j = 0; j < layer.neurons; j++)
                {
                    float *sw = (float *)malloc(sizeof(float) * sequential[i - 1].neurons);
                    for (unsigned int k = 0; k < sequential[i - 1].neurons; k++)
                    {
                        sw[k] = w[i - 1][k][j];
                    }
                    neu[i][j] = neuron(neu[i - 1], sw, b[i - 1], sequential[i - 1].neurons, sequential[i - 1].activacion);
                    free(sw);
                    printf("neu[%i][%i] = %f\n", i, j, neu[i][j]);
                }
                printf("\n");
            }
            float err = error(neu[length - 1], y[a], layer.neurons);
            float devErr = devError(neu[length - 1], y[a], layer.neurons);
            if (layer.neurons == 1)
                printf("error: %f, devError: %f, prediccion: %f, y:%f\n", err, devErr, neu[length - 1][0], y[a][0]);
            float delta;
            float devNeu;
            float devAct;
            layer = sequential[length - 1];
            Layer before = sequential[length - 2];
            for (int j = 0; j < layer.neurons; j++)
            {
                devNeu = devErr;
                printf("derivada neurona: %f    ", devNeu);
                float *sw = (float *)malloc(sizeof(float) * before.neurons);
                for (unsigned int k = 0; k < before.neurons; k++)
                {
                    sw[k] = w[length - 2][k][j];
                }
                devAct = neuron(neu[length - 2], sw, b[length - 2], before.neurons, before.derivada);
                printf("derivada activación: %f\n", devAct);
                for (int k = 0; k < before.neurons; k++)
                {
                    deltas[length - 2][k][j] = devNeu * devAct;
                    printf("devZ[%i][%i][%i] = %f\n", length - 2, k, j, deltas[length - 2][k][j]);
                    b[length - 2][k] -= learningRate * deltas[length - 2][k][j];
                    printf("b[%i][%i] = %f  ", length - 2, j, b[length - 2][j]);
                    w[length - 2][k][j] -= learningRate * deltas[length - 2][k][j] * neu[length - 2][k];
                    printf("w[%i][%i][%i] = %f\n", length - 2, k, j, w[length - 2][k][j]);
                }
            }
            for (int i = length - 2; i > 0; i--)
            {
                layer = sequential[i];
                before = sequential[i - 1];
                float *sw = (float *)malloc(sizeof(float) * before.neurons);
                for (int j = 0; j < layer.neurons; j++)
                {
                    for (int k = 0; k < before.neurons; k++)
                    {
                        sw[k] = w[i - 1][k][j];
                    }
                    printf("derivada neurona: %f    ", devNeu);
                    devAct = neuron(neu[i - 1], sw, b[i - 1], before.neurons, before.derivada);
                    printf("derivada activación: %f\n", devAct);
                    for (int k = 0; k < before.neurons; k++)
                    {
                        devNeu = deltas[i - 1][k][j] * w[i - 1][k][j];
                        deltas[i - 1][k][j] = devNeu * devAct;
                        b[i - 1][k] -= learningRate * deltas[i - 1][k][j];
                        printf("b[%i][%i] = %f  ", length - 2, j, b[i - 1][j]);
                        w[i - 1][k][j] -= learningRate * delta * neu[i - 1][k];
                        printf("w[%i][%i][%i] = %f\n", length - 2, k, j, w[i - 1][k][j]);
                    }
                }
            }
        }
    }
    Model *model = (Model *)malloc(sizeof(Model));
    model->b = b;
    model->w = w;
    model->sequential = sequential;
    free(deltas);
    free(neu);
    return model;
}

int main()
{
    srand(time(NULL));
    int n = 2;
    float **x = (float **)malloc(sizeof(float *) * 4);
    x[0] = (float *)malloc(sizeof(float) * 2);
    x[0][0] = 0;
    x[0][1] = 0;
    x[1] = (float *)malloc(sizeof(float) * 2);
    x[1][0] = 0;
    x[1][1] = 1;
    x[2] = (float *)malloc(sizeof(float) * 2);
    x[2][0] = 1;
    x[2][1] = 0;
    x[3] = (float *)malloc(sizeof(float) * 2);
    x[3][0] = 1;
    x[3][1] = 1;
    float **y = (float **)malloc(sizeof(float *) * 4);
    y[0] = (float *)malloc(sizeof(float) * 1);
    y[0][0] = 0;
    y[1] = (float *)malloc(sizeof(float) * 1);
    y[1][0] = 1;
    y[2] = (float *)malloc(sizeof(float) * 1);
    y[2][0] = 1;
    y[3] = (float *)malloc(sizeof(float) * 1);
    y[3][0] = 0;
    Layer inicio = {2, identity, devIdentity, "inicio"};
    Layer final = {1, identity, devIdentity, "final"};
    Sequential se = (Sequential)malloc(sizeof(Layer) * n);
    se[0] = inicio;
    se[1] = final;
    Model m = *(fit(x,
                    y,
                    4,
                    2,
                    0.5,
                    se,
                    mse,
                    devMse,
                    0,
                    n));
    // printf("%f", cal(m, 1));
    free(m.sequential);
    free(m.w);
    free(m.b);
    return 0;
}
