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

float preZ(float *x, float *w, int n)
{
    float result = 0;
    while (n--)
        result += x[n] * w[n];
    return result;
}

Model *fit(float **x,
           float **y,
           unsigned int n,
           unsigned int epochs,
           float learningRate,
           Sequential sequential,
           float (*error)(float *, float *, int),
           float (*devError)(float , float , int),
           float InitB,
           unsigned int length)
{
    Layer layer;
    float ***w = (float ***)malloc(sizeof(float **) * (length - 1));
    float **deltas = (float **)malloc(sizeof(float *) * length);
    float **neu = (float **)malloc(sizeof(float *) * length);
    float **z = (float **)malloc(sizeof(float *) * length);
    float **b = (float **)malloc(sizeof(float *) * length);
    printf("configuración inicial\n");
    for (int i = 0; i < length; i++)
    {
        layer = sequential[i];
        neu[i] = (float *)malloc(sizeof(float) * layer.neurons);
        z[i] = (float *)malloc(sizeof(float) * layer.neurons);
        b[i] = (float *)malloc(sizeof(float) * layer.neurons);
        deltas[i] = (float *)calloc(sequential[i].neurons, sizeof(float));
        for (unsigned int bj = 0; bj < layer.neurons; ++bj)
            b[i][bj] = InitB;
        if (i < (length - 1))
        {
            w[i] = (float **)malloc(sizeof(float *) * layer.neurons);
            for (unsigned int j = 0; j < layer.neurons; j++)
            {
                w[i][j] = (float *)malloc(sizeof(float) * (sequential[i + 1].neurons));
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
                    z[i][j] = preZ(neu[i - 1], sw, sequential[i - 1].neurons) + b[i][j];
                    neu[i][j] = sequential[i].activacion(z[i][j]);
                    free(sw);
                    printf("neu[%i][%i] = %f\n", i, j, neu[i][j]);
                }
                printf("\n");
            }
            float err = error(neu[length - 1], y[a], layer.neurons);
            printf("error: %f\n", err);
            float delta;
            float devNeu;
            float devAct;
            layer = sequential[length - 1];
            Layer before = sequential[length - 2];
            for (int j = 0; j < layer.neurons; ++j)
            {
                printf("prediccion: %f, real: %f\n", neu[length - 1][j], y[a][j]);
                float dEdy = devError(neu[length - 1][j], y[a][j], 1);
                deltas[length - 1][j] = dEdy * sequential[length - 1].derivada(z[length - 1][j]);
            }
            for (int i = length - 2; i > 0; --i)
            {
                for (int j = 0; j < sequential[i].neurons; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < sequential[i + 1].neurons; ++k)
                    {
                        sum += w[i][j][k] * deltas[i + 1][k];
                    }
                    deltas[i][j] = sequential[i].derivada(z[i][j]) * sum;
                }
            }
            for (int i = 0; i < length - 1; ++i)
            {
                for (int j = 0; j < sequential[i].neurons; ++j)
                {
                    for (int k = 0; k < sequential[i + 1].neurons; ++k)
                    {
                        float grad = neu[i][j] * deltas[i + 1][k];
                        w[i][j][k] -= learningRate * grad;
                    }
                }
                for (int k = 0; k < sequential[i + 1].neurons; ++k)
                {
                    b[i + 1][k] -= learningRate * deltas[i + 1][k];
                }
            }
        }
    }
    Model *model = (Model *)malloc(sizeof(Model));
    model->b = b;
    model->w = w;
    model->sequential = sequential;
    free(w);

    for (int i = 0; i < length; ++i)
    {
        free(b[i]);
        free(z[i]);
        free(neu[i]);
        free(deltas[i]);
    }
    free(b);
    free(z);
    free(neu);
    free(deltas);
    return model;
}

int main()
{
    srand(time(NULL));
    int n = 3;
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
    Layer inicio = {2, sigmoidea, devSigmoidea, "inicio"};
    Layer medio = {2, sigmoidea, devSigmoidea, "inicio"};
    Layer final = {1, sigmoidea, devSigmoidea, "final"};
    Sequential se = (Sequential)malloc(sizeof(Layer) * n);
    se[0] = inicio;
    se[1] = medio;
    se[2] = final;
    Model m = *(fit(x,
                    y,
                    4,
                    10,
                    0.5,
                    se,
                    mse,
                    devMse,
                    0,
                    n));
    free(m.sequential);
    free(m.w);
    free(m.b);
    return 0;
}
