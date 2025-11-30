#include "funcionesActivacion.c"
#include <math.h>

float mse(float *predicted, float *actually, int n)
{
    float result = 0.0f;
    for (unsigned int i = 0; i < n; i++)
        result += powf(predicted[i] - actually[i], 2);
    return result / n;
}

float mae(float *predicted, float *actually, int n)
{
    float result = 0.0f;
    for (unsigned int i = 0; i < n; i++)
    {
        float micro = predicted[i] - actually[i];
        result += (micro > 0) ? micro : -micro;
    }
    return result / n;
}

float rmse(float *predicted, float *actually, int n)
{
    return sqrtf(mse(predicted, actually, n));
}

float lostEntropy(float *predicted, float *actually, int n)
{
    const float eps = 1e-7f;
    float sum = 0.0f;
    for (unsigned int i = 0; i < n; i++)
    {
        float p = predicted[i];
        float y = actually[i];
        /* clamp p to avoid log(0) */
        if (p < eps)
            p = eps;
        if (p > 1.0f - eps)
            p = 1.0f - eps;
        sum += -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
    }
    return sum / (float)n;
}

float devMse(float predicted, float actually, int n)
{
    return 2 * (predicted - actually);
}

float devMae(float predicted, float actually, int n)
{
    return (2 / n) * (predicted - actually);
}

// float devRmse(float *predicted, float *actually, int n)
// {
//     float mse_val = mse(predicted, actually, n);
//     if (mse_val <= 0.0f)
//         return 0.0f;
//     return (1.0f / (2.0f * sqrtf(mse_val))) * devMse(predicted, actually, n);
// }

float devLostEntropy(float predicted, float actually, int n)
{
    const float eps = 1e-7f;
    float sum = 0.0f;
    float p = predicted;
    float y = actually;
    if (p < eps)
        p = eps;
    if (p > 1.0f - eps)
        p = 1.0f - eps;
    sum += (-(y / p) + ((1.0f - y) / (1.0f - p)));
    return (-(y / p) + ((1.0f - y) / (1.0f - p))) / (float)n;
}