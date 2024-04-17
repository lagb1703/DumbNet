#include "funcionesActivacion.c"

float mse(float *predicted, float *actually, int n){
    float result = 0.0f;
    for(unsigned int i = 0; i < n; i++)
        result += powf(predicted[i] - actually[i], 2);
    return result/n;
}

float mae(float *predicted, float *actually, int n){
    float result = 0.0f;
    for(unsigned int i = 0; i < n; i++){
        float micro = predicted[i] - actually[i];
        result += (micro > 0)?micro:-micro;
    }
    return result/n;
}

float rmse(float *predicted, float *actually, int n){
    return sqrtf(mse(predicted, actually, n));
}

float devMse(float predicted, float actually){
    return 2*(predicted-actually);
}

float devMae(float predicted, float actually){
    float micro = predicted - actually;
    return (micro)/(micro > 0)?micro:-micro;
}

float devRmse(float predicted, float actually){
    return sqrtf(devMse(predicted, actually));
}