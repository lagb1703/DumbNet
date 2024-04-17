#include "funciones.c"

float sgd(float *parameters){
    float w = parameters[0];
    float n = parameters[1];
    float wp = parameters[2];
    return w - n*wp;
}