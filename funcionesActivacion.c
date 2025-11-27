#include "tools.c"

float relu(float x){
    return (x > 0)?x:0;
}

float devRelu(float x){
    return (x > 0)?1:0;
}

float identity(float x){
    return x;
}

float devIdentity(float x){
    return 1;
}

float sigmoidea(float x){
    return 1/(1+powf(e, -x));
}

float devSigmoidea(float x){
    float s = sigmoidea(x);
    return s * (1 - s);
}