//
// Created by Patrik Rac on 08.03.23.
//

#ifndef CHALLENGE_REDUCTION_H
#define CHALLENGE_REDUCTION_H
#include <stdlib.h>
extern "C"
{
void deviceReduce(double *in, double *out, int N);
void deviceReduceMax(double *in, double *out, int N);
};
#endif //CHALLENGE_REDUCTION_H
