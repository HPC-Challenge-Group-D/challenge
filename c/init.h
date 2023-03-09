//
// Created by Patrik Rac on 08.03.23.
//

#ifndef CHALLENGE_INIT_H
#define CHALLENGE_INIT_H
#include "proc_info.h"
#include <math.h>
#include <stdio.h>
extern "C"
{
void initDevice();
void prepareDataMemory(double **v, double **vp, double **f, int nx, int ny, int offset_x, int offset_y);
unsigned int prepareMiscMemory(double **w, double **e, double **w_device, double **e_device);
void freeDataMemory(double **v, double **vp, double **f);
void freeMiscMemory(double **w, double **e, double **w_device, double **e_device);
};
#endif //CHALLENGE_INIT_H

