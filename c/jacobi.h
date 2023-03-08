//
// Created by Patrik Rac on 08.03.23.
//

#ifndef CHALLENGE_JACOBI_H
#define CHALLENGE_JACOBI_H

void jacobiStep(double *vp, double *v, double *f, int nx, int ny, double *e, double *w);
void weightBoundary_x(double *v, int nx, int ny, double *w, int iy);
void weightBoundary_y(double *v, int nx, int ny, double *w, int ix);
void sync();

#endif //CHALLENGE_JACOBI_H
