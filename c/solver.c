/*
 * Copyright (c) 2021, Dirk Pleiter, KTH
 *
 * This source code is in parts based on code from Jiri Kraus (NVIDIA) and
 * Andreas Herten (Forschungszentrum Juelich)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND Any
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR Any DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON Any THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN Any WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "proc_info.h"

#include "jacobi.h"
#include "init.h"
#include "reduction.h"


int solver(double *v, double *vp, double *f, int nx, int ny, double eps, int nmax, struct proc_info *proc)
{
    int n = 0;
    //double e = 2. * eps;

    double *w_device, *e_device;
    double *e, *w;

    unsigned int num_gpu_threads = prepareMiscMemory(w, e, w_device, e_device);

    *e = 2. * eps;

    while (((*e) > eps) && (n < nmax))
    {
        *w = 0;
        *e = 0;

        /*Start of computation phase*/
        jacobiStep(vp, v, f, nx, ny, e_device, w_device);
        sync();
        /*End of computation phase*/

        /*Communication Phase*/
 
        MPI_Sendrecv(&v[nx*(ny-2)+1], 1, proc->row, proc->neighbors[NORTH], 0,
             &v[1], 1, proc->row, proc->neighbors[SOUTH], 0, proc->cartcomm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&v[nx*1 + 1], 1, proc->row, proc->neighbors[SOUTH], 0,
             &v[nx*(ny-1) + 1], 1, proc->row, proc->neighbors[NORTH], 0, proc->cartcomm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&v[nx*2 - 2], 1, proc->column, proc->neighbors[EAST], 0, 
            &v[nx], 1, proc->column, proc->neighbors[WEST], 0, proc->cartcomm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&v[nx+1], 1, proc->column, proc->neighbors[WEST], 0,
             &v[nx*2 - 1], 1, proc->column, proc->neighbors[EAST], 0, proc->cartcomm, MPI_STATUS_IGNORE);

        /*End of communication phase*/

        /*Compute weight on the boundary*/
        if (proc->coords[0] == 0)
        {
            weightBoundary_x(v,nx,ny,w_device,0);
        }

        if(proc->coords[0] == proc->dims[0]-1)
        {
            weightBoundary_x(v,nx,ny,w_device,ny-1);
        }

        if(proc->coords[1] == 0)
        {
            weightBoundary_y(v,nx,ny,w_device,0);
        }

        if(proc->coords[1] == proc->dims[1]-1)
        {
            weightBoundary_y(v,nx,ny,w_device,nx-1);
        }

        deviceReduce(w_device, w, num_gpu_threads);
        deviceReduceMax(e_device, e, num_gpu_threads);

        sync();

        MPI_Allreduce(MPI_IN_PLACE, e, 1, MPI_DOUBLE, MPI_MAX, proc->cartcomm);
        MPI_Allreduce(MPI_IN_PLACE, w, 1, MPI_DOUBLE, MPI_SUM, proc->cartcomm);

        *w /= (NX * NY);
        *e /= *w;
        
        /*
        if(proc->rank == 0)
        {
            if ((n % 10) == 0)
                printf("%5d, %0.4e\n", n, e);
        }*/
        //if ((n % 10) == 0)
        //    printf("%5d, %0.4e\n", n, e);

        n++;
    }

    double e_local = *e;

    freeMiscMemory(w,e,w_device,e_device);


    if(proc->rank == 0)
    {
        if (e_local < eps)
            printf("Converged after %d iterations (nx=%d, ny=%d, e=%.2e)\n", n, NX, NY, e_local);
        else
            printf("ERROR: Failed to converge\n");
    }

    return (e_local < eps ? 0 : 1);
}