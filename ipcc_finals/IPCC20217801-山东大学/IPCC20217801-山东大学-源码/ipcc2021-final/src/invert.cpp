/**
** @file:  invert.cpp
** @brief:
**/

#include "invert.h"
#include "operator_mpi.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
using namespace std;

int CGinvert(complex<double> *src_p, complex<double> *dest_p, complex<double> *gauge[4],
             const double mass, const int max, const double accuracy, int *subgs, int *site_vec)
{
    lattice_gauge U(gauge, subgs, site_vec);
    lattice_fermion src(src_p, subgs, site_vec);
    lattice_fermion dest(dest_p, subgs, site_vec);
    CGinvert(src, dest, U, mass, max, accuracy);
    return 0;
}

int CGinvert(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
             const int max, const double accuracy)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    lattice_fermion r0(src.subgs, src.site_vec);
    lattice_fermion r1(src.subgs, src.site_vec);
    lattice_fermion q(src.subgs, src.site_vec);
    lattice_fermion qq(src.subgs, src.site_vec);
    lattice_fermion p(src.subgs, src.site_vec);
    lattice_fermion Mdb(src.subgs, src.site_vec);
    lattice_fermion tmp(src.subgs, src.site_vec);

    complex<double> aphi(0);
    complex<double> beta(0);

    int subgrid_vol = src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3] * 3 * 4;
    int subgrid_vol_cb = subgrid_vol >> 1;

    // M^dagger
    Dslash_even(src,tmp,U,mass,false);
    Dslash_odd(tmp,Mdb,U,mass,true);

    Dslash_odd(dest,tmp,U,mass,false);
    Dslash_odd(tmp,r0,U,mass,true);

    // Dslash(dest, tmp, U, mass, false);
    // Dslash(tmp, r0, U, mass, true);

    // for (int i = 0; i < Mdb.size; i++) {
    //     r0.A[i] = Mdb.A[i] - r0.A[i];
    // }
    for (int i = subgrid_vol_cb; i < Mdb.size; i++)
    {
        r0.A[i] = Mdb.A[i] - r0.A[i];
    }

    for (int f = 1; f < max; f++) {
        if (f == 1)
        {
            for (int i = subgrid_vol_cb; i < r0.size; i++)
                p.A[i] = r0.A[i];
        }
        else
        {
            beta = vector_p_O(r0, r0) / vector_p_O(r1, r1);
            for (int i = subgrid_vol_cb; i < r0.size; i++)
            {
                p.A[i] = r0.A[i] + beta * p.A[i];
            }
        }

        // Dslash(p, qq, U, mass, false);
        // Dslash(qq, q, U, mass, true);
        Dslash_odd(p, qq, U, mass, false);
        Dslash_odd(qq, q, U, mass, true);

        aphi = vector_p_O(r0, r0) / vector_p_O(p, q);

        for (int i = subgrid_vol_cb; i < dest.size; i++)
            dest.A[i] = dest.A[i] + aphi * p.A[i];

        for (int i = subgrid_vol_cb; i < r1.size; i++)
            r1.A[i] = r0.A[i];

        for (int i = subgrid_vol_cb; i < r0.size; i++)
            r0.A[i] = r0.A[i] - aphi * q.A[i];

        double rsd2 = norm_2_O_mpi(r0);
        double rsd = sqrt(rsd2);
        if (rsd < accuracy) {
            if (myrank == 0) {
                cout << "CG: " << f << " iterations, convergence residual |r| = " << rsd << endl;
            }
            break;
        }
#ifndef VERBOSE_SIMPLE
        if (myrank == 0) {
            cout << "CG: " << f << " iter, rsd |r| = " << rsd << endl;
        }
#endif
    }

    Restruct_even(src,dest,U, mass, false);

    return 0;
}

void Dslash_odd(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger)
{
        int subgrid_vol = src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3];
        int subgrid_vol_cb = (subgrid_vol) >> 1;

        lattice_fermion tmp1(src.subgs, src.site_vec);
        lattice_fermion tmp2(src.subgs, src.site_vec);

        DslashOO(src, tmp1, mass);

        for(int i = subgrid_vol_cb*12; i < subgrid_vol * 12; i++)
        {
            dest.A[i] = tmp1.A[i];
        }

        Dslashoffd(src, tmp1, U, dagger, 0);

        DslashEE_inv(tmp1, tmp2, mass);

        Dslashoffd(tmp2, tmp1, U, dagger, 1);

        for (int i = subgrid_vol_cb * 3 * 4; i < subgrid_vol * 3 * 4; i++)
        {

            dest.A[i] -= tmp1.A[i];
        }
}

void Dslash_even(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
                  const bool dagger)
{

    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    lattice_fermion tmp1(src.subgs, src.site_vec);
    lattice_fermion tmp2(src.subgs, src.site_vec);

    DslashEE_inv(src, tmp1, mass);

    Dslashoffd(tmp1, tmp2, U, dagger, 1);

    for (int i = subgrid_vol_cb * 3 * 4; i < subgrid_vol * 3 * 4; i++)
    {
        dest.A[i] = src.A[i] - tmp2.A[i];
    }
}

void Restruct_even(lattice_fermion &b, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger)
{
    lattice_fermion &src = dest;
    int subgrid_vol = src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3];
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    lattice_fermion tmp1(src.subgs, src.site_vec);
    lattice_fermion tmp2(src.subgs, src.site_vec);

    Dslashoffd(src, tmp1, U, dagger, 0);
        
    for (int i = 0; i < subgrid_vol_cb * 3 * 4; i++) {
        tmp2.A[i] = b.A[i]-tmp1.A[i];
    }

    DslashEE_inv(tmp2, tmp1, mass);

    for(int i = 0; i < subgrid_vol_cb * 3 * 4; i++){

        dest.A[i] = tmp1.A[i];
    }
}

void Dslash(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger)
{
    dest.clean();
    lattice_fermion tmp(src.subgs, src.site_vec);
    DslashEE(src, tmp, mass);
    dest = dest + tmp;
    DslashOO(src, tmp, mass);
    dest = dest + tmp;
    Dslashoffd(src, tmp, U, dagger, 0); // cb=0, EO
    dest = dest + tmp;
    Dslashoffd(src, tmp, U, dagger, 1);
    dest = dest + tmp;
}

void Dslash_origin(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger)
{
    dest.clean();
    lattice_fermion tmp(src.subgs, src.site_vec);
    DslashEE(src, tmp, mass);
    dest = dest + tmp;
    DslashOO(src, tmp, mass);
    dest = dest + tmp;
    Dslashoffd(src, tmp, U, dagger, 0); // cb=0, EO
    dest = dest + tmp;
    Dslashoffd(src, tmp, U, dagger, 1);
    dest = dest + tmp;
}