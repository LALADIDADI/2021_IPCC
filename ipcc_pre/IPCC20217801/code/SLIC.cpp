// SLIC.cpp: implementation of the SLIC class.
//===========================================================================
// This code implements the zero parameter superpixel segmentation technique
// described in:
//
//
//
// "SLIC Superpixels Compared to State-of-the-art Superpixel Methods"
//
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua,
// and Sabine Susstrunk,
//
// IEEE TPAMI, Volume 34, Issue 11, Pages 2274-2282, November 2012.
//
// https://www.epfl.ch/labs/ivrl/research/slic-superpixels/
//===========================================================================
// Copyright (c) 2013 Radhakrishna Achanta.
//
// For commercial use please contact the author:
//
// Email: firstname.lastname@epfl.ch
//===========================================================================

#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include "SLIC.h"
#include <chrono>

#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>
#include <string>
#include <mpi.h>

int num_procs;
int my_rank;
int numThreads = 64;

int threadNumberSmall = 8;
int threadNumberMid = 16;
int threadNumber = 64;

typedef chrono::high_resolution_clock Clock;

struct Node{
	int startY;
	int endY;
	int label;
	Node(int startY_, int endY_, int label_):startY(startY_), endY(endY_), label(label_){}
};

	

double get_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec/1000000;
}

// For superpixels
const int dx4[4] = {-1,  0,  1,  0};
const int dy4[4] = { 0, -1,  0,  1};
//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// For supervoxels
const int dx10[10] = {-1,  0,  1,  0, -1,  1,  1, -1,  0, 0};
const int dy10[10] = { 0, -1,  0,  1, -1, -1,  1,  1,  0, 0};
const int dz10[10] = { 0,  0,  0,  0,  0,  0,  0,  0, -1, 1};

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SLIC::SLIC()
{
	m_lvec = NULL;
	m_avec = NULL;
	m_bvec = NULL;

	m_lvecvec = NULL;
	m_avecvec = NULL;
	m_bvecvec = NULL;
}

SLIC::~SLIC()
{
	if(m_lvec) delete [] m_lvec;
	if(m_avec) delete [] m_avec;
	if(m_bvec) delete [] m_bvec;


	if(m_lvecvec)
	{
		for( int d = 0; d < m_depth; d++ ) delete [] m_lvecvec[d];
		delete [] m_lvecvec;
	}
	if(m_avecvec)
	{
		for( int d = 0; d < m_depth; d++ ) delete [] m_avecvec[d];
		delete [] m_avecvec;
	}
	if(m_bvecvec)
	{
		for( int d = 0; d < m_depth; d++ ) delete [] m_bvecvec[d];
		delete [] m_bvecvec;
	}
}

//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
void SLIC::RGB2XYZ(
	const int&		sR,
	const int&		sG,
	const int&		sB,
	double&			X,
	double&			Y,
	double&			Z)
{
	double R = sR/255.0;
	double G = sG/255.0;
	double B = sB/255.0;

	double r, g, b;

	if(R <= 0.04045)	r = R/12.92;
	else				r = pow((R+0.055)/1.055,2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = pow((G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = pow((B+0.055)/1.055,2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void SLIC::RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X/Xr;
	double yr = Y/Yr;
	double zr = Z/Zr;

	double fx, fy, fz;
	if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
	else				fz = (kappa*zr + 16.0)/116.0;

	lval = 116.0*fy-16.0;
	aval = 500.0*(fx-fy);
	bval = 200.0*(fy-fz);
}

//===========================================================================
///	DoRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
void SLIC::DoRGBtoLABConversion(
	const unsigned int*&		ubuff,
	double*&					lvec,
	double*&					avec,
	double*&					bvec)
{
	int sz = m_width*m_height;
	lvec = new double[sz];
	avec = new double[sz];
	bvec = new double[sz];
	
	const double epsilon = 0.008856;	//actual CIE standard
	const double kappa   = 903.3;		//actual CIE standard
	__m256d v_epsilon = _mm256_set1_pd(0.008856);
	__m256d v_kappa = _mm256_set1_pd(903.3);

	const double Xr = 0.950456;	//reference white
	const double Yr = 1.0;		//reference white
	const double Zr = 1.088754;	//reference white
	__m256d v_Xr = _mm256_set1_pd(0.950456);
	__m256d v_Yr = _mm256_set1_pd(1.0);
	__m256d v_Zr = _mm256_set1_pd(1.088754);

	#pragma omp parallel for num_threads(numThreads) //schedule(static)
	for( int j = 0; j < sz-7; j+=8 )
	{
		int r[8],g[8],b[8];
		for(int t = 0;t < 8;t ++){
		     r[t] = (ubuff[j+t] >> 16) & 0xFF;
		     g[t] = (ubuff[j+t] >>  8) & 0xFF;
		     b[t] = (ubuff[j+t]      ) & 0xFF;
		}
		__m256d v_r =  _mm256_set_pd((double)r[3],(double)r[2],(double)r[1],(double)r[0]);
		__m256d v_g =  _mm256_set_pd((double)g[3],(double)g[2],(double)g[1],(double)g[0]);
		__m256d v_b =  _mm256_set_pd((double)b[3],(double)b[2],(double)b[1],(double)b[0]);
		__m256d v_r2 =  _mm256_set_pd((double)r[7],(double)r[6],(double)r[5],(double)r[4]);
		__m256d v_g2 =  _mm256_set_pd((double)g[7],(double)g[6],(double)g[5],(double)g[4]);
		__m256d v_b2 =  _mm256_set_pd((double)b[7],(double)b[6],(double)b[5],(double)b[4]);

		__m256i v_one = _mm256_set1_epi32(0xffffffff);
		__m256d v2550 =  _mm256_set1_pd(255.0);
		__m256d v1292 =  _mm256_set1_pd(12.92);
		__m256d v1055 =  _mm256_set1_pd(1.055);
		__m256d v4045 =  _mm256_set1_pd(0.04045);
		__m256d v24 =  _mm256_set1_pd(2.4);
		__m256d v55 =  _mm256_set1_pd(0.055);

		__m256d v_R = _mm256_div_pd(v_r, v2550);
		__m256d v_G = _mm256_div_pd(v_g, v2550);
		__m256d v_B = _mm256_div_pd(v_b, v2550);
		__m256d v_R2 = _mm256_div_pd(v_r2, v2550);
		__m256d v_G2 = _mm256_div_pd(v_g2, v2550);
		__m256d v_B2 = _mm256_div_pd(v_b2, v2550);

                __m256d R_mask = _mm256_cmp_pd (v_R, v4045, 2);
		__m256d G_mask = _mm256_cmp_pd (v_G, v4045, 2);
		__m256d B_mask = _mm256_cmp_pd (v_B, v4045, 2);
		__m256d R_mask_rev = _mm256_xor_pd(R_mask, (__m256d)v_one);
		__m256d G_mask_rev = _mm256_xor_pd(G_mask, (__m256d)v_one);
		__m256d B_mask_rev = _mm256_xor_pd(B_mask, (__m256d)v_one);

		__m256d R2_mask = _mm256_cmp_pd (v_R2, v4045, 2);
		__m256d G2_mask = _mm256_cmp_pd (v_G2, v4045, 2);
		__m256d B2_mask = _mm256_cmp_pd (v_B2, v4045, 2);
		__m256d R2_mask_rev = _mm256_xor_pd(R2_mask, (__m256d)v_one);
		__m256d G2_mask_rev = _mm256_xor_pd(G2_mask, (__m256d)v_one);
		__m256d B2_mask_rev = _mm256_xor_pd(B2_mask, (__m256d)v_one);

		__m256d v_RY = _mm256_div_pd(v_R, v1292);
		__m256d v_GY = _mm256_div_pd(v_G, v1292);
		__m256d v_BY = _mm256_div_pd(v_B, v1292);
		__m256d v_RN = _mm256_pow_pd(_mm256_div_pd(_mm256_add_pd(v_R,v55),v1055),v24);
		__m256d v_GN = _mm256_pow_pd(_mm256_div_pd(_mm256_add_pd(v_G,v55),v1055),v24);
		__m256d v_BN = _mm256_pow_pd(_mm256_div_pd(_mm256_add_pd(v_B,v55),v1055),v24);

		__m256d v_RY2 = _mm256_div_pd(v_R2, v1292);
		__m256d v_GY2 = _mm256_div_pd(v_G2, v1292);
		__m256d v_BY2 = _mm256_div_pd(v_B2, v1292);
		__m256d v_RN2 = _mm256_pow_pd(_mm256_div_pd(_mm256_add_pd(v_R2,v55),v1055),v24);
		__m256d v_GN2 = _mm256_pow_pd(_mm256_div_pd(_mm256_add_pd(v_G2,v55),v1055),v24);
		__m256d v_BN2 = _mm256_pow_pd(_mm256_div_pd(_mm256_add_pd(v_B2,v55),v1055),v24);

		__m256d v_tmp0 = _mm256_and_pd(v_RY, R_mask);
		__m256d v_tmp1 = _mm256_and_pd(v_RN, R_mask_rev);
		__m256d v_rNew = _mm256_add_pd(v_tmp0, v_tmp1);
		
		__m256d v_tmp2 = _mm256_and_pd(v_GY, G_mask);
		__m256d v_tmp3 = _mm256_and_pd(v_GN, G_mask_rev);
		__m256d v_gNew = _mm256_add_pd(v_tmp2, v_tmp3);
		
		__m256d v_tmp4 = _mm256_and_pd(v_BY, B_mask);
		__m256d v_tmp5 = _mm256_and_pd(v_BN, B_mask_rev);
		__m256d v_bNew = _mm256_add_pd(v_tmp4, v_tmp5);

                __m256d v_tmp02 = _mm256_and_pd(v_RY2, R2_mask);
                __m256d v_tmp12 = _mm256_and_pd(v_RN2, R2_mask_rev);
                __m256d v_rNew2 = _mm256_add_pd(v_tmp02, v_tmp12);
  
                __m256d v_tmp22 = _mm256_and_pd(v_GY2, G2_mask);
                __m256d v_tmp32 = _mm256_and_pd(v_GN2, G2_mask_rev);
                __m256d v_gNew2 = _mm256_add_pd(v_tmp22, v_tmp32);
  
                __m256d v_tmp42 = _mm256_and_pd(v_BY2, B2_mask);
                __m256d v_tmp52 = _mm256_and_pd(v_BN2, B2_mask_rev);
                __m256d v_bNew2 = _mm256_add_pd(v_tmp42, v_tmp52);

                __m256d vr0 =  _mm256_set1_pd(0.4124564);
		__m256d vg0 =  _mm256_set1_pd(0.3575761);
		__m256d vb0 =  _mm256_set1_pd(0.1804375);
		__m256d vr1 =  _mm256_set1_pd(0.2126729);
		__m256d vg1 =  _mm256_set1_pd(0.7151522);
		__m256d vb1 =  _mm256_set1_pd(0.0721750);
		__m256d vr2 =  _mm256_set1_pd(0.0193339);
		__m256d vg2 =  _mm256_set1_pd(0.1191920);
		__m256d vb2 =  _mm256_set1_pd(0.9503041);
		
		__m256d v_X = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(v_rNew,vr0),_mm256_mul_pd(v_gNew,vg0)),_mm256_mul_pd(v_bNew,vb0));
		__m256d v_Y = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(v_rNew,vr1),_mm256_mul_pd(v_gNew,vg1)),_mm256_mul_pd(v_bNew,vb1));
		__m256d v_Z = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(v_rNew,vr2),_mm256_mul_pd(v_gNew,vg2)),_mm256_mul_pd(v_bNew,vb2));
                
		__m256d v_X2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(v_rNew2,vr0),_mm256_mul_pd(v_gNew2,vg0)),_mm256_mul_pd(v_bNew2,vb0));
		__m256d v_Y2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(v_rNew2,vr1),_mm256_mul_pd(v_gNew2,vg1)),_mm256_mul_pd(v_bNew2,vb1));
		__m256d v_Z2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(v_rNew2,vr2),_mm256_mul_pd(v_gNew2,vg2)),_mm256_mul_pd(v_bNew2,vb2));

		__m256d v_xr = _mm256_div_pd(v_X,v_Xr);
		__m256d v_yr = _mm256_div_pd(v_Y,v_Yr);
		__m256d v_zr = _mm256_div_pd(v_Z,v_Zr);

                __m256d v_xr2 = _mm256_div_pd(v_X2,v_Xr);
		__m256d v_yr2 = _mm256_div_pd(v_Y2,v_Yr);
		__m256d v_zr2 = _mm256_div_pd(v_Z2,v_Zr);

                __m256d v116 =  _mm256_set1_pd(116.0);
		__m256d v16 =  _mm256_set1_pd(16.0);
		__m256d v13 =  _mm256_set1_pd(1.0/3.0);
		__m256d xr_mask = _mm256_cmp_pd (v_xr, v_epsilon, 14);
		__m256d yr_mask = _mm256_cmp_pd (v_yr, v_epsilon, 14);
		__m256d zr_mask = _mm256_cmp_pd (v_zr, v_epsilon, 14);
		__m256d xr_mask_rev = _mm256_xor_pd(xr_mask, (__m256d)v_one);
		__m256d yr_mask_rev = _mm256_xor_pd(yr_mask, (__m256d)v_one);
		__m256d zr_mask_rev = _mm256_xor_pd(zr_mask, (__m256d)v_one);
		
                __m256d xr2_mask = _mm256_cmp_pd (v_xr2, v_epsilon, 14);
		__m256d yr2_mask = _mm256_cmp_pd (v_yr2, v_epsilon, 14);
		__m256d zr2_mask = _mm256_cmp_pd (v_zr2, v_epsilon, 14);
		__m256d xr2_mask_rev = _mm256_xor_pd(xr2_mask, (__m256d)v_one);
		__m256d yr2_mask_rev = _mm256_xor_pd(yr2_mask, (__m256d)v_one);
		__m256d zr2_mask_rev = _mm256_xor_pd(zr2_mask, (__m256d)v_one);

		__m256d v_fxY = _mm256_pow_pd(v_xr,v13);
		__m256d v_fyY = _mm256_pow_pd(v_yr,v13);
		__m256d v_fzY = _mm256_pow_pd(v_zr,v13);
		__m256d v_fxN = _mm256_div_pd(_mm256_add_pd(_mm256_mul_pd(v_kappa,v_xr),v16),v116);
		__m256d v_fyN = _mm256_div_pd(_mm256_add_pd(_mm256_mul_pd(v_kappa,v_yr),v16),v116);
		__m256d v_fzN = _mm256_div_pd(_mm256_add_pd(_mm256_mul_pd(v_kappa,v_zr),v16),v116);

		__m256d v_fxY2 = _mm256_pow_pd(v_xr2,v13);
		__m256d v_fyY2 = _mm256_pow_pd(v_yr2,v13);
		__m256d v_fzY2 = _mm256_pow_pd(v_zr2,v13);
		__m256d v_fxN2 = _mm256_div_pd(_mm256_add_pd(_mm256_mul_pd(v_kappa,v_xr2),v16),v116);
		__m256d v_fyN2 = _mm256_div_pd(_mm256_add_pd(_mm256_mul_pd(v_kappa,v_yr2),v16),v116);
		__m256d v_fzN2 = _mm256_div_pd(_mm256_add_pd(_mm256_mul_pd(v_kappa,v_zr2),v16),v116);

		
		__m256d v_tmp6 = _mm256_and_pd(v_fxY, xr_mask);
		__m256d v_tmp7 = _mm256_and_pd(v_fxN, xr_mask_rev);
		__m256d v_fx = _mm256_add_pd(v_tmp6, v_tmp7);
		
		__m256d v_tmp8 = _mm256_and_pd(v_fyY, yr_mask);
		__m256d v_tmp9 = _mm256_and_pd(v_fyN, yr_mask_rev);
		__m256d v_fy = _mm256_add_pd(v_tmp8, v_tmp9);
		
		__m256d v_tmp10 = _mm256_and_pd(v_fzY, zr_mask);
		__m256d v_tmp11 = _mm256_and_pd(v_fzN, zr_mask_rev);
		__m256d v_fz = _mm256_add_pd(v_tmp10, v_tmp11);

		__m256d v_tmp62 = _mm256_and_pd(v_fxY2, xr2_mask);
		__m256d v_tmp72 = _mm256_and_pd(v_fxN2, xr2_mask_rev);
		__m256d v_fx2 = _mm256_add_pd(v_tmp62, v_tmp72);
		
		__m256d v_tmp82 = _mm256_and_pd(v_fyY2, yr2_mask);
		__m256d v_tmp92 = _mm256_and_pd(v_fyN2, yr2_mask_rev);
		__m256d v_fy2 = _mm256_add_pd(v_tmp82, v_tmp92);
		
		__m256d v_tmp102 = _mm256_and_pd(v_fzY2, zr2_mask);
		__m256d v_tmp112 = _mm256_and_pd(v_fzN2, zr2_mask_rev);
		__m256d v_fz2 = _mm256_add_pd(v_tmp102, v_tmp112);

		__m256d v500 =  _mm256_set1_pd(500.0);
		__m256d v200 =  _mm256_set1_pd(200.0);
		__m256d v_lvec = _mm256_sub_pd(_mm256_mul_pd(v116,v_fy),v16);
		__m256d v_avec = _mm256_mul_pd(_mm256_sub_pd(v_fx,v_fy),v500);
		__m256d v_bvec = _mm256_mul_pd(_mm256_sub_pd(v_fy,v_fz),v200);

		__m256d v_lvec2 = _mm256_sub_pd(_mm256_mul_pd(v116,v_fy2),v16);
		__m256d v_avec2 = _mm256_mul_pd(_mm256_sub_pd(v_fx2,v_fy2),v500);
		__m256d v_bvec2 = _mm256_mul_pd(_mm256_sub_pd(v_fy2,v_fz2),v200);
		
		_mm256_storeu_pd(lvec + j, v_lvec);
		_mm256_storeu_pd(avec + j, v_avec);
		_mm256_storeu_pd(bvec + j, v_bvec);

		_mm256_storeu_pd(lvec + j + 4, v_lvec2);
		_mm256_storeu_pd(avec + j + 4, v_avec2);
		_mm256_storeu_pd(bvec + j + 4, v_bvec2);

	}
    //#pragma omp parallel for num_threads(64)
	for(int j = sz - sz%8; j < sz; j++){
	        int r = (ubuff[j] >> 16) & 0xFF;
	        int g = (ubuff[j] >>  8) & 0xFF;
	        int b = (ubuff[j]      ) & 0xFF;
	        //RGB2LAB( r, g, b, lvec[j], avec[j], bvec[j] );
	        //sRGB to XYZ conversion
	        double X, Y, Z;
	        double R = r / 255.0;
	        double G = g / 255.0;
	        double B = b / 255.0;
	        double rNew, gNew, bNew;
	        if(R <= 0.04045)  rNew = R/12.92;
	        else              rNew = pow((R+0.055)/1.055,2.4);
	        if(G <= 0.04045)  gNew = G/12.92;
	        else              gNew = pow((G+0.055)/1.055,2.4);
	        if(B <= 0.04045)  bNew = B/12.92;
	        else              bNew = pow((B+0.055)/1.055,2.4);
	        X = rNew*0.4124564 + gNew*0.3575761 + bNew*0.1804375;
	        Y = rNew*0.2126729 + gNew*0.7151522 + bNew*0.0721750;
	        Z = rNew*0.0193339 + gNew*0.1191920 + bNew*0.9503041;
	        //XYZ to LAB conversion
	        double xr = X/Xr;
	        double yr = Y/Yr;
	        double zr = Z/Zr;
	        double fx, fy, fz;
	        if(xr > epsilon)  fx = pow(xr, 1.0/3.0);
	        else              fx = (kappa*xr + 16.0)/116.0;
	        if(yr > epsilon)  fy = pow(yr, 1.0/3.0);
	        else              fy = (kappa*yr + 16.0)/116.0;
	        if(zr > epsilon)  fz = pow(zr, 1.0/3.0);
	        else              fz = (kappa*zr + 16.0)/116.0;
	        lvec[j] = 116.0*fy-16.0;
	        avec[j] = 500.0*(fx-fy);
	        bvec[j] = 200.0*(fy-fz);
	
	 }

//	int sz = m_width*m_height;
//	lvec = new double[sz];
//	avec = new double[sz];
//	bvec = new double[sz];
//	
//	const double epsilon = 0.008856;	//actual CIE standard
//	const double kappa   = 903.3;		//actual CIE standard
//
//	const double Xr = 0.950456;	//reference white
//	const double Yr = 1.0;		//reference white
//	const double Zr = 1.088754;	//reference white
//	#ifdef DEBUG
//	cerr << "the m_width is: " << m_width  << endl;
//	cerr << "the m_height is: " << m_height  << endl;
//	cerr << "the sz is: " << sz  << endl;
//	#endif
//
//
//	//can be vectorization addbyxxm 2021/7/3
//	#pragma omp parallel for num_threads(numThreads) //schedule(static)
//	for( int j = 0; j < sz; j++ )
//	//for( int j = ll; j < rr; j++ )
//	{
//		int r = (ubuff[j] >> 16) & 0xFF;
//		int g = (ubuff[j] >>  8) & 0xFF;
//		int b = (ubuff[j]      ) & 0xFF;
//
//		//RGB2LAB( r, g, b, lvec[j], avec[j], bvec[j] );
//		
//		////sRGB to XYZ conversion
//		double X, Y, Z;
//		double R = r / 255.0;
//		double G = g / 255.0;
//		double B = b / 255.0;
//		
//		double rNew, gNew, bNew;
//		if(R <= 0.04045)	rNew = R/12.92;
//		else				rNew = pow((R+0.055)/1.055,2.4);
//		if(G <= 0.04045)	gNew = G/12.92;
//		else				gNew = pow((G+0.055)/1.055,2.4);
//		if(B <= 0.04045)	bNew = B/12.92;
//		else				bNew = pow((B+0.055)/1.055,2.4);
//	
//		X = rNew*0.4124564 + gNew*0.3575761 + bNew*0.1804375;
//		Y = rNew*0.2126729 + gNew*0.7151522 + bNew*0.0721750;
//		Z = rNew*0.0193339 + gNew*0.1191920 + bNew*0.9503041;
//
//		////XYZ to LAB conversion
//
//		double xr = X/Xr;
//		double yr = Y/Yr;
//		double zr = Z/Zr;
//
//		double fx, fy, fz;
//		if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
//		else				fx = (kappa*xr + 16.0)/116.0;
//		if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
//		else				fy = (kappa*yr + 16.0)/116.0;
//		if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
//		else				fz = (kappa*zr + 16.0)/116.0;
//
//		lvec[j] = 116.0*fy-16.0;
//		avec[j] = 500.0*(fx-fy);
//		bvec[j] = 200.0*(fy-fz);
//
//	}

}

//==============================================================================
///	DetectLabEdges
//==============================================================================
void SLIC::DetectLabEdges(
	const double*				lvec,
	const double*				avec,
	const double*				bvec,
	const int&					width,
	const int&					height,
	vector<double>&				edges)
{
	int sz = width*height;

	edges.resize(sz,0);
	int start = 1 * width + 1;
	int end = (height-2) * width + width - 2;
	#pragma omp parallel for num_threads(numThreads) ////schedule(dynamic)
	//#pragma ivdep
	for(int i = start; i < end; i++){
	//for( int j = 1; j < height-1; j++ )
	//{
	//	for( int k = 1; k < width-1; k++ )
	//	{
	//		int i = j*width+k;

			double dx = (lvec[i-1] - lvec[i+1]) * (lvec[i-1] - lvec[i+1]) +
						(avec[i-1] - avec[i+1]) * (avec[i-1] - avec[i+1]) +
						(bvec[i-1] - bvec[i+1]) * (bvec[i-1] - bvec[i+1]);

			double dy = (lvec[i-width] - lvec[i+width]) * (lvec[i-width] - lvec[i+width]) +
						(avec[i-width] - avec[i+width]) * (avec[i-width] - avec[i+width]) +
						(bvec[i-width] - bvec[i+width]) * (bvec[i-width] - bvec[i+width]);

			//edges[i] = (sqrt(dx) + sqrt(dy));
			edges[i] = (dx + dy);
	//	}
	//}
	}
}

//===========================================================================
///	PerturbSeeds
//===========================================================================
void SLIC::PerturbSeeds(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const vector<double>&		edges)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	
	int numseeds = kseedsl.size();

	for( int n = 0; n < numseeds; n++ )
	{
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for( int i = 0; i < 8; i++ )
		{
			int nx = ox+dx8[i];//new x
			int ny = oy+dy8[i];//new y

			if( nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = ny*m_width + nx;
				if( edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if(storeind != oind)
		{
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = storeind/m_width;
			kseedsl[n] = m_lvec[storeind];
			kseedsa[n] = m_avec[storeind];
			kseedsb[n] = m_bvec[storeind];
		}
	}
}

//===========================================================================
///	GetLABXYSeeds_ForGivenK
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC::GetLABXYSeeds_ForGivenK(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					K,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int sz = m_width*m_height;
	double step = sqrt(double(sz)/double(K));
	int T = step;
	int xoff = step/2;
	int yoff = step/2;
	
	int n(0);int r(0);
	for( int y = 0; y < m_height; y++ )
	{
		int Y = y*step + yoff;
		if( Y > m_height-1 ) break;

		for( int x = 0; x < m_width; x++ )
		{
			//int X = x*step + xoff;//square grid
			int X = x*step + (xoff<<(r&0x1));//hex grid
			if(X > m_width-1) break;

			int i = Y*m_width + X;

			//_ASSERT(n < K);
			
			//kseedsl[n] = m_lvec[i];
			//kseedsa[n] = m_avec[i];
			//kseedsb[n] = m_bvec[i];
			//kseedsx[n] = X;
			//kseedsy[n] = Y;
			kseedsl.push_back(m_lvec[i]);
			kseedsa.push_back(m_avec[i]);
			kseedsb.push_back(m_bvec[i]);
			kseedsx.push_back(X);
			kseedsy.push_back(Y);
			n++;
		}
		r++;
	}

	if(perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemag);
	}
}

//===========================================================================
///	PerformSuperpixelSegmentation_VariableSandM
///
///	Magic SLIC - no parameters
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
/// This function picks the maximum value of color distance as compact factor
/// M and maximum pixel distance as grid step size S from each cluster (13 April 2011).
/// So no need to input a constant value of M and S. There are two clear
/// advantages:
///
/// [1] The algorithm now better handles both textured and non-textured regions
/// [2] There is not need to set any parameters!!!
///
/// SLICO (or SLIC Zero) dynamically varies only the compactness factor S,
/// not the step size S.
//===========================================================================

//void SLIC::PerformSuperpixelSegmentation_VariableSandM(
//	vector<double>&				kseedsl,
//	vector<double>&				kseedsa,
//	vector<double>&				kseedsb,
//	vector<double>&				kseedsx,
//	vector<double>&				kseedsy,
//	int*						klabels,
//	const int&					STEP,
//	const int&					NUMITR)
//{
void SLIC::PerformSuperpixelSegmentation_VariableSandM(
	double*				kseedsl,
	double*				kseedsa,
	double*				kseedsb,
	double*				kseedsx,
	double*				kseedsy,
	int*						klabels,
	const int&					numk,
	const int&					STEP,
	const int&					NUMITR)
{
	int sz = m_width*m_height;
	//const int numk = kseedsl.size();
	#ifdef DEBUG
	cerr << "numk is: " << numk << endl;
	cerr << "step is: " << STEP << endl;
	#endif

	#ifdef DEBUG
    auto startTime = Clock::now();
	#endif
	//double cumerr(99999.9);
	int numitr(0);

	//----------------
	int offset = STEP;
	if(STEP < 10) offset = STEP*1.5;
	//----------------

	//vector<double> sigmal(numk, 0);
	//vector<double> sigmaa(numk, 0);
	//vector<double> sigmab(numk, 0);
	//vector<double> sigmax(numk, 0);
	//vector<double> sigmay(numk, 0);
	//vector<int> clustersize(numk, 0);
	//vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	//vector<double> distxy(sz, DBL_MAX);
	//vector<double> distlab(sz, DBL_MAX);
	//vector<double> distvec(sz, DBL_MAX);
	//vector<double> maxlab(numk, 10*10);//THIS IS THE VARIABLE VALUE OF M, just start with 10
	//vector<double> maxxy(numk, STEP*STEP);//THIS IS THE VARIABLE VALUE OF M, just start with 10

	//double *sigmal = new double[numk];
	//double *sigmaa = new double[numk];
	//double *sigmab = new double[numk];
	//double *sigmax = new double[numk];
	//double *sigmay = new double[numk];
	//double *clustersize = new double[numk];
	//double *inv = new double[numk];
	//double * maxlab = new double[numk];
	
	double sigmal[numk];
	double sigmaa[numk];
	double sigmab[numk];
	double sigmax[numk];
	double sigmay[numk];
	double inv[numk];
	double maxlab[numk];
	int clustersize[numk];
	
	//int numThreads = 48;
	double sigmalArr[numThreads*numk];
	double sigmaaArr[numThreads*numk];
	double sigmabArr[numThreads*numk];
	double sigmaxArr[numThreads*numk];
	double sigmayArr[numThreads*numk];
	int clustersizeArr[numThreads*numk];

	double sil[numk];
	double sia[numk];
	double sib[numk];
	double six[numk];
	double siy[numk];
	double mxab[numk];
	int csiz[numk];
	if(my_rank == 0){
		for(int i = 0; i < numk; i++){
			sil[i] = 0.;
			sia[i] = 0.;
			sib[i] = 0.;
			six[i] = 0.;
			siy[i] = 0.;
			mxab[i] = 0.;
			csiz[i] = 0;
		}
	}

	double maxlabArr[numThreads*numk];
	for(int t = 0; t < numThreads*numk; t++){
		maxlabArr[t] = 100;
	}
	
	double * distxy = new double[sz];
	double * distlab = new double[sz];
	double * distvec = new double[sz];
	int64_t *klabels_64 = new int64_t[sz];

	#pragma omp parallel for num_threads(numThreads)
	for(int i = 0; i < sz; i++)
	{
		distxy[i] = DBL_MAX;
		distlab[i] = DBL_MAX;
		klabels_64[i] = (int64_t)klabels[i];

	}

	
	for(int i = 0 ; i < numk; i++)
	{
		maxlab[i] = 100;
		inv[i] = 0.;
	}

	double invxywt = 1.0/(STEP*STEP);//NOTE: this is different from how usual SLIC/LKM works
	#ifdef DEBUG
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout <<  "init time is: " << compTime.count()/1000 << " ms" << endl;
	#endif

	//time is calculated by ns not ms.
	#ifdef CLEAR_DEBUG
	int time_init_distvec = 0;
	int time_cvt_labels = 0;
	int time_update_maxlab = 0; 
	int time_init_sigmal_cluster = 0;
	int time_update_sigmal_cluster = 0;
	int time_update_kseeds = 0;
	#endif


	while( numitr < NUMITR )
	{
		//------
		//cumerr = 0;
		numitr++;
		//------

		//distvec.assign(sz, DBL_MAX);
		#ifdef CLEAR_DEBUG
		auto tmpTime0 = Clock::now();
		#endif

		#pragma omp parallel for num_threads(numThreads)//middle numThreas
		for(int i = 0; i < sz; i++)
			distvec[i] = DBL_MAX;
		#ifdef CLEAR_DEBUG
    	auto tmpTime1 = Clock::now();
    	auto subTime0 = chrono::duration_cast<chrono::microseconds>(tmpTime1 - tmpTime0);
		time_init_distvec += subTime0.count();
		#endif

		#ifdef DEBUG
    	startTime = Clock::now();
		#endif

		int linePerNode = ceil(1.0 * m_height / num_procs);

		//int procLL[num_procs];
		//int procRR[num_procs];
		//procLL[0] = 0;
		//procLL[1] = linePerNode - 100;
		//procRR[0] = linePerNode - 100;
		//procRR[1] = m_height;
		//int lline = procLL[my_rank];
		//int rline = procRR[my_rank];

		int lline = my_rank * linePerNode;//0, 1949
		int rline = min(m_height, (my_rank+1) * linePerNode);//1949, 3898

		int ll = lline * m_width;
		int rr = rline * m_width;

		for( int n = 0; n < numk; n++ )
		{
			int y1 = max(0,			(int)(kseedsy[n]-offset));
			int y2 = min(m_height,	(int)(kseedsy[n]+offset));
			int x1 = max(0,			(int)(kseedsx[n]-offset));
			int x2 = min(m_width,	(int)(kseedsx[n]+offset));
			if(y1 >= rline || y2 < lline) continue;
			if(y1 < rline && y2 > rline){
				//printf("rank %d, lline %d, rline %d, y1 %d, y2 %d\n", my_rank, lline, rline, y1, y2);
				y1 = y1;
				y2 = rline;
			}
			if(y1 < lline && y2 >=lline){
				//printf("rank %d, lline %d, rline %d, y1 %d, y2 %d\n", my_rank, lline, rline, y1, y2);
				y1 = lline; 
				y2 = y2;
			}
			//printf("rank: %d, y1: %d, y2: %d \n", my_rank, y1, y2);
			//continue;

			double cur_sl = kseedsl[n];
			double cur_sa = kseedsa[n];
			double cur_sb = kseedsb[n];
			double cur_sx = kseedsx[n];
			double cur_sy = kseedsy[n];
			double cur_maxlab = maxlab[n];

			__m256d v_sl = _mm256_set1_pd(kseedsl[n]);
			__m256d v_sa = _mm256_set1_pd(kseedsa[n]);
			__m256d v_sb = _mm256_set1_pd(kseedsb[n]);
			__m256d v_sx = _mm256_set1_pd(kseedsx[n]);
			__m256d v_sy = _mm256_set1_pd(kseedsy[n]);
			__m256d v_maxlab = _mm256_set1_pd(maxlab[n]);

			__m256d v_invxywt = _mm256_set1_pd(invxywt);
			__m256i v_one = _mm256_set1_epi32(0xffffffff);
			__m256i v_n = _mm256_set1_epi32(n);
			
			//if(my_rank == 1){
			//	printf("my_rank %d, y1 %d, y2 %d, x1 %d, x2 %d, n %d \n", my_rank, y1, y2, x1, x2, n);
			//}

			#pragma omp parallel for num_threads(numThreads)
			for( int y = y1; y < y2; y++ )
			{
				__m256d v_y = _mm256_set1_pd((double)y);

				for( int x = x1; x < x2 - 3; x+=4 )
				{
					int i = y*m_width + x;
					//_ASSERT( y < m_height && x < m_width && y >= 0 && x >= 0 );

					//double l = m_lvec[i];
					//double a = m_avec[i];
					//double b = m_bvec[i];

					//distlab[i] =	(l - kseedsl[n])*(l - kseedsl[n]) +
					//				(a - kseedsa[n])*(a - kseedsa[n]) +
					//				(b - kseedsb[n])*(b - kseedsb[n]);

					//distxy[i] =		(x - kseedsx[n])*(x - kseedsx[n]) +
					//				(y - kseedsy[n])*(y - kseedsy[n]);
					
					//distlab[i] = 	(l - cur_sl) * (l - cur_sl) +
					//				(a - cur_sa) * (a - cur_sa) +
					//				(b - cur_sb) * (b - cur_sb);
					//distxy[i] = 	(x - cur_sx) * (x - cur_sx) +
					//				(y - cur_sy) * (y - cur_sy);
									

					////------------------------------------------------------------------------
					////double dist = distlab[i]/maxlab[n] + distxy[i]*invxywt;//only varying m, prettier superpixels
					//double dist = distlab[i]/cur_maxlab + distxy[i]*invxywt;//only varying m, prettier superpixels
					////double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
					////------------------------------------------------------------------------

					__m256d v_x = _mm256_set_pd((double)x+3, (double)x+2, (double)x+1, (double)x+0);

					__m256d v_l = _mm256_loadu_pd(m_lvec + i);
					__m256d v_a = _mm256_loadu_pd(m_avec + i);
					__m256d v_b = _mm256_loadu_pd(m_bvec + i);

					__m256d v_sub_l = _mm256_sub_pd(v_l, v_sl);
					__m256d v_sub_a = _mm256_sub_pd(v_a, v_sa);
					__m256d v_sub_b = _mm256_sub_pd(v_b, v_sb);
					__m256d v_sub_x = _mm256_sub_pd(v_x, v_sx);
					__m256d v_sub_y = _mm256_sub_pd(v_y, v_sy);

					//__m256d v_square_l = _mm256_mul_pd(v_sub_l, v_sub_l);
					//__m256d v_square_a = _mm256_mul_pd(v_sub_a, v_sub_a);
					//__m256d v_square_b = _mm256_mul_pd(v_sub_b, v_sub_b);
					//__m256d v_square_x = _mm256_mul_pd(v_sub_x, v_sub_x);
					//__m256d v_square_y = _mm256_mul_pd(v_sub_y, v_sub_y);
					//
					//__m256d v_distlab = _mm256_add_pd(_mm256_add_pd(v_square_l, v_square_a), v_square_b);
					//__m256d v_distxy = _mm256_add_pd(v_square_x, v_square_y);

					__m256d v_distlab = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(v_sub_l, v_sub_l), _mm256_mul_pd(v_sub_a, v_sub_a)), _mm256_mul_pd(v_sub_b, v_sub_b));
					__m256d v_distxy = _mm256_add_pd(_mm256_mul_pd(v_sub_x, v_sub_x), _mm256_mul_pd(v_sub_y, v_sub_y));
					_mm256_storeu_pd(distlab+i, v_distlab);
					//_mm256_storeu_pd(distxy.data()+i, v_distxy);

					__m256d v_dist = _mm256_add_pd(_mm256_div_pd(v_distlab, v_maxlab), _mm256_mul_pd(v_distxy, v_invxywt));

					//if( dist < distvec[i] )
					//{
					//	distvec[i] = dist;
					//	klabels[i]  = n;
					//}
					
					//__m256d v_distvec = _mm256_loadu_pd(distvec.data() + i);
					__m256d v_distvec = _mm256_loadu_pd(distvec + i);
					__m256i v_klabels = _mm256_loadu_si256((__m256i*)(klabels_64 + i));
					

					__m256d v_mask = _mm256_cmp_pd(v_dist, v_distvec, 1);
					__m256d v_mask_rev = _mm256_xor_pd(v_mask, (__m256d)v_one);
					
					__m256d v_tmp0 = _mm256_and_pd(v_dist, v_mask);
					__m256d v_tmp1 = _mm256_and_pd(v_distvec, v_mask_rev);
					__m256i v_tmp2 = _mm256_and_si256(v_n, (__m256i)v_mask);
					__m256i v_tmp3 = _mm256_and_si256(v_klabels, (__m256i)v_mask_rev);

					v_distvec = _mm256_add_pd(v_tmp0, v_tmp1);
					v_klabels = _mm256_add_epi64(v_tmp2, v_tmp3);

					//_mm256_storeu_pd(distvec.data() + i, v_distvec);
					_mm256_storeu_pd(distvec + i, v_distvec);
					_mm256_storeu_si256((__m256i*)(klabels_64+i), v_klabels);

				}//end for x_body

				for( int x = x2 - (x2-x1)%4; x < x2; x++)
				{
					int i = y*m_width + x;

					double l = m_lvec[i];
					double a = m_avec[i];
					double b = m_bvec[i];

					//distlab[i] =	(l - kseedsl[n])*(l - kseedsl[n]) +
					//				(a - kseedsa[n])*(a - kseedsa[n]) +
					//				(b - kseedsb[n])*(b - kseedsb[n]);

					//distxy[i] =	(x - kseedsx[n])*(x - kseedsx[n]) +
					//				(y - kseedsy[n])*(y - kseedsy[n]);
					distlab[i] = 	(l - cur_sl) * (l - cur_sl) +
									(a - cur_sa) * (a - cur_sa) +
									(b - cur_sb) * (b - cur_sb);
					distxy[i] = 	(x - cur_sx) * (x - cur_sx) +
									(y - cur_sy) * (y - cur_sy);
									

					//------------------------------------------------------------------------
					//double dist = distlab[i]/maxlab[n] + distxy[i]*invxywt;//only varying m, prettier superpixels
					double dist = distlab[i]/cur_maxlab + distxy[i]*invxywt;//only varying m, prettier superpixels
					//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
					//------------------------------------------------------------------------
					
					if( dist < distvec[i] )
					{
						distvec[i] = dist;
						//klabels[i]  = n;
						klabels_64[i]  = n;
					}
				}//end for x_tail
				
			}//end for y

		}
	//	if(numitr == 2 && my_rank == 0){
	//		for(int i = ll; i < rr; i++){
	//			printf("distvec[%d] = %lf, klabels[%d] = %d\n", i, distvec[i], i, klabels[i]);
	//		}
	//		//exit(0);
	//	}
	//	if(numitr == 2) exit(0);

		#ifdef DEBUG
    	endTime = Clock::now();
    	compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    	cout << my_rank <<  " update distvec time=" << compTime.count()/1000 << " ms" << endl;
		//cerr << "the bandwidth is: " << (double)totalNumber * 1000000 / compTime.count() / 1024/1024/1024 << " GB/s" << endl;
		#endif
		
		#ifdef CLEAR_DEBUG
		tmpTime0 = Clock::now();
		#endif

		if(numitr == NUMITR){
			#pragma omp parallel for num_threads(numThreads)
			for(int i = ll; i < rr; i++){
				klabels[i] = (int)klabels_64[i];
			}

			if(my_rank == 0){
				MPI_Recv(klabels + rr, sz - (rr -ll), MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			else{
				MPI_Send(klabels + ll, rr -ll, MPI_INT, 0, 1, MPI_COMM_WORLD);
			}
		}

		#ifdef CLEAR_DEBUG
    	tmpTime1 = Clock::now();
    	subTime0 = chrono::duration_cast<chrono::microseconds>(tmpTime1 - tmpTime0);
		time_cvt_labels += subTime0.count();
		#endif

		//-----------------------------------------------------------------
		// Assign the max color distance for a cluster
		//-----------------------------------------------------------------
		//if(0 == numitr)
		//{
		//	maxlab.assign(numk,1);
		//	maxxy.assign(numk,1);
		//}
		
		
		#ifdef CLEAR_DEBUG
		tmpTime0 = Clock::now();
		#endif

		#pragma omp parallel for num_threads(numThreads)
		//for( int i = 0; i < sz; i++ )
		for( int i = ll; i < rr; i++ )
		{
			int tid = omp_get_thread_num();
			if(maxlabArr[tid*numk + (int)klabels_64[i]] < distlab[i]) maxlabArr[tid*numk + (int)klabels_64[i]] = distlab[i];
			//if(maxlab[klabels[i]] < distlab[i]) maxlab[klabels[i]] = distlab[i];
			////if(maxxy[klabels[i]] < distxy[i]) maxxy[klabels[i]] = distxy[i];
		}

		for(int k = 0; k < numk; k++){
			for(int t = 0; t < numThreads; t++){
				if(maxlab[k] < maxlabArr[t*numk + k]) maxlab[k] = maxlabArr[t*numk + k];
			}
		}
		//cerr << my_rank << " end update maxlab " << endl;
		//exit(-1);

		#ifdef CLEAR_DEBUG
    	tmpTime1 = Clock::now();
    	subTime0 = chrono::duration_cast<chrono::microseconds>(tmpTime1 - tmpTime0);
		time_update_maxlab += subTime0.count();
		#endif

		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		
		//sigmal.assign(numk, 0);
		//sigmaa.assign(numk, 0);
		//sigmab.assign(numk, 0);
		//sigmax.assign(numk, 0);
		//sigmay.assign(numk, 0);
		//clustersize.assign(numk, 0);
		
		#ifdef CLEAR_DEBUG
		tmpTime0 = Clock::now();
		#endif

		for(int i = 0; i < numk; i++)
		{
			sigmal[i] = 0;
			sigmaa[i] = 0;
			sigmab[i] = 0;
			sigmax[i] = 0;
			sigmay[i] = 0;
			clustersize[i] = 0;
		}


		//#pragma omp parallel for num_threads(2)
		for(int t = 0; t < numThreads*numk; t++){
			sigmalArr[t] = 0.0;
			sigmaaArr[t] = 0.0;
			sigmabArr[t] = 0.0;
			sigmaxArr[t] = 0.0;
			sigmayArr[t] = 0.0;
			clustersizeArr[t] = 0;
		}
		#ifdef CLEAR_DEBUG
    	tmpTime1 = Clock::now();
    	subTime0 = chrono::duration_cast<chrono::microseconds>(tmpTime1 - tmpTime0);
		time_init_sigmal_cluster += subTime0.count();
		#endif

		#ifdef CLEAR_DEBUG
		tmpTime0 = Clock::now();
		#endif
		#pragma omp parallel for num_threads(numThreads)
		//for( int j = 0; j < sz; j++ )
		for( int j = ll; j < rr; j++ )
		{
			//int temp = klabels[j];
			int threads_id = omp_get_thread_num();
			sigmalArr[threads_id * numk + (int)klabels_64[j]] += m_lvec[j];
			sigmaaArr[threads_id * numk + (int)klabels_64[j]] += m_avec[j];
			sigmabArr[threads_id * numk + (int)klabels_64[j]] += m_bvec[j];
			sigmaxArr[threads_id * numk + (int)klabels_64[j]] += (j%m_width);
			sigmayArr[threads_id * numk + (int)klabels_64[j]] += (j/m_width);

			clustersizeArr[threads_id * numk + (int)klabels_64[j]]++;
			//sigmalArr[threads_id * numk + temp] += m_lvec[j];
			//_ASSERT(klabels[j] >= 0);
		//	sigmal[klabels[j]] += m_lvec[j];
		//	sigmaa[klabels[j]] += m_avec[j];
		//	sigmab[klabels[j]] += m_bvec[j];
		//	sigmax[klabels[j]] += (j%m_width);
		//	sigmay[klabels[j]] += (j/m_width);

		//	clustersize[klabels[j]]++;
		}

		//#pragma omp parallel for num_threads(2)
		for(int k = 0; k < numk; k++){
			for(int t = 0; t < numThreads; t++){
				sigmal[k] += sigmalArr[t * numk + k];
				sigmaa[k] += sigmaaArr[t * numk + k];
				sigmab[k] += sigmabArr[t * numk + k];
				sigmax[k] += sigmaxArr[t * numk + k];
				sigmay[k] += sigmayArr[t * numk + k];
				clustersize[k] += clustersizeArr[t * numk + k];
			}
		}

		if(my_rank == 0){
			MPI_Recv(sil, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(sia, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(sib, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(six, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(siy, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(mxab, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(csiz, numk, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for(int i = 0; i < numk; i++){
				//printf("mxab[%d] is: %lf\n", i, mxab[i]);
				//printf("sil[%d] is: %lf\n", i, sil[i]);
				//printf("sia[%d] is: %lf\n", i, sia[i]);
				//printf("sib[%d] is: %lf\n", i, sib[i]);
				//printf("six[%d] is: %lf\n", i, six[i]);
				//printf("siy[%d] is: %lf\n", i, siy[i]);
				//printf("csiz[%d] is: %d\n", i, csiz[i]);// TODO have bugs
			}
			for(int k = 0; k < numk; k++){
				double inv = 1.0 / (clustersize[k] + csiz[k]);
				kseedsl[k] = (sigmal[k] + sil[k]) *inv;
				kseedsa[k] = (sigmaa[k] + sia[k]) *inv;
				kseedsb[k] = (sigmab[k] + sib[k]) *inv;
				kseedsx[k] = (sigmax[k] + six[k]) *inv;
				kseedsy[k] = (sigmay[k] + siy[k]) *inv;
				maxlab[k] = max(maxlab[k], mxab[k]);
			}
			//for(int i = 0; i < numk; i++){
			//	printf("maxlab[%d] is: %lf\n", i, maxlab[i]);
			//	printf("kseedsl[%d] is: %lf\n", i, kseedsl[i]);
			//	printf("kseedsa[%d] is: %lf\n", i, kseedsa[i]);
			//	printf("kseedsb[%d] is: %lf\n", i, kseedsb[i]);
			//	printf("kseedsx[%d] is: %lf\n", i, kseedsx[i]);
			//	printf("kseedsy[%d] is: %lf\n", i, kseedsy[i]);
			//}
			
			//cerr << "end mpi recv sigmal etc =======================================================================================" << endl;

		}
		else{
			MPI_Send(sigmal, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			MPI_Send(sigmaa, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			MPI_Send(sigmab, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			MPI_Send(sigmax, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			MPI_Send(sigmay, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			MPI_Send(maxlab, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
			MPI_Send(clustersize, numk, MPI_INT, 0, 1, MPI_COMM_WORLD);
		}
		//exit(0);


		if(my_rank == 0){
			MPI_Send(kseedsl, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
			MPI_Send(kseedsa, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
			MPI_Send(kseedsb, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
			MPI_Send(kseedsx, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
			MPI_Send(kseedsy, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
			MPI_Send(maxlab, numk, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
		}
		else{
			MPI_Recv(kseedsl, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(kseedsa, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(kseedsb, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(kseedsx, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(kseedsy, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(maxlab, numk, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		#ifdef CLEAR_DEBUG
    	tmpTime1 = Clock::now();
    	subTime0 = chrono::duration_cast<chrono::microseconds>(tmpTime1 - tmpTime0);
		time_update_sigmal_cluster += subTime0.count();
		#endif

		#ifdef CLEAR_DEBUG
		tmpTime0 = Clock::now();
		#endif
		//if(my_rank == 1){
		//	for(int i = 0; i < numk; i++){
		//		printf("maxlab[%d] is: %lf\n", i, maxlab[i]);
		//		printf("kseedsl[%d] is: %lf\n", i, kseedsl[i]);
		//		printf("kseedsa[%d] is: %lf\n", i, kseedsa[i]);
		//		printf("kseedsb[%d] is: %lf\n", i, kseedsb[i]);
		//		printf("kseedsx[%d] is: %lf\n", i, kseedsx[i]);
		//		printf("kseedsy[%d] is: %lf\n", i, kseedsy[i]);
		//	}
		//}
		//exit(0);

//		//#pragma omp parallel for num_threads(2)
//		for( int k = 0; k < numk; k++ )
//		{
//			//_ASSERT(clustersize[k] > 0);
//			if( clustersize[k] <= 0 ) clustersize[k] = 1;
//			inv[k] = 1.0/double(clustersize[k]);//computing inverse now to multiply, than divide later
//		}
//		
//		//#pragma omp parallel for num_threads(2)
//		for( int k = 0; k < numk; k++ )
//		{
//			kseedsl[k] = sigmal[k]*inv[k];
//			kseedsa[k] = sigmaa[k]*inv[k];
//			kseedsb[k] = sigmab[k]*inv[k];
//			kseedsx[k] = sigmax[k]*inv[k];
//			kseedsy[k] = sigmay[k]*inv[k];
//		}
		#ifdef CLEAR_DEBUG
    	tmpTime1 = Clock::now();
    	subTime0 = chrono::duration_cast<chrono::microseconds>(tmpTime1 - tmpTime0);
		time_update_kseeds += subTime0.count();
		#endif
		//if(my_rank == 1) cerr << "1 end iteration " << numitr << endl;
	
	}
	#ifdef CLEAR_DEBUG
	if(my_rank == 0){
		cout << "time of init distvec is: " << time_init_distvec << " us" << endl;
		cout << "time of convert labels64 is: " << time_cvt_labels << " us" << endl;
		cout << "time of update maxlab is: " << time_update_maxlab << " us" << endl;
		cout << "time of init sigmal cluster is: " << time_init_sigmal_cluster << " us" << endl;
		cout << "time of update sigmal cluster is: " << time_update_sigmal_cluster << " us" << endl;
		cout << "time of update kseeds is: " << time_update_kseeds << " us" << endl;
	}
	#endif
}

//===========================================================================
///	SaveSuperpixelLabels2PGM
///
///	Save labels to PGM in raster scan order.
//===========================================================================
void SLIC::SaveSuperpixelLabels2PPM(
	char*                           filename, 
	int *                           labels, 
	const int                       width, 
	const int                       height)
{
    FILE* fp;
    char header[20];
 
    fp = fopen(filename, "wb");
 
    // write the PPM header info, such as type, width, height and maximum
    fprintf(fp,"P6\n%d %d\n255\n", width, height);
	//cerr << "after fprintf " << endl;
 
    // write the RGB data
    unsigned char *rgb = new unsigned char [ (width)*(height)*3 ];
    int k = 0;
	unsigned char c = 0;
    for ( int i = 0; i < (height); i++ ) {
        for ( int j = 0; j < (width); j++ ) {
			c = (unsigned char)(labels[k]);
            rgb[i*(width)*3 + j*3 + 2] = labels[k] >> 16 & 0xff;  // r
            rgb[i*(width)*3 + j*3 + 1] = labels[k] >> 8  & 0xff;  // g
            rgb[i*(width)*3 + j*3 + 0] = labels[k]       & 0xff;  // b

			// rgb[i*(width) + j + 0] = c;
            k++;
        }
    }
    fwrite(rgb, width*height*3, 1, fp);
	//cerr << "after fwrite " << endl;

    delete [] rgb;
 
    fclose(fp);

}

//===========================================================================
///	EnforceLabelConnectivity
///
///		1. finding an adjacent label for each new component at the start
///		2. if a certain component is too small, assigning the previously found
///		    adjacent label to this component, and not incrementing the label.
//===========================================================================

//void SLIC::EnforceLabelConnectivity(
//	const int*					labels,//input labels that need to be corrected to remove stray labels
//	const int&					width,
//	const int&					height,
//	int*						nlabels,//new labels
//	int&						numlabels,//the number of labels changes in the end if segments are removed
//	const int&					K) //the number of superpixels desired by the user
//{
////	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
////	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
//
//	const int dx4[4] = {-1,  0,  1,  0};
//	const int dy4[4] = { 0, -1,  0,  1};
//
//	const int sz = width*height;
//	const int SUPSZ = sz/K;
//	//nlabels.resize(sz, -1);
//	for( int i = 0; i < sz; i++ ) nlabels[i] = -1;
//	int label(0);
//	int* xvec = new int[sz];
//	int* yvec = new int[sz];
//	int oindex(0);
//	int adjlabel(0);//adjacent label
//	for( int j = 0; j < height; j++ )
//	{
//		for( int k = 0; k < width; k++ )
//		{
//			if( 0 > nlabels[oindex] )
//			{
//				nlabels[oindex] = label;
//				//--------------------
//				// Start a new segment
//				//--------------------
//				xvec[0] = k;
//				yvec[0] = j;
//				//-------------------------------------------------------
//				// Quickly find an adjacent label for use later if needed
//				//-------------------------------------------------------
//				{for( int n = 0; n < 4; n++ )
//				{
//					int x = xvec[0] + dx4[n];
//					int y = yvec[0] + dy4[n];
//					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
//					{
//						int nindex = y*width + x;
//						if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
//					}
//				}}
//
//				int count(1);
//				for( int c = 0; c < count; c++ )
//				{
//					for( int n = 0; n < 4; n++ )
//					{
//						int x = xvec[c] + dx4[n];
//						int y = yvec[c] + dy4[n];
//
//						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
//						{
//							int nindex = y*width + x;
//
//							if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
//							{
//								xvec[count] = x;
//								yvec[count] = y;
//								nlabels[nindex] = label;
//								count++;
//							}
//						}
//
//					}
//				}
//				//-------------------------------------------------------
//				// If segment size is less then a limit, assign an
//				// adjacent label found before, and decrement label count.
//				//-------------------------------------------------------
//				if(count <= SUPSZ >> 2)
//				{
//					for( int c = 0; c < count; c++ )
//					{
//						int ind = yvec[c]*width+xvec[c];
//						nlabels[ind] = adjlabel;
//					}
//					label--;
//				}
//				label++;
//			}
//			oindex++;
//		}
//	}
//	numlabels = label;
//
//	if(xvec) delete [] xvec;
//	if(yvec) delete [] yvec;
//}

void SLIC::EnforceLabelConnectivity(
        const int *labels,//input labels that need to be corrected to remove stray labels
        const int &width,
        const int &height,
        int *nlabels,//new labels
        int &numlabels,//the number of labels changes in the end if segments are removed
        const int &K) //the number of superpixels desired by the user
{
//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    const int sz = width * height;
    const int SUPSZ = sz / K;
    //nlabels.resize(sz, -1);
#ifdef DEBUG
    double minCost0 = 0;
    double minCost1 = 0;
    double minCost2 = 0;
    double minCost3 = 0;
    double minCost4 = 0;
    auto startTime = Clock::now();
#endif
    int *tmpLables = new int[sz];
#pragma omp parallel for num_threads(threadNumber)
    for (int i = 0; i < sz; i++) nlabels[i] = -1;
#pragma omp parallel for num_threads(threadNumber)
    for (int i = 0; i < sz; i++) tmpLables[i] = -1;
//TODO vector P size
    vector<int> P[numlabels * numlabels];
    //vector<int> P[sz];

#ifdef DEBUG
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    minCost0 += compTime.count() / 1000.0;
    startTime = Clock::now();
#endif
    vector<int> G[threadNumberSmall][numlabels];
#ifdef DEBUG
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    minCost4 += compTime.count() / 1000.0;
    startTime = Clock::now();
#endif
#pragma omp parallel for num_threads(threadNumberSmall)
    for (int i = 0; i < sz; i++) {
        int tid = omp_get_thread_num();
//        int tid = 0;
        G[tid][labels[i]].push_back(i);
    }
#ifdef DEBUG
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    minCost1 += compTime.count() / 1000.0;
    startTime = Clock::now();
#endif

    int label(0);
    int adjlabel(0);//adjacent label
    int oindex(0);

    int nowTot = 0;
    int mxLable = 0;
#pragma omp parallel for num_threads(threadNumberMid)
    for (int id = 0; id < numlabels; id++) {
//        cout << "now solve " << id << endl;
        int nowLable = id;

        int siz = 0;
        for (int tid = 0; tid < threadNumberSmall; tid++)
            siz += G[tid][id].size();
//        cout << "siz " << siz << endl;
        int hasOk = 0;
        int *que = new int[siz];
        while (hasOk < siz) {
            int now = -1;
            for (int tid = 0; tid < threadNumberSmall; tid++) {
                for (int i = 0; i < G[tid][id].size(); i++) {
                    if (tmpLables[G[tid][id][i]] == -1) {
                        now = G[tid][id][i];
                        break;
                    }
                }
            }

//            printf("this round seed is %d\n", now);
            if (now == -1) {
                cout << "GG" << endl;
                break;
            }
            int head = 0, tail = 0;
            que[tail++] = now;
            tmpLables[now] = nowLable;
            oindex = now;
            while (head < tail) {
                int k = que[head++];
                int x = k % width;
                int y = k / width;
                for (int i = 0; i < 4; i++) {
                    int xx = x + dx4[i];
                    int yy = y + dy4[i];
                    if ((xx >= 0 && xx < width) && (yy >= 0 && yy < height)) {
                        int nindex = yy * width + xx;

                        if (0 > tmpLables[nindex] && labels[oindex] == labels[nindex]) {
                            que[tail++] = nindex;
                            tmpLables[nindex] = nowLable;
                        }
                    }
                }

            }
            P[nowLable].resize(tail);
            for (int i = 0; i < tail; i++)
                P[nowLable][i] = que[i];
//#pragma omp critical
//            {
//                mxLable = max(mxLable, nowLable);
//                nowTot += tail;
//            }
            hasOk += tail;
            nowLable += numlabels;
        }

        delete[]que;
    }


#ifdef DEBUG
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    minCost2 += compTime.count() / 1000.0;
    startTime = Clock::now();
#endif
    oindex = 0;
    for (int j = 0; j < height; j++) {
        for (int k = 0; k < width; k++) {
            if (0 > nlabels[oindex]) {
                nlabels[oindex] = label;
                int bel = tmpLables[oindex];
                int count2 = P[bel].size();
                if (count2 <= SUPSZ >> 2) {
                    for (int n = 3; n >= 0; n--) {
                        int x = k + dx4[n];
                        int y = j + dy4[n];
                        if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
                            int nindex = y * width + x;
                            if (nlabels[nindex] >= 0) {
                                adjlabel = nlabels[nindex];
                                break;
                            }
                        }
                    }

#pragma omp parallel for num_threads(threadNumberSmall)
                    for (int c = 0; c < count2; c++) {
                        nlabels[P[bel][c]] = adjlabel;
                    }
                    label--;
                } else {
#pragma omp parallel for num_threads(threadNumberSmall)
                    for (int c = 0; c < count2; c++) {
                        nlabels[P[bel][c]] = label;
                    }
                }
                label++;
            }
            oindex++;
        }
    }
    delete[]tmpLables;
    numlabels = label;

#ifdef DEBUG
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    minCost3 += compTime.count() / 1000.0;
    cout << "minCost0 : " << minCost0 << endl;
    cout << "minCost1 : " << minCost1 << endl;
    cout << "minCost2 : " << minCost2 << endl;
    cout << "minCost3 : " << minCost3 << endl;
    cout << "minCost4 : " << minCost4 << endl;
#endif


}


//===========================================================================
///	PerformSLICO_ForGivenK
///
/// Zero parameter SLIC algorithm for a given number K of superpixels.
//===========================================================================
void SLIC::PerformSLICO_ForGivenK(
	const unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&				m)//weight given to spatial distance
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);

	//--------------------------------------------------
	m_width  = width;
	m_height = height;
	int sz = m_width*m_height;
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for( int s = 0; s < sz; s++ ) klabels[s] = -1;
	//--------------------------------------------------
	if(1)//LAB
	{
	#ifdef DEBUG
    	auto startTime = Clock::now();
	#endif
		DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	#ifdef DEBUG
    	auto endTime = Clock::now();
    	auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    	cout <<  "DoRGBtoLABConversion time=" << compTime.count()/1000 << " ms" << endl;
		//80ms 2021/7/3
	#endif
	}
	else//RGB
	{
		m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz];
		for( int i = 0; i < sz; i++ )
		{
			m_lvec[i] = ubuff[i] >> 16 & 0xff;
			m_avec[i] = ubuff[i] >>  8 & 0xff;
			m_bvec[i] = ubuff[i]       & 0xff;
		}
	}
	//--------------------------------------------------

	bool perturbseeds(true);
	vector<double> edgemag(0);
	#ifdef DEBUG
    auto startTime = Clock::now();
	#endif
	if(perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	#ifdef DEBUG
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
   	cout <<  "DetectLabEdges time=" << compTime.count()/1000 << " ms" << endl;
	//13ms 2021/7/3
	#endif
	
	#ifdef DEBUG
    startTime = Clock::now();
	#endif
	GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, K, perturbseeds, edgemag);
	#ifdef DEBUG
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
   	cout <<  "GetLABXYSeeds_ForGivenK time=" << compTime.count()/1000 << " ms" << endl;
	//0ms 2021/7/3
	#endif

	int STEP = sqrt(double(sz)/double(K)) + 2.0;//adding a small value in the even the STEP size is too small.
	
	#ifdef DEBUG
    startTime = Clock::now();
	#endif
	int numk = kseedsl.size();
	double kseedsls[numk];
	double kseedsas[numk];
	double kseedsbs[numk];
	double kseedsxs[numk];
	double kseedsys[numk];
	for(int i = 0; i < numk; i++){
		kseedsls[i] = kseedsl[i];
		kseedsas[i] = kseedsa[i];
		kseedsbs[i] = kseedsb[i];
		kseedsxs[i] = kseedsx[i];
		kseedsys[i] = kseedsy[i];
	}
	//PerformSuperpixelSegmentation_VariableSandM(kseedsl,kseedsa,kseedsb,kseedsx,kseedsy,klabels,STEP,10);
	PerformSuperpixelSegmentation_VariableSandM(kseedsls,kseedsas,kseedsbs,kseedsxs,kseedsys,klabels, numk, STEP,10);
	#ifdef DEBUG
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
   	cout <<  "PerformSuperpixelSegmentation_VariableSandM time=" << compTime.count()/1000 << " ms" << endl;
	//1083ms 2021/7/3 former
	//505ms 2021/7/3 later
	#endif

	numlabels = kseedsl.size();

	#ifdef DEBUG
    startTime = Clock::now();
	#endif
	if(my_rank == 0){
		int* nlabels = new int[sz];
		EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, K);

		#pragma omp parallel for num_threads(numThreads)
		for(int i = 0; i < sz; i++ ) 
			klabels[i] = nlabels[i];
		if(nlabels) delete [] nlabels;
	}
	#ifdef DEBUG
    endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
   	cout <<  "EnforceLabelConnectivity time=" << compTime.count()/1000 << " ms" << endl;
	//147ms 2021/7/3
	#endif
}

//===========================================================================
/// Load PPM file
///
/// 
//===========================================================================
void LoadPPM(char* filename, unsigned int** data, int* width, int* height)
{
    char header[1024];
    FILE* fp = NULL;
    int line = 0;
 
    fp = fopen(filename, "rb");
 
    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2) {    
        fgets(header, 1024, fp);
        if (header[0] != '#') {
            ++line;
        }
    }
    // read width and height
    sscanf(header,"%d %d\n", width, height);
 
    // read the maximum of pixels
    fgets(header, 20, fp);
 
    // get rgb data
    unsigned char *rgb = new unsigned char [ (*width)*(*height)*3 ];
    fread(rgb, (*width)*(*height)*3, 1, fp);

    *data = new unsigned int [ (*width)*(*height)*4 ];
    int k = 0;
    for ( int i = 0; i < (*height); i++ ) {
        for ( int j = 0; j < (*width); j++ ) {
            unsigned char *p = rgb + i*(*width)*3 + j*3;
                                      // a ( skipped )
            (*data)[k]  = p[2] << 16; // r
            (*data)[k] |= p[1] << 8;  // g
            (*data)[k] |= p[0];       // b
            k++;
        }
    }

    // ofc, later, you'll have to cleanup
    delete [] rgb;
 
    fclose(fp);
}

//===========================================================================
/// Load PPM file
///
/// 
//===========================================================================
int CheckLabelswithPPM(char* filename, int* labels, int width, int height)
{
    char header[1024];
    FILE* fp = NULL;
    int line = 0, ground = 0;
 
    fp = fopen(filename, "rb");
 
    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2) {    
        fgets(header, 1024, fp);
        if (header[0] != '#') {
            ++line;
        }
    }
    // read width and height
	int w(0);
	int h(0);
    sscanf(header,"%d %d\n", &w, &h);
	if (w != width || h != height) return -1;
 
    // read the maximum of pixels
    fgets(header, 20, fp);
 
    // get rgb data
    unsigned char *rgb = new unsigned char [ (w)*(h)*3 ];
    fread(rgb, (w)*(h)*3, 1, fp);

    int num = 0, k = 0;
    for ( int i = 0; i < (h); i++ ) {
        for ( int j = 0; j < (w); j++ ) {
            unsigned char *p = rgb + i*(w)*3 + j*3;
                                  // a ( skipped )
            ground  = p[2] << 16; // r
            ground |= p[1] << 8;  // g
            ground |= p[0];       // b
            
			if (ground != labels[k])
				num++;

			k++;
        }
    }

    // ofc, later, you'll have to cleanup
    delete [] rgb;
 
    fclose(fp);

	return num;
}

//===========================================================================
///	The main function
///
//===========================================================================
int main (int argc, char **argv)
{
	int proc_len;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Get_processor_name(processor_name, &proc_len);
	printf("proces %d of %d, process name is: %s\n", my_rank, num_procs, processor_name);

	unsigned int* img = NULL;
	int width(0);
	int height(0);
	
	string fileIndex = "1";
	int m_spcount;
	if(argc >= 2){
		//fileIndex = stoi(argv[1]);
		fileIndex = argv[1];
	}
	if(fileIndex == "1"){
		LoadPPM((char *)"input_image1.ppm", &img, &width, &height);
		m_spcount = 200;
	}
	else if(fileIndex == "2"){
		LoadPPM((char *)"input_image2.ppm", &img, &width, &height);
		m_spcount = 400;
	}
	else if(fileIndex == "3"){
		LoadPPM((char *)"input_image3.ppm", &img, &width, &height);
		m_spcount = 150;
	}
	else{
		cerr << "ivalid input file index, exit()" << endl;
		return 1;
	}
	

	if (width == 0 || height == 0) return -1;

	int sz = width*height;
	int* labels = new int[sz];
	int numlabels(0);
	SLIC slic;
	double m_compactness;
	//m_spcount = 200;
	m_compactness = 10.0;
    auto startTime = Clock::now();
	slic.PerformSLICO_ForGivenK(img, width, height, labels, numlabels, m_spcount, m_compactness);//for a given number K of superpixels
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout <<  "process " << my_rank << ": Computing time=" << compTime.count()/1000 << " ms" << endl;
	//if(my_rank == 1) return 0;

	int num = 10000;
	if(my_rank == 0){
		if(fileIndex == "1"){
			num = CheckLabelswithPPM((char *)"check1.ppm", labels, width, height);
		}
		else if(fileIndex == "2"){
			num = CheckLabelswithPPM((char *)"check2.ppm", labels, width, height);
		}
		else if(fileIndex == "3"){
			num = CheckLabelswithPPM((char *)"check3.ppm", labels, width, height);
		}
		if (num < 0) {
			cout <<  "The result for labels is different from output_labels.ppm." << endl;
		} else {
			cout <<  "There are " << num << " points' labels are different from original file." << endl;
		}
		
		slic.SaveSuperpixelLabels2PPM((char *)"output_labels.ppm", labels, width, height);
	}//end my_rank == 0
	//cerr << "end the main " << endl;
	if(labels) delete [] labels;
	
	if(img) delete [] img;

	return 0;
}
