#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif
int isInsideMatrix(int2 p1,int2 dim){
	if(p1.x>=dim.x)
		return (int)0;
	if(p1.y>=dim.y)
		return (int)0;
	if(p1.x<0)
		return (int)0;
	if(p1.y<0)
		return (int)0;
	return (int)1;		
}


int getIndex(int2 pts, int2 dim){
	return (int)pts.x+dim.x*pts.y;
}

/*
=====================================
updateMP  - measures the euclidian and polar gradient

=====================================
Obs: the matricesphases, mods, gx, and gy must have the same dimensions of mat (the input).
     dim1 > 3, dim2>3

=====================================
	mat -> matrix with Amplitudes
	phases -> this kernel updates the phases matrix
	mods -> this kernel updates the absolute matrix	
	gx -> euclidian gradient x
	gy -> euclidian gradient y
	boundary -> type of boundary on measuring gradient (1 = reflexive, otherwise = periodic(default) ) 
	dim1,dim2 -> matrix dimensions
*/

__kernel void updateMP(__global const double *mat, __global double *phases, __global double *mods,__global double *gx, __global double *gy, const int boundary){
	int2 p1 = (int2)(get_global_id(0), get_global_id(1));
        int2 dim = (int2)(get_global_size(0),get_global_size(1));
        __constant int tr = 1;
	int p = getIndex(p1,dim);
 
	///Measuring Dx	
	if( (isInsideMatrix(p1 +(int2)(1,0),dim) == tr) && (isInsideMatrix(p1 +(int2)(-1,0),dim) == tr) ){
		gx[p] = (mat[getIndex(p1 +(int2)(1,0),dim)]-mat[getIndex(p1 +(int2)(-1,0),dim)])/2.0;	
	} else if(boundary == 1){//reflexive boundary -- obs: assumes that we have at least 3x3 matrices
		if(isInsideMatrix(p1 +(int2)(1,0),dim) == tr)
			gx[p] = mat[getIndex(p1 +(int2)(1,0),dim)]-mat[p];
		else
			gx[p] = mat[p]-mat[getIndex(p1+(int2)(-1,0),dim)];
	} else {//periodic boundary -- obs: assumes that we have at least 3x3 matrices
		if(isInsideMatrix(p1 +(int2)(1,0),dim) == tr)
			gx[p] = (mat[getIndex(p1 +(int2)(1,0),dim)]-mat[getIndex((int2)(0,p1.y),dim)])/2.0;
		else
			gx[p] = (mat[getIndex((int2)(dim.x-1,p1.y),dim)]-mat[getIndex(p1+(int2)(-1,0),dim)])/2.0;
	}

	///Measuring Dy	
	if( isInsideMatrix(p1 +(int2)(0,1),dim) == tr && isInsideMatrix(p1 +(int2)(0,-1),dim) == tr ){
		gy[p] = (mat[getIndex(p1 +(int2)(0,1),dim)]-mat[getIndex(p1 +(int2)(0,-1),dim)])/2.0;	
	} else if(boundary == 1){//reflexive boundary -- obs: assumes that we have at least 3x3 matrices
		if(isInsideMatrix(p1 +(int2)(0,1),dim) == tr)
			gy[p] = mat[getIndex(p1 +(int2)(0,1),dim)]-mat[p];
		else
			gy[p] = mat[p]-mat[getIndex(p1+(int2)(0,-1),dim)];
	} else {//periodic boundary -- obs: assumes that we have at least 3x3 matrices
		if(isInsideMatrix(p1 +(int2)(0,1),dim) == tr)
			gy[p] = (mat[getIndex(p1+(int2)(0,1),dim)]-mat[getIndex((int2)(p1.x,dim.y-1),dim)])/2.0;
		else
			gy[p] = (mat[getIndex((int2)(p1.x,0),dim)]-mat[getIndex(p1+(int2)(0,-1),dim)])/2.0;
	}

	//Polar coordinate section
	mods[p] = sqrt(gx[p]*gx[p]+gy[p]*gy[p]);
	phases[p] = atan2(gy[p],gx[p])*(180.0/M_PI);
	if(isnan(phases[p]) || isinf(phases[p])){
		phases[p] = -90.0;
	}
}
