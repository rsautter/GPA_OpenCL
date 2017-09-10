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

__kernel void updateMP(__global const float *mat, __global float *phases, __global float *mods,__global float *gx, __global float *gy, const int boundary){
	int2 p1 = (int2)(get_global_id(0), get_global_id(1));
        int2 dim = (int2)(get_global_size(0),get_global_size(1));
        __constant int tr = 1;
 
	///Measuring Dx	
	if( (isInsideMatrix(p1 +(int2)(1,0),dim) == tr) && (isInsideMatrix(p1 +(int2)(-1,0),dim) == tr) ){
		gx[getIndex(p1,dim)] = (mat[getIndex(p1 +(int2)(1,0),dim)]-mat[getIndex(p1 +(int2)(-1,0),dim)])/2.0f;	
	} else if(boundary == 1){//reflexive boundary -- obs: assumes that we have at least 3x3 matrices
		if(isInsideMatrix(p1 +(int2)(1,0),dim) == tr)
			gx[getIndex(p1,dim)] = mat[getIndex(p1 +(int2)(1,0),dim)]-mat[getIndex(p1,dim)];
		else
			gx[getIndex(p1,dim)] = mat[getIndex(p1,dim)]-mat[getIndex(p1+(int2)(-1,0),dim)];
	} else {//periodic boundary -- obs: assumes that we have at least 3x3 matrices
		if(isInsideMatrix(p1 +(int2)(1,0),dim) == tr)
			gx[getIndex(p1,dim)] = (mat[getIndex(p1 +(int2)(1,0),dim)]-mat[getIndex((int2)(0,p1.y),dim)])/2.0f;
		else
			gx[getIndex(p1,dim)] = (mat[getIndex((int2)(dim.x-1,p1.y),dim)]-mat[getIndex(p1+(int2)(-1,0),dim)])/2.0f;
	}

	///Measuring Dy	
	if( isInsideMatrix(p1 +(int2)(0,1),dim) == tr && isInsideMatrix(p1 +(int2)(0,-1),dim) == tr ){
		gy[getIndex(p1,dim)] = (mat[getIndex(p1 +(int2)(0,1),dim)]-mat[getIndex(p1 +(int2)(0,-1),dim)])/2.0f;	
	} else if(boundary == 1){//reflexive boundary -- obs: assumes that we have at least 3x3 matrices
		if(isInsideMatrix(p1 +(int2)(0,1),dim) == tr)
			gy[getIndex(p1,dim)] = mat[getIndex(p1 +(int2)(0,1),dim)]-mat[getIndex(p1,dim)];
		else
			gy[getIndex(p1,dim)] = mat[getIndex(p1,dim)]-mat[getIndex(p1+(int2)(0,-1),dim)];
	} else {//periodic boundary -- obs: assumes that we have at least 3x3 matrices
		if(isInsideMatrix(p1 +(int2)(0,1),dim) == tr)
			gy[getIndex(p1,dim)] = (mat[getIndex(p1+(int2)(0,1),dim)]-mat[getIndex((int2)(p1.x,dim.y-1),dim)])/2.0f;
		else
			gy[getIndex(p1,dim)] = (mat[getIndex((int2)(p1.x,0),dim)]-mat[getIndex(p1+(int2)(0,-1),dim)])/2.0f;
	}

	//Polar coordinate section
	mods[getIndex(p1,dim)] = sqrt(pow(gx[getIndex(p1,dim)],2.0f)+pow(gy[getIndex(p1,dim)],2.0f));
	phases[getIndex(p1,dim)] = atan2(gy[getIndex(p1,dim)],gx[getIndex(p1,dim)]);
	if(isnan(phases[getIndex(p1,dim)]) || isinf(phases[getIndex(p1,dim)])){
		phases[getIndex(p1,dim)] = -M_PI_F/2.0f;
	}
}
