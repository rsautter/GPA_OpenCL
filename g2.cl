#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif
/*
=====================================
getG2 -> return g2 from the assymetrical vector set 
=====================================
	phases,mods -> polar coordinate vector
	tableau -> matrix that shows whether a vector is symmetrical or (0 = symmetrical, 1=asymmetrical)
	partialX, partialY,partialMS, countAsym -> temporary vector(same dim of phasesand mods)
	g2 -> the output value (a vector with only one element)
	nelements -> rows*cols

=====================================

*/

__kernel void getG2(__global double *gx, __global double * gy, __global double *mods,__global int *tableau,
		    __global double *partialX, __global double *partialY,__global double *partialMS, __global int* countAsym,
		    __local double *tpartialX, __local double *tpartialY,__local double *tpartialMS, __local int* tcountAsym
){
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int gsize = get_global_size(0);
	int lsize = get_local_size(0);

	//every element verify if it is a symmetric vector
	if(tableau[gid] > 0){
		tpartialX[lid] = 0.0;
		tpartialY[lid] = 0.0;
		tpartialMS[lid] = 0.0;
		tcountAsym[lid] = 0;
	}else{
		tpartialX[lid] = gx[gid];
		tpartialY[lid] = gy[gid];
		tpartialMS[lid] = mods[gid];
		tcountAsym[lid] = 1;
	}

	int oldStride = lsize;	
	// measuring diversity,using partial sum method
	for (int stride = lsize/2; stride>0; stride = stride/2){
		barrier(CLK_LOCAL_MEM_FENCE); //wait everyone update
		if (lid < stride){
        		tpartialX[lid] += tpartialX[lid+stride];
			tpartialY[lid] += tpartialY[lid+stride];
			tpartialMS[lid] += tpartialMS[lid+stride];
			tcountAsym[lid] += tcountAsym[lid+stride];
		}
		if(oldStride%2 != 0 && lid == stride -1){
			tpartialX[lid] += tpartialX[lid+stride+1];
			tpartialY[lid] += tpartialY[lid+stride+1];
			tpartialMS[lid] += tpartialMS[lid+stride+1];
			tcountAsym[lid] += tcountAsym[lid+stride+1];
		}
		oldStride = stride;
	}
	barrier(CLK_LOCAL_MEM_FENCE); //wait everyone update
	//merge each local sum in an array
	if(lid == 0){
		partialX[get_group_id(0)] = tpartialX[lid];
		partialY[get_group_id(0)] = tpartialY[lid];
		partialMS[get_group_id(0)] = tpartialMS[lid];
		countAsym[get_group_id(0)] = tcountAsym[lid];
	}
	
}

