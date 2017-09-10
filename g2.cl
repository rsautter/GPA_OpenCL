/*
=====================================
getG2 -> return g2 fromthe assymetrical vector set 
=====================================
	phases,mods -> polar coordinate vector
	tableau -> matrix that shows whether a vector is symmetrical or (0 = symmetrical, 1=asymmetrical)
	partialX, partialY,partialMS, countAsym -> temporary vector(same dim of phasesand mods)
	g2 -> the output value (a vector with only one element)
	nelements -> rows*cols

=====================================

*/

__kernel void getG2(__global float *gx, __global float * gy, __global float *mods,__global int *tableau,
		    __global float *partialX, __global float *partialY,__global float *partialMS, __global int* countAsym,
		    __global float* g2,
		    const int nelements

){
	int id = get_global_id(0);
	float div = 0.0f;
	//every element verify if it is a symmetric vector
	if(tableau[id] == 1){
		partialX[id] = 0.0f;
		partialY[id] = 0.0f;
		partialMS[id] = 0.0f;
		countAsym[id] = 0;
	}else{
		partialX[id] = gx[id];
		partialY[id] = gy[id];
		partialMS[id] = mods[id];
		countAsym[id] = 1;
	}
	// measuring diversity,using partial sum method
	for (int stride = nelements/2; stride>0; stride=stride/2){
		barrier(CLK_GLOBAL_MEM_FENCE); //wait everyone update
		if (id < stride){
        		partialX[id] += partialX[id + stride];
			partialY[id] += partialY[id + stride];
			partialMS[id] += partialMS[id + stride];
			countAsym[id] += countAsym[id+stride];
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE); //wait everyone update
	if(nelements%2 != 0 && id == 0){
		partialX[id] += partialX[nelements-1];
		partialY[id] += partialY[nelements-1];
		partialMS[id] += partialMS[nelements-1];
		countAsym[id] += countAsym[nelements-1];
	}
	if(id == 0){
		if(countAsym[id]<3)
			g2[id] = 0.0f;
		else {
			div = sqrt(pow(partialX[id],2.0f)+pow(partialY[id],2.0f)) / partialMS[id];
			g2[id] = (convert_float(countAsym[id])/convert_float(nelements)) * (2.0f-div);
		}
	}
}

