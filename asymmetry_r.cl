#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

double angleDifference(double a1,double a2){
	return min(fabs(a1-a2),fabs(fabs(a1-a2)-360.0));
}

int isSymmetrycAngle(double a1,double a2, double tol){
	if(fabs(angleDifference(a1,a2)-180.0)<= tol){
		return (int)1;
	}
	return (int)0;
}

int isSymmetrycMod(double m1,double m2, double tol){
	if(fabs(m1-m2)/max(fabs(m1),fabs(m2))<= fabs(tol)){
		return (int)1;
	}
	return (int)0;
}
double euclid_distance(double2 p1, double2 p2){
	return pow(pow(p1.x-p2.x,2.0)+pow(p1.y-p2.y,2.0),0.5);
}

int isInsideMatrix(int2 p1,int2 dim){
	if(p1.x>=dim.x-2)
		return (int)0;
	if(p1.y>=dim.y-2)
		return (int)0;
	if(p1.x<2)
		return (int)0;
	if(p1.y<2)
		return (int)0;
	return (int)1;		
}

int getIndex(int2 pts, int2 dim){
	return (int)pts[0]+dim[0]*pts[1];
}

__kernel void asymmetry_r(__global double *phases, __global double *mods, __global int *tableau, const double mtol, const double ftol){
	int2 p1 = (int2)(get_global_id(0), get_global_id(1));
        int2 dim = (int2)(get_global_size(0), get_global_size(1));
        double2 center = (double2)(convert_double(dim.x-1)/2.0,convert_double(dim.y-1)/2.0);
        double tempR,incr;

        __constant int tr = 1;
        double myDist = euclid_distance(center,convert_double2(p1));
	int2 p3 = (int2)(0);
        int x, y;
		
	tableau[getIndex(p1,dim)] = 0;

	incr = M_PI/(8.0*max(dim.x,dim.y));
	if( mods[getIndex(p1,dim)] < mtol/2){
		tableau[getIndex(p1,dim)] = 1;
		return;
	}
	if(isInsideMatrix(p1,dim)!=tr){
		tableau[getIndex(p1,dim)] = 1;
		return;
	}
	
        //  lazy search
	for(x=0; x < dim.x; x++){
		for(y=0; y < dim.y; y++){
			p3 = (int2)(x,y);
			if((fabs(euclid_distance(convert_double2(p3),center)-myDist)<2.0)){
				if((isSymmetrycMod(mods[getIndex(p3,dim)], mods[getIndex(p1,dim)], mtol) == tr) &&
		   		   (isSymmetrycAngle(phases[getIndex(p3,dim)], phases[getIndex(p1,dim)], ftol) == tr)){
					tableau [getIndex(p1,dim)] = 1;//seems like we found a symmetric vector 
					return;
				}
			}
		}
	}
}
