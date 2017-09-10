
float angleDifference(float a1,float a2){
	return min(fabs(a1-a2),fabs(fabs(a1-a2)-(2.0f*M_PI_F)));
}

int isSymmetrycAngle(float a1,float a2, float tol){
	if(fabs(angleDifference(a1,a2)-M_PI_F)<= tol){
		return (int)1;
	}
	return (int)0;
}

int isSymmetrycMod(float m1,float m2, float tol){
	if(fabs(m1-m2)<= tol){
		return (int)1;
	}
	return (int)0;
}
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
	return (int)pts[0]+dim[0]*pts[1];
}

__kernel void asymmetry_r(__global float *phases, __global float *mods, __global int *tableau, const float mtol, const float ftol){
	int2 p1 = (int2)(get_global_id(0), get_global_id(1));
        int2 dim = (int2)(get_global_size(0), get_global_size(1));
        float2 center = (float2)(convert_float(dim.x-1)/2.0f,convert_float(dim.y-1)/2.0f);
        float tempR,incr;

        __constant int tr = 1;
        float myDist = distance(center,convert_float2(p1));
	int2 p2 = (int2)(0);
	int2 p3 = (int2)(0);
        float x;
		
	tableau[getIndex(p1,dim)] = 0;

	if(mods[getIndex(p1,dim)]<mtol){
		tableau [getIndex(p1,dim)] = 1;
		return;
	}

	incr = M_PI_F/max(dim.x,dim.y);
        // minimal --and lazy-- search
	for(x =incr; x < 2.0f*M_PI_F; x=x+incr){
		p3 = convert_int2(center + (float2)(myDist*cos(x),myDist*sin(x)));	
		if((isInsideMatrix(p3,dim) == tr) && (fabs(distance(convert_float2(p3),center)-myDist)<1.0f)){
			if((isSymmetrycMod(mods[getIndex(p3,dim)], mods[getIndex(p1,dim)], mtol) == tr) &&
		   	   (isSymmetrycAngle(phases[getIndex(p3,dim)], phases[getIndex(p1,dim)], ftol) == tr)){
				tableau [getIndex(p1,dim)] = 1;//seems like we found a symmetric vector 
				return;
			}
		}
		p3 = convert_int2(center + (float2)(myDist*cos(x),myDist*sin(x)))+(int2)(1,0);	
		if((isInsideMatrix(p3,dim) == tr) && (fabs(distance(convert_float2(p3),center)-myDist)<1.0f)){
			if((isSymmetrycMod(mods[getIndex(p3,dim)], mods[getIndex(p1,dim)], mtol) == tr) &&
		   	   (isSymmetrycAngle(phases[getIndex(p3,dim)], phases[getIndex(p1,dim)], ftol) == tr)){
				tableau [getIndex(p1,dim)] = 1;//seems like we found a symmetric vector 
				return;
			}
		}
		p3 = convert_int2(center + (float2)(myDist*cos(x),myDist*sin(x)))+(int2)(0,1);	
		if((isInsideMatrix(p3,dim) == tr) && (fabs(distance(convert_float2(p3),center)-myDist)<1.0f)){
			if((isSymmetrycMod(mods[getIndex(p3,dim)], mods[getIndex(p1,dim)], mtol) == tr) &&
		   	   (isSymmetrycAngle(phases[getIndex(p3,dim)], phases[getIndex(p1,dim)], ftol) == tr)){
				tableau [getIndex(p1,dim)] = 1;//seems like we found a symmetric vector 
				return;
			}
		}
		p3 = convert_int2(center + (float2)(myDist*cos(x),myDist*sin(x)))+(int2)(-1,0);	
		if((isInsideMatrix(p3,dim) == tr) && (fabs(distance(convert_float2(p3),center)-myDist)<1.0f)){
			if((isSymmetrycMod(mods[getIndex(p3,dim)], mods[getIndex(p1,dim)], mtol) == tr) &&
		   	   (isSymmetrycAngle(phases[getIndex(p3,dim)], phases[getIndex(p1,dim)], ftol) == tr)){
				tableau [getIndex(p1,dim)] = 1;//seems like we found a symmetric vector 
				return;
			}
		}
		p3 = convert_int2(center + (float2)(myDist*cos(x),myDist*sin(x)))+(int2)(0,-1);	
		if((isInsideMatrix(p3,dim) == tr) && (fabs(distance(convert_float2(p3),center)-myDist)<1.0f)){
			if((isSymmetrycMod(mods[getIndex(p3,dim)], mods[getIndex(p1,dim)], mtol) == tr) &&
		   	   (isSymmetrycAngle(phases[getIndex(p3,dim)], phases[getIndex(p1,dim)], ftol) == tr)){
				tableau [getIndex(p1,dim)] = 1;//seems like we found a symmetric vector 
				return;
			}
		}
	}
}
