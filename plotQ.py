import numpy as np
from matplotlib.mlab import bivariate_normal
import matplotlib.pyplot as plt

def distance(start, end):
	sx, sy = start
	ex, ey = end
	return ((sx-ex)**2.0+(sy-ey)**2.0)**0.5
def gradient(mat):
        w, h = len(mat[0]),len(mat)
        dx = np.array([[0.0 for i in range(w) ] for j in range(h)])
        dy = np.array([[0.0 for i in range(w) ] for j in range(h)])
        for i in range(w):
           for j in range(h):
                 divy =  2.0 if (j<len(mat)-1 and j>0) else 1.0
                 divx =  2.0 if (i<len(mat[j])-1 and i>0) else 1.0
                 i1 = (i+1) if i<len(mat[j])-1 else i
                 j1 = (j+1) if j<len(mat)-1 else j
                 i2 = (i-1) if i>0 else i
                 j2 = (j-1) if j>0 else j
              	 dy[j, i] = (mat[j2, i] - mat[j1, i])/divy
                 dx[j, i] = (mat[j, i1] - mat[j, i2] )/divx
        return dy,dx

nt = 9

X,Y = np.meshgrid( [n-(nt-1)/2 for n in range(nt)], [n-(nt-1)/2 for n in range(nt)])
gaussian = bivariate_normal(X, Y,sigmax=2.0,sigmay=2.0)
#plt.imshow(gaussian)
gaussian[6][7]+= 0.02
dy,dx = gradient(gaussian)
fig = plt.figure(figsize=(8,8))

plt.quiver(dx,-dy)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
print(gaussian)
plt.savefig("exemploGaussian.png")

