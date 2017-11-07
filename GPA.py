import numpy
from math import radians
from scipy.spatial import Delaunay as Delanuay
import pyopencl as cl
import time
import matplotlib.pyplot as plt

class GPA:

    def __init__(self, mat,device=None):
        # setting matrix
        self.mat = numpy.array(mat,dtype=numpy.float64)
        self.boundaryType = "reflexive"
        self.profile = True
   
        # percentual Ga proprieties
        self.cols = len(self.mat[0])
        self.rows = len(self.mat)
        self.dim = mat.shape
        self.triangulation_points = []
        self.phaseDiversity = 0.0

        self.mods = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float64)
        self.phases = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float64)
        self.gradient_dx = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float64)
        self.gradient_dy = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float64)
        self.gradient_asymmetric_dx = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float64)
        self.gradient_asymmetric_dy = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float64)
        self.tableau = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.int32) 

        #Opencl constants
	if(device is None):
		self.device = self.getDevice()
	else:
        	self.device = device
        self.ctx = cl.Context(devices=[self.device])
        if(self.profile):
            self.queue = cl.CommandQueue(self.ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
    
    def getDevice(self):
	temp = 0
	valid = False
	devices = cl.get_platforms()[0].get_devices()
	msg = "Choose a  device:\n"
	for i in range(len(devices)):
		msg += str(i)+" - "+str(devices[i])+"\n"
	if(len(devices)<1):
		raise Exception("No device found!")
	while(not valid):
		try:
			temp = int(raw_input(msg))
			if temp<len(devices) and temp>-1:
				
				valid = True
		except:
			valid = False
	return devices[temp]

    def readKernel(self,kernelFile):
        with open(kernelFile, 'r') as f:
            data=f.read()
        return data

    def _update_asymmetric_tableau(self,mtol,ftol):
        k = self.readKernel("asymmetry_conc.cl")
        prg = cl.Program(self.ctx,k).build()
        self.run2 = prg.asymmetry_r(self.queue, self.dim, None,self.ph,self.md,self.tb,numpy.float64(mtol),numpy.float64(ftol))
        self.run2.wait()

    def _update_Mods_Phases(self):
        btype = 0
        if(self.boundaryType == "reflexive"):
            btype = 1
        # local buffers
        tm = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.mat)
        self.gx = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.gradient_dx)       
        self.gy = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.gradient_dy)

        k = self.readKernel("updateMP.cl")
        prg = cl.Program(self.ctx,k).build()
        self.run1 = prg.updateMP(self.queue,self.dim,None,tm,self.ph,self.md,self.gx,self.gy,numpy.int32(btype))
	self.run1.wait()


    def _get_G1(self):
        for i in range(self.rows):
            for j in range(self.cols):
    		if(self.tableau[j,i] == 0):    
            		self.triangulation_points.append([j+0.5*self.gradient_asymmetric_dx[i, j], i+0.5*self.gradient_asymmetric_dy[i, j]])
        self.triangulation_points = numpy.array(self.triangulation_points)
        self.n_points = len(self.triangulation_points)
	numpy.savetxt("teste.txt", self.tableau, fmt="%i")
        if self.n_points <= 3:
            self.n_edges = 0
            self.G1 = 0.0
        else:
            self.triangles = Delanuay(self.triangulation_points)
            neigh = self.triangles.vertex_neighbor_vertices
            self.n_edges = len(neigh[1])/2
            self.G1 = float(self.n_edges-self.n_points)/float(self.n_points)
        return self.G1

    def _get_G2(self):
	gsize = int(self.device.max_work_group_size)
	l = self.dim[0]*self.dim[1]
	bsize  = max(1,2*l/gsize)
	partialX = numpy.zeros(bsize).astype(numpy.float64)
	partialY = numpy.zeros(bsize).astype(numpy.float64)
	partialMS = numpy.zeros(bsize).astype(numpy.float64)
	countAsymm = numpy.zeros(bsize).astype(numpy.int32)

        tpartialX = cl.LocalMemory(8*gsize)
        tpartialY = cl.LocalMemory(8*gsize)
        tpartialMS = cl.LocalMemory(8*gsize)
        tcountAsymm = cl.LocalMemory(8*gsize)  

        px = cl.Buffer(self.ctx, self.mf.WRITE_ONLY | self.mf.COPY_HOST_PTR, hostbuf=partialX)
        py = cl.Buffer(self.ctx, self.mf.WRITE_ONLY | self.mf.COPY_HOST_PTR, hostbuf=partialY)
        pms = cl.Buffer(self.ctx, self.mf.WRITE_ONLY | self.mf.COPY_HOST_PTR, hostbuf=partialMS)
        ca = cl.Buffer(self.ctx, self.mf.WRITE_ONLY | self.mf.COPY_HOST_PTR, hostbuf=countAsymm)

	loc_buf = cl.LocalMemory(8*gsize)

        k = self.readKernel("g2.cl")
        prg = cl.Program(self.ctx,k).build()
        self.run3 = prg.getG2(self.queue,(l,),None,self.gx,self.gy,self.md,self.tb,\
						   px,py,pms,ca,\
						   tpartialX,tpartialY,tpartialMS,tcountAsymm)
        cl.enqueue_copy(self.queue, partialX, px).wait()
	cl.enqueue_copy(self.queue, partialY, py).wait()
	cl.enqueue_copy(self.queue, partialMS, pms).wait()
        cl.enqueue_copy(self.queue, countAsymm, ca).wait()

	self.div = ((sum(partialX.astype(numpy.float32))**2.0+sum(partialY.astype(numpy.float32))**2.0)**0.5)/sum(partialMS.astype(numpy.float32)) if sum(partialMS.astype(numpy.float32))>0.001 else 0.0
	self.prop = float(sum(countAsymm))/float(l)
        self.G2 = self.prop*(2.0-self.div)
        return self.G2

    def _get_variables_GPU(self):
       cl.enqueue_copy(self.queue, self.gradient_dx, self.gx).wait()
       cl.enqueue_copy(self.queue, self.gradient_dy,self.gy).wait()
       cl.enqueue_copy(self.queue, self.phases, self.ph).wait()
       cl.enqueue_copy(self.queue, self.mods, self.md).wait()
       cl.enqueue_copy(self.queue, self.tableau, self.tb).wait() 
       #plt.figure(0)
       #plt.imshow(self.tableau)
       #plt.colorbar()
       #plt.figure(1)
       #plt.imshow(self.phases)
       #plt.colorbar()
       #plt.figure(2)
       #plt.imshow((self.mods))
       #plt.colorbar()
       #plt.show()
       

    # This function estimates both asymmetric gradient coeficient (geometric and algebric), with the given tolerances
    def evaluate(self,mtol, ftol,moment=["G2"]):
        #Buffers on GPU
        self.ph = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.phases)
        self.md = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.mods)       
        self.tb = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.tableau)
 
        self._update_Mods_Phases()
        self._update_asymmetric_tableau(mtol, ftol)
	if("G2" in moment):
        	self._get_G2()
		self._get_variables_GPU()
		self.t3 = 1e-9*(self.run3.profile.end-self.run3.profile.start)
	if("G1" in moment):
		self._get_variables_GPU()
		self.t3 = time.time()
		self._get_G1()
		self.t3 = time.time() - self.t3
	self.t1 = 1e-9*(self.run1.profile.end-self.run1.profile.start)
	self.t2 = 1e-9*(self.run2.profile.end-self.run2.profile.start)
	
        if(self.profile):
            print("")
            print("Gradient Time  "+str(self.t1)+" (s)")
            print("Tableau Time   "+str(self.t2)+" (s)")
            print("Moment Time        "+str(self.t3)+" (s)")
            print("")

                 
                 
 
               


