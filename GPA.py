import numpy
from math import radians
from scipy.spatial import Delaunay as Delanuay
import pyopencl as cl

class GPA:

    def __init__(self, mat):
        # setting matrix,and calculating the gradient field
        self.mat = numpy.array(mat,dtype=numpy.float32)
        self.boundaryType = "reflexive"
   
        # percentual Ga proprieties
        self.cols = len(self.mat[0])
        self.rows = len(self.mat)
        self.dim = mat.shape
        self.triangulation_points = []
        self.phaseDiversity = 0.0

        self.mods = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float32)
        self.phases = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float32)
        self.gradient_dx = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float32)
        self.gradient_dy = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float32)
        self.gradient_asymmetric_dx = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float32)
        self.gradient_asymmetric_dy = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.float32)
        self.tableau = numpy.zeros(shape=(self.mat.shape[0],self.mat.shape[1]),dtype=numpy.int32) 

        #Opencl constants
        self.device = cl.get_platforms()[0].get_devices(cl.device_type.GPU)[0]#uses the first GPU  listed, if not found raises an error
        #self.ctx = cl.create_some_context() #ask which context to use 
        self.ctx = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags

    def readKernel(self,kernelFile):
        with open(kernelFile, 'r') as f:
            data=f.read()
        return data

    def _update_asymmetric_tableau(self,mtol,ftol):
        k = self.readKernel("asymmetry_r.cl")
        prg = cl.Program(self.ctx,k).build()
        prg.asymmetry_r(self.queue, self.dim, None,self.ph,self.md,self.tb,numpy.float32(mtol),numpy.float32(ftol))

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
        prg.updateMP(self.queue,self.dim,None,tm,self.ph,self.md,self.gx,self.gy,numpy.int32(btype))

    def _get_G2(self):
        partialX = numpy.zeros(shape=self.mat.shape,dtype=numpy.float32)
        partialY = numpy.zeros(shape=self.mat.shape,dtype=numpy.float32)
        partialMS = numpy.zeros(shape=self.mat.shape,dtype=numpy.float32)
        countAsymm = numpy.zeros(shape=self.mat.shape,dtype=numpy.int32)  
        g2v = numpy.array([0.0],dtype=numpy.float32)    

        px = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=partialX)
        py = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=partialY)
        pms = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=partialMS)
        ca = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=countAsymm)
        g2b = cl.Buffer(self.ctx, self.mf.WRITE_ONLY| self.mf.COPY_HOST_PTR, hostbuf=g2v)

        k = self.readKernel("g2.cl")
        prg = cl.Program(self.ctx,k).build()
        prg.getG2(self.queue,[self.rows*self.cols],None,self.gx,self.gy,self.md,self.tb,px,py,pms,ca,g2b,numpy.int32(self.rows*self.cols))
        cl.enqueue_copy(self.queue, g2v, g2b).wait()
        self.G2 = g2v[0]
        return g2v[0]

    def _get_variables_GPU(self):
       cl.enqueue_copy(self.queue, self.gradient_dx, self.gx).wait()
       cl.enqueue_copy(self.queue, self.gradient_dy,self.gy).wait()
       cl.enqueue_copy(self.queue, self.phases, self.ph).wait()
       cl.enqueue_copy(self.queue, self.mods, self.md).wait()
       cl.enqueue_copy(self.queue, self.tableau, self.tb).wait() 
       numpy.savetxt("teste.txt", (self.phases*180.0/3.1415).astype(numpy.int32),fmt='%i')

    # This function estimates both asymmetric gradient coeficient (geometric and algebric), with the given tolerances
    def evaluate(self,mtol, ftol,moment=["G2"]):
        #Buffers on GPU
        self.ph = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.phases)
        self.md = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.mods)       
        self.tb = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.tableau)
 
        self._update_Mods_Phases()
        self._update_asymmetric_tableau(mtol, ftol)
        self._get_G2()
        self._get_variables_GPU()

        return [0.0]

                 
                 
 
               


