import matplotlib.pyplot as plt
import sys
from GPA import GPA as ga
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import timeit
import pyopencl as cl

def getDevice():
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
			

def printError():
        print('================================')
        print('Syntax:')
        print('python main.py Gn filename tol rad_tol')
        print('python main.py Gn -l filelist tol rad_tol output')
        print('================================')
        print('Number of given arguments:')
	print(len(sys.argv),sys.argv[1])
        exit() 

if __name__ == "__main__":
    if('-h' in sys.argv) or ('--help' in sys.argv):
        printError()
    if(sys.argv[2] == "-l") and (len(sys.argv) != 7):
        printError()
    if(sys.argv[2] != "-l") and (len(sys.argv) != 5):
        printError()        
    if not("-l" in sys.argv):  
        fileName = sys.argv[2]
        tol = float(sys.argv[3])
        rad_tol = float(sys.argv[4])

        print("Reading "+fileName)
        inputMatrix = np.loadtxt(fileName)
        inputMatrix=inputMatrix.astype(np.float32)
        gaObject = ga(inputMatrix)
        gaObject.evaluate(tol,rad_tol,[sys.argv[1]])
        if(sys.argv[1] == "G1"):
            print("NC",gaObject.n_edges)
	    print("NV",gaObject.n_points)
            print("G1",gaObject.G1)
        if(sys.argv[1] == "G2"):
	    print("Proportion",gaObject.prop)
	    print("Div",gaObject.div)
            print("G2 ",gaObject.G2)   
    else:
        files = [line.rstrip() for line in open(sys.argv[3])]
        tol = float(sys.argv[4])
        rad_tol = float(sys.argv[5])
        save = []
	dev = getDevice()

        for f in files:
            inputMatrix = np.loadtxt(f)
            inputMatrix=inputMatrix.astype(np.float32)
            gaObject = ga(inputMatrix,dev)
            gaObject.cx, gaObject.cy = len(inputMatrix[0])/2., len(inputMatrix)/2.
            gaObject.evaluate(tol,rad_tol,[sys.argv[1]])
            if(sys.argv[1] == "G1"):
                print(f+" - G1 -",gaObject.G1)
                newline = [f,gaObject.G1,gaObject.n_edges,gaObject.n_points]
                save.append(newline)
                np.savetxt(sys.argv[6], np.array(save), fmt="%s", header="file,Ga,Nc,Nv", delimiter=',')
            else:
                print(f+" - G2 -",gaObject.G2)
                newline = [f,gaObject.G2,gaObject.prop,gaObject.div,gaObject.t1,gaObject.t2,gaObject.t3]
                save.append(newline)
                np.savetxt(sys.argv[6], np.array(save), fmt="%s", header="file,G2,Na,Diversity,t1,t2,t3", delimiter=',')
        
       
    plt.show()
