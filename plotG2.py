import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = pd.read_csv(sys.argv[1])
fig = plt.figure(figsize=(10, 10))

plt.subplot(311)
plt.yticks(np.arange(0.0,2.0,0.3))
plt.plot(f["G2"])
plt.ylabel("$G_2$")

plt.subplot(312)
plt.plot(f["Na"])
plt.yticks(np.arange(0.0,1.0,0.3))
plt.ylabel("$N_a$")

plt.subplot(313)
plt.plot(100.0*f["Diversity"])
plt.xlabel("Iteration")
plt.ylabel("Similarity ($10^{-2}$)")
#plt.yticks(np.arange(0.0,0.8,0.3))
plt.tight_layout()
plt.subplots_adjust(hspace=0.02, bottom=0.06)

plt.savefig("plot.png")
