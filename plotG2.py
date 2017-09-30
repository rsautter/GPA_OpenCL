import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = pd.read_csv(sys.argv[1])
fig = plt.figure(figsize=(10, 15))

plt.subplot(311)
plt.yticks(np.arange(0.0,2.0,0.3))
plt.plot(f["G2"])
plt.ylabel("G2")

plt.subplot(312)
plt.plot(f["Na"])
plt.yticks(np.arange(0.0,1.0,0.3))
plt.ylabel("Asymmetrical Proportion")

plt.subplot(313)
plt.plot(f["Diversity"])
plt.xlabel("Iteration")
plt.ylabel("Simmilarity")
plt.tight_layout()
plt.subplots_adjust(hspace=0.01, bottom=0.01)

plt.show()
