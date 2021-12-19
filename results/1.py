import matplotlib.pyplot as plt
import numpy as np
import csv 
import pathlib
import math




def f(x): return -20*np.exp(-0.2*(0.5*x**2)**1/2)-np.exp(0.5*(np.cos(2*np.pi*x)+1))+math.e+20

x_min = -5
x_max = 5
dx = 0.01
x = np.arange(x_min, x_max+dx, dx)
y = f(x)



path = pathlib.Path("results")
path.mkdir(exist_ok=True)

csvfile="N x y \n"
for i in range(0, len(x)):
    csvfile+= str(i)+", " +str(x[i])+ ', '  +str(y[i])+"\n"

file = path/"result_1.csv"
out = file.open("w")
out.write(csvfile)
out.close




plt.plot(x, y)
plt.grid()
plt.savefig("results/task1.png")
plt.show()
