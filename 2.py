import numpy as np
import requests
import matplotlib.pyplot as plt
from scipy.constants import pi, speed_of_light
from scipy.special import spherical_jn as jn
from scipy.special import spherical_yn as yn
import json
import  pathlib


def hn(n, x): return jn(n, x) + 1j * yn(n, x)
def bn(n, x): return (x * jn(n - 1, x) - n * jn(n, x)) / (x * hn(n - 1, x) - n * hn(n, x))
def an(n, x): return jn(n, x) / hn(n, x)


url = requests.get('https://jenyay.net/uploads/Student/Modelling/task_02_02.txt')
par_str = url.text
variant = 2
my_str = par_str.split('\n')[variant].split()
D=float(my_str[1])
fmin=float(my_str[2])
fmax=float(my_str[3])
fstep = 1e+6
R = D / 2
freq = np.arange(fmin, fmax, fstep)
lambd = speed_of_light / freq 
k = 2 * pi / lambd


arr_sum=[((-1) ** n) * (n+0.5) * (bn(n, k * R) - an(n, k * R)) for n in range(1, 20)]
summa = np.sum(arr_sum, axis=0)
rcs = (lambd ** 2) / pi * (np.abs(summa) ** 2)


res = {
"data": [
{"freq": float(freq1), "lambd": float(lambd1),"rcs": float(rcs1)} for freq1, lambd1, rcs1 in zip(freq, lambd, rcs)
]
}

path = pathlib.Path("results")
path.mkdir(exist_ok=True)
file = path / "result_2.json"
out = file.open("w")
json.dump(res, out, indent=2)
out.close()



plt.plot(freq / 10e6, rcs)
plt.xlabel("$f, МГц$")
plt.ylabel(r"$\sigma, м^2$")
plt.grid()
plt.show()
