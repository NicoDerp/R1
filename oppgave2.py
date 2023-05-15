
import matplotlib.pyplot as plt
import numpy as np
from regresjon import *

# Får numpy til å ikke printe med e men bare tallet
np.set_printoptions(suppress=True)

with open("klima.csv", "rb") as f:
    lines = f.read().decode().split("\n")
    title = lines[0].strip('"')
    lines = [line for line in lines[1:] if line != ""]
    lines = [line.strip().split(";") for line in lines]
    header = np.array([float(h.strip('"')) for h in lines[0][2:]])
    lines = lines[1:]
    colheader = [(line[0], line[1]) for line in lines]
    data = np.array([[float(x.strip('"')) for x in line[2:]] for line in lines])
    #print(data)


X = header - 1999

#func = exponential(header, np.sum(data, 0))
func = polynomial(1, X, np.sum(data, 0))
#func = polynomial(1, header, np.log(np.sum(data, 0)))
plotFunction(func, (0, 2021-1999))

plt.plot(X, np.sum(data, 0))
plt.xlabel("År")
plt.ylabel("100 tonn farlig avfall")
plt.title(title)
plt.legend()
plt.grid()
plt.show()


