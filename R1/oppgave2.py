
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


# X is from 0 to (2021 - 1999) instead of 1999 to 2021
X = header - 1999
Y = np.sum(data, 0)
yearCount = 2021-1999 + 1

Xplot = np.linspace(0, yearCount, 51)
print(header, Xplot)

#func = exponential(X, Y)
#func = linear(X, Y)
func = logarithmic(X, Y)
plt.plot(Xplot+1999, func(Xplot))

print(func.prettify())

plt.plot(header, Y)
plt.xlabel("År")
plt.ylabel("1000 tonn farlig avfall")
plt.title(title)
plt.legend()
plt.grid()
plt.show()


