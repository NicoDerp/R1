from pylab import *


a = float(input())
c = float(input())
x = linspace(-10, 10, 1001)

for i in range(5):
    y = a * x**i + i*x + c
    plot(x, y)

title("Funksjon")
xlabel("x")
ylabel("f(x)")
xlim(-5, 5)
ylim(-5, 20)
axhline(y=0, color="k")
axvline(x=0, color="k")
grid()
show()

