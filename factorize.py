
from pylab import *


def formatSign(n):
	if n < 0:
		return f"- {abs(n)}"

	return f"+ {n}"


def factorize(a, b, c):
	n = b**2 - 4*a*c
	if n < 0:
		# Ingen løsninger
		return "No real solutions"
	elif n == 0:
		# En løsning
		x = -b / (2*a)
		xs = formatSign(-x)
		return f"f(x) = (x {xs})"
		
	# To løsninger
	n = sqrt(n)
	x1 = (-b+n)/(2*a)
	x2 = (-b-n)/(2*a)

	s1 = formatSign(-x1)
	s2 = formatSign(-x2)

	return f"f(x) = (x {s1})(x {s2})"



a, b, c = [float(i.strip()) for i in input("a, b, c: ").split(",")]

bs = formatSign(b)
cs = formatSign(c)

print(f"\nf(x) = {a}x^2 {bs}x {cs}")

print(factorize(a, b, c))


