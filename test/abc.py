from math import sqrt


a, b, c = [float(i.strip()) for i in input("a,b,c:").split(',')]

print(f"f(x) = {a}x^2 + {b}x + {c}")
print("f(x) = 0")

n = b**2 - 4*a*c
if n < 0:
    # Ingen løsninger
    print("Ingen reele løsninger!")
elif n == 0:
    # En løsning
    x = -b / (2*a)
    print(f"x = {x}")
else:
    # To løsninger
    n = sqrt(n)
    x1 = (-b+n)/(2*a)
    x2 = (-b-n)/(2*a)
    print(f"x = {x1}  V  x = {x2}")
