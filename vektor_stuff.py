import math


class Vector:
    def __init__(self, values):
        if isinstance(values, Point):
            self.values = values.coord
        else:
            self.values = tuple(values)
        self.dim = len(self.values)

    def format(self):
        return "[" + ", ".join([str(v) for v in self.values]) + "]"

    def __repr__(self):
        return self.format()

    def __str__(self):
        return self.format()

    def _check(self, other):
        if not isinstance(other, Vector):
            raise ValueError("Other is not vector")
        if self.dim != other.dim:
            raise ValueError("Tried dot but dimensions {self.dim} does not match {other.dim} for {self} and {other}")

    def add(self, other):
        self._check(other)
        return Vector([a+b for a, b in zip(self.values, other.values)])

    def scale(self, n):
        return Vector([n*v for v in self.values])

    def magnitude(self):
        return math.sqrt(sum([v**2 for v in self.values]))

    def magnitudeSquared(self):
        return sum([v**2 for v in self.values])

    def dot(self, other):
        self._check(other)
        return sum([a*b for a, b in zip(self.values, other.values)])

    def cross(self, other):
        self._check(other)
        if self.dim != 3:
            raise Exception("Dimensions aren't 3 for cross product")
        #angle = math.acos(self.dot(other) / (self.magnitude() * other.magnitude()))
        x = self.values[1] * other.values[2] - self.values[2] * other.values[1]
        y = self.values[2] * other.values[0] - self.values[0] * other.values[2]
        z = self.values[0] * other.values[1] - self.values[1] * other.values[0]
        return Vector((x, y, z))

    def ortho(self):
        if self.dim != 2:
            raise ValueError("Only works for 2 dimensions")
        return Vector((-self.values[1], self.values[0]))

    def isOrtho(self, other):
        self._check(other)
        if self.dim != 2:
            raise ValueError("Only works for 2 dimensions")
        d = self.dot(other)
        return True if d == 0 else False

class Point:
    def __init__(self, coord):
        self.coord = tuple(coord)
        self.dim = len(coord)

    def sub(self, other):
        return Point([a-b for a, b in zip(self.coord, other.coord)])

def vectorPointDistance(A, B, P):
    AB = Vector(B.sub(A))
    PA = Vector(A.sub(P))
    k = -AB.dot(PA) / AB.magnitudeSquared()
    PQ = PA.add(AB.scale(k))
    return PQ.magnitude()

def area1(A, B, h):
    g = Vector(B.sub(A)).magnitude()
    return g*h/2

def area2(A, B, P):
    u = Vector(B.sub(A))
    v = Vector(P.sub(A))
    vtverr = v.ortho()
    T = abs(u.dot(vtverr))/2
    return T

A = Point((1, -8))
B = Point((5, 8))
P = Point((6, 2))
h = vectorPointDistance(A, B, P)
print(area1(A, B, h))
print(area2(A, B, P))

u = Vector((4, 1))
v = Vector((-2, 8))

if u.isOrtho(v):
    print("Vektorene er ortogonale")
else:
    print("Vektorene er ikke ortogonale")


