
import turtle
import math

WIDTH = 400
HEIGHT = 400

turtle.setup(WIDTH, HEIGHT)
wn = turtle.Screen()
wn.title("Turtle oo")
t = turtle.Turtle()

t.color("red")
t.shape("turtle")

turtle.penup()
turtle.goto(-turtle.window_width()/8, turtle.window_height()/4)
turtle.pendown()

for i in range(100):
    turtle.color((((255-i)/255) % 1, (i/500) % 1, (i/255) % 1))
    turtle.forward(10)
    turtle.right(math.log(i+1))

turtle.exitonclick()

