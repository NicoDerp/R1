
from random import randint

#####
# 1

i = 0
while i <= 10:
    print(i)
    i += 1

#####

print("\n\n\n")

#####
# 2
i = 0
while i < 100:
    print(i)
    i += 3

#####

print("\n\n\n")

#####
# 3

t = int(input("Write a number: "))

for i in range(0, t+1):
    print(i)

#####

print("\n\n\n")

#####
# 4

correct = randint(0, 10)

while True:
    n = int(input("Guess the number between 0 and 10: "))
    if n == correct:
        print("Correct!")
        break
    
    print("Wrong!")

#####

print("\n\n\n")

#####
# 5

correct = randint(0, 10)

while True:
    n = int(input("Guess the number between 0 and 10: "))
    if n == correct:
        print("Correct!")
        break
    
    print("Wrong! Guess", "higher" if n<correct else "lower")

#####


