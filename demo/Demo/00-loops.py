"""
PYTHON LOOPS - Class Demo
Run the whole file:  python python_loops.py
Python has only TWO loop statements: `for` and `while`.
Everything else (nested, break, continue, else, comprehensions) builds on them.
"""

def title(text):
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)


# ------------------------------------------------------------
# 1. FOR LOOP - iterate over a sequence
#    Syntax: for variable in iterable:
# ------------------------------------------------------------
title("1. for loop over a list")
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

title("1b. for loop over a string")
for ch in "Java":
    print(ch, end=" ")
print()

title("1c. for loop over tuple, set, dict")
for n in (10, 20, 30):              # tuple
    print("tuple:", n)
for n in {1, 2, 3}:                 # set (order not guaranteed)
    print("set:", n)

student = {"name": "Rajat", "course": "DSA", "level": "Advanced"}
for key in student:                          # keys (default)
    print("key:", key)
for value in student.values():               # values
    print("value:", value)
for key, value in student.items():           # both
    print(key, "->", value)


# ------------------------------------------------------------
# 2. range() - generate numbers
#    range(stop)  |  range(start, stop)  |  range(start, stop, step)
#    stop is EXCLUDED (like i < n in Java)
# ------------------------------------------------------------
title("2. range()")
print("range(5):", list(range(5)))              # 0..4
print("range(2, 7):", list(range(2, 7)))        # 2..6
print("range(0, 10, 2):", list(range(0, 10, 2)))  # even numbers
print("range(5, 0, -1):", list(range(5, 0, -1)))  # countdown

for i in range(1, 6):
    print("5 x", i, "=", 5 * i)


# ------------------------------------------------------------
# 3. enumerate() - index + value together
#    Java equivalent: for (int i = 0; i < arr.length; i++)
# ------------------------------------------------------------
title("3. enumerate()")
for index, fruit in enumerate(fruits):
    print(index, fruit)
for index, fruit in enumerate(fruits, start=1):   # start counting from 1
    print(index, fruit)


# ------------------------------------------------------------
# 4. zip() - loop over two lists in parallel
# ------------------------------------------------------------
title("4. zip()")
names = ["Asha", "Ravi", "Meena"]
marks = [88, 92, 79]
for name, mark in zip(names, marks):
    print(name, "scored", mark)


# ------------------------------------------------------------
# 5. WHILE LOOP - repeat while a condition is True
#    Syntax: while condition:
#    Use when you don't know the number of iterations in advance.
# ------------------------------------------------------------
title("5. while loop")
count = 1
while count <= 5:
    print("count =", count)
    count += 1          # forget this -> infinite loop!

title("5b. while: sum of digits")
num = 4523
total = 0
while num > 0:
    total += num % 10   # last digit
    num //= 10          # remove last digit
print("Sum of digits:", total)


# ------------------------------------------------------------
# 6. NESTED LOOPS - loop inside a loop
# ------------------------------------------------------------
title("6. nested loops: multiplication table")
for i in range(1, 4):
    for j in range(1, 6):
        print(i * j, end="\t")
    print()             # new line after each row

title("6b. nested loops: star pattern")
rows = 5
for i in range(1, rows + 1):
    print("*" * i)

title("6c. nested loops: matrix multiplication (ties to last class)")
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = [[0, 0], [0, 0]]
for i in range(2):
    for j in range(2):
        for k in range(2):
            C[i][j] += A[i][k] * B[k][j]
print(C)   # [[19, 22], [43, 50]]


# ------------------------------------------------------------
# 7. LOOP CONTROL STATEMENTS
#    break    -> exit the loop immediately
#    continue -> skip the rest of this iteration, go to the next
#    pass     -> do nothing (placeholder)
# ------------------------------------------------------------
title("7a. break")
for n in range(1, 10):
    if n == 5:
        print("Found 5, stopping.")
        break
    print(n)

title("7b. continue")
for n in range(1, 8):
    if n % 2 == 0:
        continue        # skip even numbers
    print(n)

title("7c. pass")
for n in range(3):
    pass                # empty body is allowed only with pass
print("pass does nothing; useful as a placeholder")


# ------------------------------------------------------------
# 8. for/while ... ELSE  (unique to Python!)
#    The else block runs ONLY if the loop finished WITHOUT break.
# ------------------------------------------------------------
title("8. loop-else: prime check")
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

for n in [7, 10]:
    for i in range(2, n):
        if n % i == 0:
            print(n, "is NOT prime (divisible by", i, ")")
            break
    else:                       # no break happened
        print(n, "is prime")


# ------------------------------------------------------------
# 9. INFINITE LOOP + break  (Python's "do-while")
#    Python has no do-while, so use while True + break.
# ------------------------------------------------------------
title("9. while True (do-while style)")
attempts = 0
while True:
    attempts += 1
    print("Attempt", attempts)
    if attempts == 3:       # replace with input check in real programs
        break


# ------------------------------------------------------------
