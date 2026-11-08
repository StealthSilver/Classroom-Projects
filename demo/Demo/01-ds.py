# Data structures: lists, dictionaries, sets, tuples
# A data structure is a way to store and organize a group of values.

# ---------------------------------------------------------------------------
# LIST — ordered, changeable, duplicates allowed. Use [] 
# ---------------------------------------------------------------------------
scores = [88, 92, 75, 92]
print(scores[0])          # first item (index starts at 0)
print(scores[-1])         # last item
print(scores[1:3])        # slice: from index 1 up to (not including) 3

scores.append(80)         # add at the end
scores.insert(1, 70)      # add at a given index
scores.pop()              # remove and return the last item
scores.remove(92)         # remove the first matching value
print(len(scores))        # how many items
print(scores)

# ---------------------------------------------------------------------------
# DICTIONARY — key → value pairs, changeable. Use {}
# Keys are unique. Look up a value by its key, not by position.
# ---------------------------------------------------------------------------
student = {"name": "Asha", "age": 20, "city": "Pune"}
print(student["name"])            # value for a key (error if key is missing)
print(student.get("grade", "N/A"))  # same lookup, with a default if missing

student["age"] = 21               # update an existing key
student["grade"] = "A"            # add a new key
del student["city"]               # delete a key
print(student.keys())             # all keys
print(student.values())           # all values
print(student)

# ---------------------------------------------------------------------------
# SET — unordered, unique items only, changeable. Use {} or set()
# Good for membership checks and combining groups.
# ---------------------------------------------------------------------------
colors = {"red", "blue", "green"}
colors.add("yellow")              # add one item (duplicates are ignored)
colors.remove("blue")             # delete an item
print("red" in colors)            # True if the item is in the set

a = {1, 2, 3}
b = {3, 4, 5}
print(a | b)                      # union: items in either set
print(a & b)                      # intersection: items in both sets

# ---------------------------------------------------------------------------
# TUPLE — ordered, NOT changeable, duplicates allowed. Use ()
# Use when the group of values should stay fixed.
# ---------------------------------------------------------------------------
point = (10, 20)
print(point[0])                   # read by index, same as a list
x, y = point                      # unpack into separate variables
print(x, y)
# point[0] = 5                    # this would fail: tuples cannot be changed
