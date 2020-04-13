
from functools import reduce

a = range(1, 101)
print(reduce(lambda x, y: x + y, a))