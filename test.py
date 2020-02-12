class t_c():
    def __init__(self):
        self.v = 1

def add_1(t):
    print(t)
    t = t + 1
    print(t)

def add_1_with_class(a):
    print(a.v)
    a.v = a.v + 1
    print(a.v)

abc = t_c()
print(abc.v)
abc.v = abc.v + 1
print(abc.v)
add_1_with_class(abc)
print(abc.v)


