a = 1
b = a
print(eval("a + 1"))
if True:
    print([eval("b + 1") for _ in range(5)])