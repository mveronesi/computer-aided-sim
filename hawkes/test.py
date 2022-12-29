def fun()-> tuple[float, float, float]:
    t = (1, 2)
    return 3, *t

a, b, c = fun()
print(a, b, c)