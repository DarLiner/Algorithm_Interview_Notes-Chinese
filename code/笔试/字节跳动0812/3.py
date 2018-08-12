import sys

n = int(input().strip())
x = []
y = []

for i in range(n):
    line = list(map(int, input().strip().split()))
    x.append(line[0])
    y.append(line[1])

xy = [(x[i], y[i]) for i in range(len(x))]
xy = sorted(xy, key=lambda tmp: tmp[1])


ret = 0
if not sum(x) % 2:
    print(sum(y))
else:
    for i in range(len(xy)):
        if xy[i][0] % 2 == 1:
            ret = sum([xy[j][1] for j in range(len(xy)) if j != i])
            print(ret)
            exit()
