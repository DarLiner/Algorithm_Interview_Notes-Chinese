n = int(input().strip())
m = int(input().strip())
_tmp = list(map(int, input().strip().split()))

s, t = [], []
for i in range(len(_tmp)):
    if i % 2 == 0:
        s.append(_tmp[i])
    else:
        t.append(_tmp[i])

if _tmp[-1] == 0:
    t[-1] = m

st = list(zip(s, t))
# print(st)

st = sorted(st, key=lambda x: x[1])
# print(st)

ret = 0
end = 0
for item in st:
    if end <= item[0]:
        ret += 1
        end = item[1]

print(ret)

