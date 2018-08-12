"""文章病句标识"""

m = int(input())

tmp = []
# _line = input().strip().split(';')
# ret = [list(map(int, be.split(','))) for be in _line]

# line = []
for _ in range(m):
    _line = input().strip().split(';')
    line = [list(map(int, be.split(','))) for be in _line]

    tmp.extend(line)

tmp = sorted(tmp, key=lambda x: x[0])

ret = [tmp[0]]
for item in tmp:
    if ret[-1][1] >= item[0]:
        ret[-1][1] = max(ret[-1][1], item[1])
    else:
        ret.append(item)

s = ''
for item in ret[:-1]:
    s += str(item[0])+','+str(item[1])+';'
s += str(ret[-1][0])+','+str(ret[-1][1])
print(s)





