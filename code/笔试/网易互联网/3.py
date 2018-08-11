def swap(array, i, j):
    temp = array[i]
    array[i] = array[j]
    array[j] = temp


def perm(ret, s, step):
    if step == len(s) - 1:
        ret.append(''.join(s))
    else:
        for i in range(step, len(s)):
            if i != step and s[i] == s[step]:
                continue
            swap(s, i, step)
            tmp = [_ for _ in s]
            perm(ret, tmp, step + 1)


n, m, k = list(map(int, input().strip().split()))

s = list(n * 'a' + m * 'z')

if not s:
    print(-1)
    exit()

ret = []
perm(ret, s, 0)

print(sorted(ret)[k - 1])
