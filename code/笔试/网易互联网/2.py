

n = int(input())
a = list(map(int, input().strip().split()))

a_sum = []
sum = 0
for i in a:
    sum += i
    a_sum.append(sum)
# print(a_sum)

m = int(input())
q = list(map(int, input().strip().split()))
# print(q)

for i in q:
    for j in range(len(a_sum)):
        if i <= a_sum[j]:
            print(j+1)
            break
