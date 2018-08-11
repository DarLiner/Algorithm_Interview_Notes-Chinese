import sys

if __name__ == "__main__":
    # 读取第一行的n

    p = dict()  # peoples

    n = int(sys.stdin.readline().strip())
    for i in range(n):

        line = sys.stdin.readline().strip().split()

        p[line[1]] = [int(line[0]), 0]

    m = int(sys.stdin.readline().strip())
    for i in range(m):
        line = sys.stdin.readline().strip().split()
        p[line[0]][1] = int(line[1])

    p_list = []
    for k, v in p.items():
        p_list.append([-v[1], -v[0], k])

    p_list = sorted(p_list, key=lambda x: (x[0], x[1], x[2]))

    for p in p_list:
        print(p[2])



