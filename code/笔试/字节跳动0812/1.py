m, n = list(map(int, input().strip().split(',')))

book = []
for i in range(m):
    line = input().strip().split(',')
    line = [int(x) for x in line]
    book.append(line)


class Solution:
    def __init__(self, pos):
        self.pos = pos
        self.cnt = 0
        self.dp = []

    def dfs(self, i, j):
        if 0 <= i < m and 0 <= j < n:
            if self.pos[i][j] == 1:
                self.cnt += 1
                self.pos[i][j] = 0
                self.dfs(i - 1, j)
                self.dfs(i + 1, j)
                self.dfs(i, j - 1)
                self.dfs(i, j + 1)
                self.dfs(i - 1, j - 1)
                self.dfs(i + 1, j + 1)
                self.dfs(i + 1, j - 1)
                self.dfs(i - 1, j + 1)
        return

    def solve(self):
        for i in range(m):
            for j in range(n):
                if self.pos[i][j] == 1:
                    self.cnt = 0
                    self.dfs(i, j)
                    if self.cnt > 0:
                        self.dp.append(self.cnt)
        return len(self.dp), max(self.dp)


so = Solution(book)
ret = so.solve()
print(str(ret[0]) + ',' + str(ret[1]))
