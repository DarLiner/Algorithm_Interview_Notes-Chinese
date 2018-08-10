from datetime import datetime, timedelta
from dateutil.parser import parse


def d(s1, s2):
    """"""
    week1, time1 = s1.split()
    week2, time2 = s2.split()

    sum = 0


if __name__ == "__main__":

    T = int(input())

    for i in range(T):
        K = int(input())

        time_list = []
        for K in range(K):
            line = input().split(maxsplit=2)
            week = int(line[0])
            n = int(line[1])
            ts = line[2].split()
            for i in range(n):
                beg, end = ts[i].split('-')  #
                time_list.append(str(week) + ' ' + beg)
                time_list.append(str(week) + ' ' + end)

        time_list = sorted(time_list)
        print(time_list)

        ret = []
        M = int(input())
        for i in range(M):
            line = input().strip()

            j = -1
            for t in time_list:
                j += 1
                if line < t:
                    break
            if j % 2 == 1:
                ret.append(0)
            else:
                ret.append(d(time_list[j], line))

        for i in ret:
            print(i)
