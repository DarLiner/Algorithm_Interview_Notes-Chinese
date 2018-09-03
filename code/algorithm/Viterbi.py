Weather = ('Rainy', 'Sunny')
Activity = ('walk', 'shop', 'clean')

obs = list(range(len(Activity)))  # 观测序列
states_h = list(range(len(Weather)))  # 隐状态

# 初始概率（隐状态）
start_p = [0.6, 0.4]
# 转移概率（隐状态）
trans_p = [[0.7, 0.3],
           [0.4, 0.6]]
# 发射概率（隐状态表现为显状态的概率）
emit_p = [[0.1, 0.4, 0.5],
          [0.6, 0.3, 0.1]]


def viterbi(obs, states_h, start_p, trans_p, emit_p):
    """维特比算法"""
    dp = [[0.0] * len(states_h)] * len(obs)
    path = [[0] * len(obs)] * len(states_h)

    # 初始化
    for i in start_p:
        dp[0][i] = states_h[i] * emit_p[i][obs[0]]
        path[i][0] = i


