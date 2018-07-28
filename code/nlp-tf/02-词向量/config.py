"""
参数设置
"""


class config_sg:
    """"""
    batch_size = 128
    embedding_size = 128
    window_size = 4
    vocabulary_size = 50000

    # A random validation set
    valid_size = 16
    valid_window = 50

    num_sampled = 32  # 负采样

    num_steps = 100001
    top_k = 8
