import os
import re
import json
import nltk
from tqdm import tqdm
from collections import Counter

# from .utils import *
from data_helper.squad.utils import *

sent_tokenize = nltk.sent_tokenize
word_tokenize = nltk.word_tokenize

TRAIN_PATH = r"D:\OneDrive\workspace\data\nlp\阅读理解\squad\train-v1.1.json".replace('\\', '/')
DEV_PATH = r"D:\OneDrive\workspace\data\nlp\阅读理解\squad\dev-v1.1.json".replace('\\', '/')


class SQuADGeneration:
    """"""

    def __init__(self, file_path, train_radio=1.0, train_or_test=True):
        """

        Args:
            file_path(str):
            train_radio(float):
                用于训练的数据比例：data = [train; test]
            train_or_test(bool):
                用于训练或测试：用于训练则取数据切分的前半部分
        """
        self.src_data = _load_src_data(file_path)

    def data_deal(self):
        """"""


def _load_src_data(file_path):
    """加载源数据
    SQuAD 数据结构：
        {
            'version'(str): 1.1,
            'data'(list of dict): [
                {
                    'title'(str): str,
                    'paragraphs'(list of dict): [
                        {
                            'context'(str): "Architecturally, the school has a Catholic ...",
                            'qas'(list of dict): [
                                {
                                    'id'(str): '5733be284776f41900661182',
                                    'question'(str): 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
                                    'answers'(list of dict): [
                                        {
                                            'answer_start'(int): 515,
                                            'text'(str): "Saint Bernadette Soubirous",

                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    Args:
        file_path(str):

    Usage:
        # >>> src_data = _load_src_data(TRAIN_PATH)
        # >>> src_data.keys()  # dict
        # dict_keys(['data', 'version'])
        # >>> len(src_data['data'])  # list
        # 442
        # >>> src_data['data'][0].keys()  # dict
        # dict_keys(['title', 'paragraphs'])
        # >>> src_data['data'][0]['title']
        # 'University_of_Notre_Dame'
        # >>> len(src_data['data'][0]['paragraphs'])  # list
        # 55
        # >>> src_data['data'][0]['paragraphs'][0].keys()  # dict
        # dict_keys(['context', 'qas'])
        # >>> len(src_data['data'][0]['paragraphs'][0]['context'])  # str
        # 695
        # >>> len(src_data['data'][0]['paragraphs'][0]['qas'])  # list
        # 5
        # >>> src_data['data'][0]['paragraphs'][0]['qas'][0]
        # {'answers': [{'answer_start': 515, 'text': 'Saint Bernadette Soubirous'}],
        #  'id': '5733be284776f41900661182',
        #  'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'}
    """
    return json.load(open(file_path, 'r'))


def save(data, shared, data_type):
    data_path = os.path.join('__data', "data_{}.json".format(data_type))
    shared_path = os.path.join('__data', "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def prepro_each(file_path, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    # source_path = in_path or os.path.join(args.source_dir, "{}-v1.1.json".format(data_type))
    source_data = json.load(open(file_path))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))

    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']  # 材料
            context = context.replace("''", '" ')  # 替换材料中的引号
            context = context.replace("``", '" ')
            xi = list(map(word_tokenize, sent_tokenize(context)))  #
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    w0 = xi[yi0[0]][yi0[1]]
                    w1 = xi[yi1[0]][yi1[1] - 1]
                    i0 = get_word_idx(context, xi, yi0)
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1] - 1))
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])

                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                rcx.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)

    # word2vec_dict = get_word2vec(args, word_counter)
    # lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              # 'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict
              }

    print("saving ...")
    save(data, shared, out_name)


if __name__ == '__main__':
    """"""
    prepro_each(TRAIN_PATH)
