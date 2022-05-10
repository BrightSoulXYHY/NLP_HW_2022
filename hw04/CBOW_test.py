import numpy as np
from utils import WordTable,cal_similarity


word_table = WordTable()
word_table.load_dict()


# word_vec_path = "out/word_vec/20220510-164802_epoch=40_loss=2.45.vec"
word_vec_path = "out/word_vec/20220510-164802_epoch=75_loss=1.40.vec"

word_vec_all = np.loadtxt(word_vec_path)



def test1():
    nameL = [
        ("金毛狮王","谢逊"),
        ("蛛儿","殷离"),
        ("白眉鹰王","殷天正"),
        ("青翼蝠王","韦一笑"),
    ]

    for [w1,w2] in nameL:
        v1 = word_vec_all[word_table.word2id(w1)]
        v2 = word_vec_all[word_table.word2id(w2)]

        similarity = cal_similarity(v1,v2)
        prefix_text = f"{w1}&{w2}"
        print(f"{prefix_text:20s} \t similarity={similarity:.6f}")



test1()