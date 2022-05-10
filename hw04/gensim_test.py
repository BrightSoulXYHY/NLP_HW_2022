# from gensim.models.word2vec import Word2Vec
from gensim.models import Word2Vec



save_path="out/word_vec/gensim.model"
model = Word2Vec.load(save_path)




def test1():
    nameL = [
        ("金毛狮王","谢逊"),
        ("蛛儿","殷离"),
        ("白眉鹰王","殷天正"),
        ("青翼蝠王","韦一笑"),
    ]

    for [w1,w2] in nameL:
        similarity = model.wv.similarity(w1, w2)
        prefix_text = f"{w1}&{w2}"
        print(f"{prefix_text:20s} \t similarity={similarity:.6f}")


def test2():
    testL = [ "张无忌","周芷若","屠龙刀","倚天剑" ]
    for test_text in testL:
        print(test_text)
        similarityL = model.wv.most_similar(test_text, topn=10)
        for w,sim in similarityL:
            print(f"{w} {sim:06f}")
    
    # print(sims)

# test1()
print("="*10)
test2()