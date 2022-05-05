import numpy as np
import time
import jieba
import os

data_dir = "../data"
out_para_file = "para_data.txt"


alpha = 5
beta = 0.1
epoch_num = 100


num_topic = 10  # 主题数量
start_time = time.time()


with open("stop_word.txt", 'r', encoding='utf-8') as fp:
    stopWordL = fp.readlines()
stopWordL = [i.strip() for i in stopWordL]

class LDA:
    def __init__(self) -> None:
        self.docs = None
        self.word2id_dict = None
        self.id2word_dict = None
        
        self.num_doc = 0
        self.num_word = 0
        self.Z = []

        # 在param_init里面初始化参数
        self.ndz = None
        self.nzw = None
        self.nz = None
        self.theta = None
        self.phi = None


    def gen_dict(self,documentL):
        word2id_dict = {}
        id2word_dict = {}
        docs = []
        cnt_document = []
        cnt_word_id = 0

        for document in documentL:
            segList = jieba.cut(document)
            for word in segList:
                word = word.strip()
                if len(word) > 1 and word not in stopWordL:
                    if word in word2id_dict:
                        cnt_document.append(word2id_dict[word])
                    else:
                        cnt_document.append(cnt_word_id)
                        word2id_dict[word] = cnt_word_id
                        id2word_dict[cnt_word_id] = word
                        cnt_word_id += 1
            docs.append(cnt_document)
            cnt_document = []
        self.docs, self.word2id_dict, self.id2word_dict = docs, word2id_dict, id2word_dict
        self.num_doc = len(self.docs)
        self.num_word = len(self.word2id_dict)


    # 随机初始化参数
    def param_init(self):
        # 各文档的词在各主题上的分布数目
        self.ndz = np.zeros([self.num_doc,num_topic]) + alpha  
        # 词在主题上的分布数
        self.nzw = np.zeros([num_topic,self.num_word]) + beta  
        # 每个主题的总词数
        self.nz = np.zeros([num_topic]) + self.num_word*beta  
        self.theta = np.zeros([self.num_doc,num_topic])
        self.phi = np.zeros([num_topic,self.num_word])

        
        for d, doc in enumerate(self.docs):
            zCurrentDoc = []
            for w in doc:
                self.pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
                z = np.random.multinomial(1, self.pz / self.pz.sum()).argmax()
                zCurrentDoc.append(z)
                self.ndz[d, z] += 1
                self.nzw[z, w] += 1
                self.nz[z] += 1
            self.Z.append(zCurrentDoc)


    # gibbs采样
    def gibbs_sampling_update(self):
        # 为每个文档中的每个单词重新采样topic
        for d, doc in enumerate(self.docs):
            for index, w in enumerate(doc):
                z = self.Z[d][index]
                # 将当前文档当前单词原topic相关计数减去1
                self.ndz[d,z] -= 1
                self.nzw[z,w] -= 1
                self.nz[z] -= 1
                # 重新计算当前文档当前单词属于每个topic的概率
                self.pz = np.divide(np.multiply(self.ndz[d,:], self.nzw[:,w]), self.nz)
                # 按照计算出的分布进行采样
                z = np.random.multinomial(1, self.pz / self.pz.sum()).argmax()
                self.Z[d][index] = z
                # 将当前文档当前单词新采样的topic相关计数加上1
                self.ndz[d, z] += 1
                self.nzw[z, w] += 1
                self.nz[z] += 1

        self.theta = [(self.ndz[i]+alpha)/(len(self.docs[i])+num_topic*alpha) for i in range(self.num_doc)]
        self.phi = [(self.nzw[i]+beta)/(self.nz[i]+self.num_word*beta) for i in range(num_topic)]

        


    def cal_perplexity(self):
        nd = np.sum(self.ndz, 1)
        n = 0
        ll = 0.0
        for d, doc in enumerate(self.docs):
            for w in doc:
                ll = ll + np.log(((self.nzw[:, w] / self.nz) * (self.ndz[d, :] / nd[d])).sum())
                n = n + 1
        return np.exp(ll / (-n))
    
    def show_topwords(self,num=10):
        for z in range(num_topic):
            ids = self.nzw[z, :].argsort()
            topicword = []
            for j in ids:
                topicword.insert(0, self.id2word_dict[j])
            # topicwords.append(topicword[:min(num, len(topicword))])
            print(topicword[:min(num, len(topicword))])

    
    def save_param(self,postfix):
        np.savetxt(f"theta_{postfix:03d}.csv", self.theta,fmt="%.9f",delimiter=',')
        np.savetxt(f"phi_{postfix:03d}.csv", self.phi,fmt="%.9f",delimiter=',')
        


if __name__ == '__main__':
    hw_lda = LDA()
    
    documentL = []
    with open(out_para_file,"r",encoding="utf-8") as fp:
        fileL = fp.readlines()
    for file in fileL:
        with open(file.strip(), 'r', encoding='utf-8') as f:
            documentL.append(f.read())

    hw_lda.gen_dict(documentL)
    print("gen_dict done")

    hw_lda.param_init()

    perplexityL = []
    for i in range(epoch_num):
        hw_lda.gibbs_sampling_update()
        perplexity = hw_lda.cal_perplexity()
        perplexityL.append(perplexity)
        print(f"[{time.time()-start_time:.2f}] epoch_num={i} perplexity={perplexity:.4f}")
        if not i%10:
            hw_lda.save_param(i)


    np.savetxt("perplexity.csv", perplexityL,fmt="%.9f",delimiter=',')

    hw_lda.show_topwords()
    