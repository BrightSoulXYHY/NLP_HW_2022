# -*- coding: utf-8 -*-
# @Date    : 2022-04-05 21:08:12
# @Author  : BrightSoul (653538096@qq.com)

import os
import re
import numpy as np
import jieba


re_preprocess = re.compile('[a-zA-Z0-9’"#$%&\'()*+,-./:：;<=>?@?★、…【】《》？“”‘’！[\\]^_`{|}~]+')

# 预处理函数，将原始文本处理为断句断好的列表
def getCorpus(text_raw):
    text_raw = re_preprocess.sub("",text_raw)
    punctuationL =["\t","\n","\u3000","\u0020","\u00A0"," "]
    for i in punctuationL:
        text_raw = text_raw.replace(i,"")
    text_raw = text_raw.replace("，","。")
    corpus = text_raw.split("。")
    return corpus

# 获得字的统计字典
def getCharacterFrequency(corpus,n=1):
    cf_dict = {}
    for line in corpus:
        for i in range(len(line)+1-n):
            key = "".join(line[i:i+n])
            cf_dict[key] = cf_dict.get(key,0) + 1
    return cf_dict

# 获得词的统计字典
def getWordFrequency(corpus,n=1):
    cf_dict = {}
    for line in corpus:
        words = list(jieba.cut(line))
        for i in range(len(words)+1-n):
            key = tuple(words[i:i+n])
            cf_dict[key] = cf_dict.get(key,0) + 1
    return cf_dict

# 计算字的信息熵
def calChineseCharacterEntropy(corpus,n=1):
    entropy = -1
    if n > 1:
        cf_dict_n = getCharacterFrequency(corpus,n)
        cf_dict_n1 = getCharacterFrequency(corpus,n-1)
        all_sum = np.sum(list(cf_dict_n.values()))
        entropy = -np.sum([v*np.log2(v/cf_dict_n1[k[:n-1]]) for k,v in  cf_dict_n.items()])/all_sum
    else:
        cf_dict = getCharacterFrequency(corpus)
        all_sum = np.sum(list(cf_dict.values()))
        entropy = -np.sum([i*np.log2(i/all_sum) for i in  cf_dict.values()])/all_sum
    return entropy

# 计算单词信息熵
def calChineseWordEntropy(corpus,n=1):
    entropy = -1
    if n > 1:
        cf_dict_n = getWordFrequency(corpus,n)
        cf_dict_n1 = getWordFrequency(corpus,n-1)
        all_sum = np.sum(list(cf_dict_n.values()))
        entropy = -np.sum([v*np.log2(v/cf_dict_n1[k[:n-1]]) for k,v in  cf_dict_n.items()])/all_sum
    else:
        cf_dict = getWordFrequency(corpus)
        all_sum = np.sum(list(cf_dict.values()))
        entropy = -np.sum([i*np.log2(i/all_sum) for i in  cf_dict.values()])/all_sum
    return entropy



if __name__ == '__main__':
    txt_path = "../data"
    text_fileL = os.listdir(txt_path)
    for text_file in text_fileL:
        print(text_file,end=" ")
        with open(f"{txt_path}/{text_file}","r",encoding="GB18030") as fp:
            text_raw = "".join(fp.readlines())
        corpus = getCorpus(text_raw)
        # cf_dict_word = getWordFrequency(corpus)
        # all_sum_word = np.sum(list(cf_dict_word.values()))
        # cf_dict_char = getCharacterFrequency(corpus)
        # all_sum_char = np.sum(list(cf_dict_char.values()))
        # print(f"{all_sum_char} {all_sum_word}")

        for i in range(1,4):
            entropy = calChineseCharacterEntropy(corpus,i)
            print(f"{entropy:.4f}",end=" ")
        print()
        # for i in range(1,4):
        #     entropy = calChineseWordEntropy(corpus,i)
        #     print(f"{entropy:.4f}",end=" ")
        # print()

