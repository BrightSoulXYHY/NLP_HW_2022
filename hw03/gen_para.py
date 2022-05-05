import os
import re
import random
import shutil

data_dir = "../data"
out_dir = "para_data"
out_para_file = "para_data.txt"

text_fileL = [
    "鹿鼎记",
    "天龙八部",
    "笑傲江湖",
    "倚天屠龙记",
    "神雕侠侣",
]


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





if __name__ == '__main__':
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    if os.path.exists(out_para_file):
        os.remove(out_para_file)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for text_file in text_fileL:
    # text_file = text_fileL[0]
        with open(f"{data_dir}/{text_file}.txt","r",encoding="GB18030") as fp:
            text_raw = "".join(fp.readlines())
        corpus = getCorpus(text_raw)

        paraL = []
        para_len = 0
        file_id = 0
        for corpu in corpus:
            paraL.append(corpu)
            para_len += len(corpu)
            if para_len > 2000:
                para_len = 0
                with open(f"{out_dir}/{text_file}-{file_id:03d}.txt","w",encoding="utf-8") as fp:
                    fp.writelines(paraL)
                paraL = []
                file_id += 1
        
        random_paramL = [i for i in range(file_id)]
        random.shuffle(random_paramL)
        random_paramL_40 = random_paramL[:40]
        random_paramL_40.sort()
        random_paramL_40 = [f"{out_dir}/{text_file}-{i:03d}.txt\n" for i in random_paramL_40]
        with open(out_para_file,"a",encoding="utf-8") as fp:
            fp.writelines(random_paramL_40)