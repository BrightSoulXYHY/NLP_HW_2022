import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm

WEIGHT_INIT_RANGE = 0.1

class CbowDataset(Dataset):
    def __init__(self, corpus, context_size=5):
        '''
            传入one-hot形式的语料库
        '''
        self.data = []
        context_size_half = int((context_size-1)/2)
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            if len(sentence) < context_size:
                continue
            for i in range(context_size_half, len(sentence) - context_size_half):
                # 模型输入：左右分别取context_size长度的上下文
                context = sentence[i-context_size_half:i] + sentence[i+1:i+1+context_size_half]
                # 模型输出：当前词
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples])
        targets = torch.tensor([ex[1] for ex in examples])
        return (inputs, targets)



class CbowModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CbowModel, self).__init__()
        # 词嵌入层
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        # 隐含层->输出层
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        self.init_weights()

    def forward(self, inputs):
        embeds = self.embedding_layer(inputs)
        # 对上下文词向量求平均
        hidden = embeds.mean(dim=1)
        output = self.output_layer(hidden)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

    def init_weights(self):
        for name, param in self.named_parameters():
            if "embedding" not in name:
                nn.init.uniform_(param, a=-WEIGHT_INIT_RANGE, b=WEIGHT_INIT_RANGE)
    
    def word_vec(self, word_id):
        return self.embedding_layer.weight[word_id].cpu().detach().numpy()

    def word_vec_all(self):
        return self.embedding_layer.weight.cpu().detach().numpy()
    



if __name__ == '__main__':
    from utils import WordTable

    embedding_dim = 128
    word_table = WordTable()
    word_table.load_dict()
    with open("out/corpus_onehot.txt","r",encoding="utf-8") as fp:
        textL = fp.readlines()
    
    corpus_one_hot = [
        [int(i) for i in text.split(",")]
        for text in textL
    ]

    train_dataset = CbowDataset(corpus_one_hot)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, collate_fn=train_dataset.collate_fn, shuffle=True)
    
    
    # model = CbowModel(len(word_table), embedding_dim)
    # print(model.word_vec(word_id=0))


    # print(len(train_dataset))
    # print(train_dataloader[0],train_dataloader[1])
    # for inputs, targets in train_dataloader:
    #     print(inputs.shape,targets.shape)
    #     log_probs = model(inputs)
    #     print(log_probs.shape)
    #     break

    