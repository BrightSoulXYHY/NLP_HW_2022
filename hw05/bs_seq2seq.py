import torch
import torch.nn as nn
from torch.utils.data import Dataset



class BS_Seq2Seq_Data(Dataset):
    def __init__(self,corpus_onehotL):
        self.corpus_onehot_LL = []
        self.max_len = 131+2
        for corpus_onehot in corpus_onehotL:
            if corpus_onehot.strip() == "":
                continue
            data_floatL = [int(i) for i in corpus_onehot.strip().split(",")]
            self.corpus_onehot_LL.append(data_floatL)

    

    def __getitem__(self,index):
        inputs = self.corpus_onehot_LL[index]
        targets = self.corpus_onehot_LL[index+1]
        # "1": "<PAD>",
        # "2": "<BOS>",
        # "3": "<EOS>",
        inputs = inputs + [1] * (self.max_len-len(inputs))
        targets = [2]+ targets + [3] + [1]*(self.max_len-2-len(targets))


        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)
        return (inputs, targets)
    
    def __len__(self):
        return len(self.corpus_onehot_LL)-1
    
    # def collate_fn(self, examples):

    #     inputsL = []
    #     targetsL = []
    #     for inputs,targets in examples:
    #         inputsL.append( inputs + [1] * (self.max_len-len(inputs)))
    #         targetsL.append([2]+ targets + [3] + [1]*(self.max_len-2-len(targets)))

    #     inputs = torch.tensor(inputsL)
    #     targets = torch.tensor(targetsL)
    #     return inputs, targets


class Encoder(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(corpus_len,encoder_embedding_num)
        self.lstm = nn.LSTM(encoder_embedding_num,encoder_hidden_num,batch_first=True)

    def forward(self,en_index):
        en_embedding = self.embedding(en_index)
        _,encoder_hidden =self.lstm(en_embedding)
        return encoder_hidden

class Decoder(nn.Module):
    def __init__(self,decoder_embedding_num,decoder_hidden_num,corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(corpus_len,decoder_embedding_num)
        self.lstm = nn.LSTM(decoder_embedding_num,decoder_hidden_num,batch_first=True)

    def forward(self,decoder_input,hidden):
        embedding = self.embedding(decoder_input)
        decoder_output,decoder_hidden = self.lstm(embedding,hidden)
        return decoder_output,decoder_hidden



class Seq2Seq(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,decoder_embedding_num,decoder_hidden_num,corpus_len):
        super().__init__()
        self.encoder = Encoder(encoder_embedding_num,encoder_hidden_num,corpus_len)
        self.decoder = Decoder(decoder_embedding_num,decoder_hidden_num,corpus_len)
        self.classifier = nn.Linear(decoder_hidden_num,corpus_len)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,inputs_tensor,targets_tensor):
        decoder_input = targets_tensor[:,:-1]
        label = targets_tensor[:,1:]

        encoder_hidden = self.encoder(inputs_tensor)
        decoder_output,_ = self.decoder(decoder_input,encoder_hidden)

        pre = self.classifier(decoder_output)
        loss = self.cross_loss(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))

        return loss
    
    def predict(self,inputs_tensor):
        result = []
        tensor_device = inputs_tensor.device
        encoder_hidden = self.encoder(inputs_tensor)
        decoder_hidden = encoder_hidden
        # "2": "<BOS>",

        decoder_input = torch.tensor([[2]],device=tensor_device)
        while True:
            decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            pre = self.classifier(decoder_output)

            w_index = int(torch.argmax(pre,dim=-1))

            # "3": "<EOS>",
            if w_index == 3 or len(result) > 133:
                break

            result.append(w_index)
            decoder_input = torch.tensor([[w_index]],device=tensor_device)
        return result


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    data_path = "out/corpus_onehot.txt"
    with open(data_path,"r",encoding="utf-8") as fp:
        textL = fp.readlines()
    
    bs_dataset = BS_Seq2Seq_Data(textL)
    bs_dataloader = DataLoader(bs_dataset,batch_size=128,shuffle=False)

    encoder_embedding_num = 50
    encoder_hidden_num = 100
    decoder_embedding_num = 107
    decoder_hidden_num = 100
    corpus_len = 45118
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = Seq2Seq(encoder_embedding_num,encoder_hidden_num,decoder_embedding_num,decoder_hidden_num,corpus_len)
    # model = model.to(device)

    # print(bs_dataloader)
    # import random
    # print(random.randint(0,len(textL)-1))
    # for inputs,targets in bs_dataloader: 
    #     print(inputs.shape,targets.shape)
    #     inputs,targets = inputs.to(device),targets.to(device)
    #     loss = model(inputs,targets)
    #     break