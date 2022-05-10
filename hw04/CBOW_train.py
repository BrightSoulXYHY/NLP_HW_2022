import torch
from torch.utils.data import DataLoader

from utils import WordTable
from CBOW import CbowDataset,CbowModel


from tqdm import tqdm

import time
import logging
import numpy as np

context_size = 5
embedding_dim = 128
batch_size = 1024
num_epoch = 10
learning_rate = 0.01
num_epochs = 50


if __name__ == '__main__':
    start_time = time.time()
    time_str = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f'log/{time_str}.log',
        format='%(asctime)s - %(levelname)s: %(message)s'
    )



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    word_table = WordTable()
    word_table.load_dict()
    with open("out/corpus_onehot.txt","r",encoding="utf-8") as fp:
        textL = fp.readlines()
    
    corpus_one_hot = [
        [int(i) for i in text.split(",")]
        for text in textL
    ]

    train_dataset = CbowDataset(corpus_one_hot,context_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
    
    
    model = CbowModel(len(word_table), embedding_dim)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    print(f"[{time.time()-start_time:.2f}] start train")
    for epoch in range(num_epochs):
        total_loss = 0
        val_loss = 0
        pbar = tqdm(train_dataloader)
        if epoch >= 40:
            learning_rate = 0.001
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
        for iteration,[inputs, targets] in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            log_probs = model(inputs)
            loss = criterion(log_probs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            val_loss = total_loss / (iteration + 1)
            

            desc_str = f"{'Train':8s} [{epoch + 1}/{num_epochs}] loss:{val_loss:.6f}"
            pbar.desc = f"{desc_str:40s}"
        
        if not (epoch+1)%5:
            word_vec_all = model.word_vec_all()
            np.savetxt(f"out/word_vec/{time_str}_epoch={epoch+1}_loss={val_loss:.2f}.vec",word_vec_all,fmt="%.6f")

        logging.info(f"epoch={epoch+1} total_loss={val_loss:.6f}")