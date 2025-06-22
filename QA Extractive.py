from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
dataset = load_dataset("squad")["train"]
#building vocab

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for example in data_iter:
        yield tokenizer(example["context"])
        yield tokenizer(example["question"])

vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>", "<pad>", "<sep>"])
vocab.set_default_index(vocab["<unk>"])

#creating model
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.start_fc = nn.Linear(hidden_dim, 1)
        self.end_fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.embedding(x) #(batch_size,seq_len)->(batch_size,seq_len,embed_dim)
        x, _ = self.lstm(x) #(batch_size,seq_len,embed_dim)->(batch_size,seq_len,hidden_dim)
        start= self.start_fc(x).squeeze(-1) #(batch_size,seq_len,hidden_dim)->(batch_size,seq_len)
        end = self.end_fc(x).squeeze(-1)
        return start, end
text_pipline=lambda x: torch.tensor(vocab(tokenizer(x)), dtype=torch.long)
#making batch of data
batch=[]
for i, example in enumerate(dataset):
    tokens_context = tokenizer(example["context"])
    tokens_question = tokenizer(example["question"])
    tokens_answer = tokenizer(example["answers"]["text"][0])

    context = [vocab[token] for token in tokens_context]
    question = [vocab[token] for token in tokens_question]
    answer=[vocab[token]for token in tokens_answer]

    start,end=None,None
    for j in range(len(context)-len(answer)+1):
        if context[j:j+len(answer)] == answer:
            start = j
            end = j + len(answer) - 1
            break
    if start is None or end is None:
        continue
    context = torch.tensor(context, dtype=torch.long)
    question = torch.tensor(question, dtype=torch.long)
    start = torch.tensor(start, dtype=torch.long)
    end = torch.tensor(end, dtype=torch.long)
    batch.append({"context": context, "question": question, "start": start, "end": end})
#creating dataloader
def collate_fn(batch):
    inputs= [torch.cat([example["question"],torch.tensor([vocab["<sep>"]],dtype=torch.long),example["context"]]) 
            for example in batch]
    starts = [example["start"]+len(example["question"])+1 for example in batch]
    ends = [example["end"]+len(example["question"])+1 for example in batch]
    starts = torch.tensor(starts,dtype = torch.long)
    ends = torch.tensor(ends,dtype = torch.long)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    return inputs, starts, ends
data_loader = DataLoader(list(batch), batch_size=32, collate_fn=collate_fn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#training the model
model = LSTM(len(vocab), 128, 128).to(device)
optimizer=optim.Adam(model.parameters(),lr=.001)
criterion = nn.CrossEntropyLoss()
num_epochs=30
for epoch in range(num_epochs):
    total_loss=0
    model.train()
    for i,(inputs,starts,ends) in enumerate(data_loader):
        inputs,starts,ends = inputs.to(device), starts.to(device), ends.to(device)
        optimizer.zero_grad()
        start_logits, end_logits = model(inputs)
        start_loss = criterion(start_logits, starts)
        end_loss = criterion(end_logits, ends)
        loss= (start_loss + end_loss)/2
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch+1} loss is {total_loss}")
#evaluating the model
data_loader_eval = DataLoader(list(batch), batch_size=32,shuffle = True, collate_fn=collate_fn)
inputs_eval, starts_eval, ends_eval = next(iter(data_loader_eval))
inputs_eval, starts_eval, ends_eval = inputs_eval.to(device), starts_eval.to(device), ends_eval.to(device)
data_eval = list(zip(inputs_eval, starts_eval, ends_eval))
model.eval()
total=0
with torch.no_grad():
    for input,start,end in data_eval:
        input = input.unsqueeze(0)
        input = input.to(device)
        start_logits,end_logits = model(input)
        start_pred = torch.argmax(start_logits, dim=1)
        end_pred = torch.argmax(end_logits, dim=1)
        if (start_pred.item() ==start.item()) and (end_pred.item()== end.item()):
            print("correct")
            total+=1
        else:
            print("wrong")
print(f"accuracy is {total/len(data_eval):.2f}%")
