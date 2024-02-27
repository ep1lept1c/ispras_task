import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available()  else 'cpu'

class Attention(nn.Module):
    def __init__(self, hidden_dim, is_bidirectional):
        super(Attention, self).__init__()
        self.is_bidirectional = is_bidirectional
        self.attn = nn.Linear(hidden_dim * (4 if is_bidirectional else 2), hidden_dim * (2 if is_bidirectional else 1))
        self.v = nn.Linear(hidden_dim * (2 if is_bidirectional else 1), 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        if self.is_bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            hidden = hidden[-1]
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        attn_weights = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        attn_weights = self.v(attn_weights).squeeze(2)
        return nn.functional.softmax(attn_weights, dim=1)

class LSTM_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, lstm_layers, is_bidirectional):
        super(LSTM_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = lstm_layers
        self.is_bidirectional = is_bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layers, batch_first=True, bidirectional=is_bidirectional)
        self.attention = Attention(hidden_dim, is_bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if is_bidirectional else 1), num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):

        embedded = self.embedding(x)

        out, hidden = self.lstm(embedded, hidden)
        attn_weights = self.attention(hidden[0], out)

        context = attn_weights.unsqueeze(1).bmm(out).squeeze(1)

        out = self.softmax(self.fc(context))
        return out, hidden, attn_weights

    def init_hidden(self, batch_size):

        factor = 2 if self.is_bidirectional else 1

        h0 = torch.zeros(self.num_layers * factor, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers * factor, batch_size, self.hidden_dim).to(DEVICE)
        return (h0, c0)





def evaluate(model, device, dataloader, loss_fn):
    losses  = []
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            h = model.init_hidden(y_batch.size(0))
            output, h, attn = model(x_batch, h)
            
            loss = loss_fn(output, y_batch)
            losses.append(loss.item())

            y_pred = torch.argmax(output, dim=1)

            correct += torch.sum(y_pred==y_batch).item()
            total += y_batch.size(0)

    accuracy = round(100 * correct / total, 2)
    return accuracy, np.mean(losses)

def train(model, train_loader, valid_loader, device, loss_fn, optimizer, n_epoch=6):
    early_stopping_patience = 4
    early_stopping_counter = 0

    epoch_acc_max = 0
    for epoch in range(1, n_epoch + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            h = model.init_hidden(y_batch.size(0))

            output, h, attn = model(x_batch, h)
            
            optimizer.zero_grad()
            loss = loss_fn(output, y_batch)

            loss.backward()
            optimizer.step()



        epoch_accuracy, epoch_loss = evaluate(model, device, valid_loader, loss_fn)
        print(f"Epoch {epoch}/{n_epoch} finished: train_accuracy = {epoch_accuracy}%, train_loss = {epoch_loss}")
        if epoch_accuracy >= epoch_acc_max:
            torch.save(model.state_dict(), './state_dict.pt')
            print(f'Validation accuracy increased ({epoch_acc_max:.2f} --> {epoch_accuracy:.2f}).  Saving model ...')
            epoch_acc_max = epoch_accuracy
            early_stopping_counter = 0
        else:
            print(f'Validation accuracy did not increase')
            early_stopping_counter += 1
            
        if early_stopping_counter > early_stopping_patience:
            print('Early stopped at epoch :', epoch)
            break
    return model

def test(model, device, dataloader, loss_fn):
    losses  = []
    y_pred_list = []
    y_test_list = []
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            h = model.init_hidden(y_batch.size(0))
            output, h, attn = model(x_batch, h)
            
            loss = loss_fn(output, y_batch)
            losses.append(loss.item())

            y_pred = torch.argmax(output, dim=1)

            correct += torch.sum(y_pred==y_batch).item()
            total += y_batch.size(0)
            y_pred_list.extend(y_pred.squeeze().tolist())
            y_test_list.extend(y_batch.squeeze().tolist())

    accuracy = round(100 * correct / total, 2)
    return accuracy, np.mean(losses), y_pred_list, y_test_list

