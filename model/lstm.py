import torch
import torch.nn as nn

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class NER_LSTMNet(nn.Module):
    def __init__(self, vocab_size, n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len = 20, drop_prob=0.5, batch_size = 1):
        super(NER_LSTMNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_encode = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.lstm_decode = nn.LSTM(output_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fclst = [nn.Linear(hidden_dim, output_size) for i in range(seq_len)]
        self.sigmoid = nn.Sigmoid()
        self.softmaxlst = [nn.Softmax(2) for i in range(seq_len)]
 
    def forward(self, x):
        '''
        x: sequence of encoded words
        shape of x: [batch_size, seq_len]

        hidden: initial hidden
        '''
        assert len(x.size()) == 2, 'Input must have 2 dimension (batch_size, seq_len)'
        assert x.size(1) == self.seq_len, 'Sequence length is not as same as specified'
        self.batch_size = x.size(0)
        x = x.long()
        hidden = self.init_hidden(self.batch_size)
        encode_in = self.embedding(x)
        encode_out, hidden = self.lstm_encode(encode_in, hidden)
 
        pred_entity = []
        in_decode = torch.zeros([self.batch_size, 1, self.output_size])
        for i in range(self.batch_size):
            in_decode[i][0][0] = 1
        out_decode = None
        for i in range(self.seq_len):
            out_decode, hidden = self.lstm_decode(in_decode, hidden)
            out_decode = self.fclst[i](out_decode)
            out_decode = self.softmaxlst[i](out_decode)
            pred_entity.append(out_decode)

            in_decode = torch.zeros([self.batch_size, 1, self.output_size])
            max_ind = torch.argmax(out_decode, dim = 2)
            for i in range(self.batch_size):
                in_decode[i][0][max_ind[i][0]] = 1
 
        return torch.cat(pred_entity, dim = 1)
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
 
    def train(self, train_loader, epochs, batch_size, learning_rate, criterion, optimizer, clip):
        '''
        train_loader:
            input: (batch_size, seq_len, embedding_dim)
            output: (batch_size, seq_len, out_dim)
        '''
        self.batch_size = batch_size
        assert next(iter(train_loader))[0].shape[1] == self.seq_len, 'Not match length of sequences'
        assert criterion in ['BCE']
        assert optimizer in ['Adam']
        if criterion == 'BCE':
            criterion = nn.BCELoss()
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        
        loss = 0
        for i in range(epochs):
            super().train()
            for inputs, labels in train_loader:
                out_decode = self(inputs)
                self.zero_grad()
                loss = criterion(out_decode, labels)
                loss.backward()
                optimizer.step()
            super().train(False)
            print('Epoch %d:\nLoss: %f\n-----------\n'%(i,loss.item()))

    def eval(self, dataloader, criterion):
        score = []
        criterion = nn.BCELoss()
        for stn, label in dataloader:
            score += [criterion(self(stn), label)]
        return sum(score.item())/len(score)
