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
    def __init__(self, n_entity, output_size, embedding_dim, hidden_dim, n_layers, seq_len = 20, drop_prob=0.5, batch_size = 1):
        super(NER_LSTMNet, self).__init__()
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.gpu = torch.cuda.is_available()
        
        if n_layers > 1:
            self.lstm_encode = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
            self.lstm_decode = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        else:
            self.lstm_encode = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
            self.lstm_decode = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fclst = nn.ModuleList([nn.Linear(hidden_dim, output_size) for i in range(seq_len)])
        self.sigmoid = nn.Sigmoid()
        self.softmaxlst = nn.ModuleList([nn.Softmax(2) for i in range(seq_len)])
 
    def forward(self, x):
        '''
        x: sequence of encoded words
        shape of x: [batch_size, seq_len]
        hidden: initial hidden
        '''
        assert len(x.size()) == 3, 'Input must have 3 dimension (batch_size, seq_len, dim)'
        assert x.size(1) == self.seq_len, 'Sequence length is not as same as specified: expected %d, but got %d'%(self.seq_len, x.size(1))
        self.batch_size = x.size(0)
        # ENCODE
        x = x.type(torch.FloatTensor).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        hidden = self.init_hidden(self.batch_size)
        encode_out, hidden = self.lstm_encode(x, hidden)
        
        # DECODE
        pred_entity = []
        out_decode = torch.zeros([self.batch_size, 1, self.embedding_dim])
        for i in range(self.seq_len):
            out_decode, hidden = self.lstm_decode(out_decode, hidden)
            pred = self.fclst[i](out_decode)
            pred = self.softmaxlst[i](pred)
            pred_entity.append(pred)
 
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
        assert criterion in ['CE']
        assert optimizer in ['Adam']
        if criterion == 'CE':
            criterion = nn.CrossEntropyLoss()
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        
        loss = 0
        for i in range(epochs):
            super().train()
            for inputs, labels in train_loader:
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                out_decode = self(inputs).view(-1, self.output_size)
                self.zero_grad()
                labels = labels.to(torch.long)
                loss = criterion(out_decode, labels.view(-1))
                loss.backward()
                optimizer.step()
            super().train(False)
            a = next(iter(train_loader))
            if torch.cuda.is_available():
                a = a[0].cuda(), a[1].cuda()
            b = self(a[0])
            x = ((a[1] == b.argmax(dim=2))).view(-1)
            print('Epoch %d:\nLoss: %f\nAccuracy: %f\n-----------\n'%(i,loss.item(),x.sum().item()/x.shape[0]))

    def eval(self, dataloader, criterion):
        score = []
        criterion = nn.BCELoss()
        for stn, label in dataloader:
            score += [criterion(self(stn), label)]
        return sum(score.item())/len(score)
