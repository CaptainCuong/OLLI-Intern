import torch
import torch.nn as nn


class NER_BiLSTMNet(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, seq_len = 20, batch_size = 1):
        super(NER_BiLSTMNet, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(2*hidden_dim, num_classes)
        self.softmax = nn.Softmax(2)
        
 
    def forward(self, x):
        '''
        x: sequence of encoded words
        shape of x: [batch_size, seq_len]
        hidden: initial hidden
        '''
        assert len(x.size()) == 3, 'Input must have 3 dimension (batch_size, seq_len, dim)'
        assert x.size(1) == self.seq_len, 'Sequence length is not as same as specified: expected %d, but got %d'%(self.seq_len, x.size(1))
        self.batch_size = x.size(0)
        
        h = torch.zeros(2, self.batch_size, self.hidden_dim)
        c = torch.zeros(2, self.batch_size, self.hidden_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)
        class_tensor = self.linear(lstm_out)
        tuned_class = self.softmax(class_tensor)
        return tuned_class
     
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
                out_decode = self(inputs).view(-1, self.num_classes)
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
