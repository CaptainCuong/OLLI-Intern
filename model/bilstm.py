import torch
import torch.nn as nn


class NER_BiLSTMNet(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, pos_dim, seq_len = 20, batch_size = 1):
        super(NER_BiLSTMNet, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        self.cnn = nn.Conv1d(input_dim, hidden_dim, 5, padding=2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(2*hidden_dim, num_classes)
        self.softmax = nn.Softmax(2)
        self.embedding = nn.Embedding(25, input_dim)
 
    def forward(self, x, pos_tag):
        '''
        x: sequence of encoded words
        shape of x: [batch_size, seq_len]
        hidden: initial hidden
        '''
        assert len(x.size()) == 3, 'Input must have 3 dimension (batch_size, seq_len, dim)'
        assert x.size(1) == self.seq_len, 'Sequence length is not as same as specified: expected %d, but got %d'%(self.seq_len, x.size(1))
        self.batch_size = x.size(0)
        
        pos_tag = self.embedding(pos_tag)
        x = 0.1*x+0.9*pos_tag
        # CNN
        x = torch.transpose(x,1,2)
        x = self.cnn(x)
        x = torch.transpose(x,1,2)
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
            for inputs, pos_tag, labels in train_loader:
                if torch.cuda.is_available():
                    inputs, pos_tag, labels = inputs.cuda(), pos_tag.cuda(), labels.cuda()
                out_decode = self(inputs, pos_tag).view(-1, self.num_classes)
                print(out_decode.shape)
                self.zero_grad()
                labels = labels.to(torch.long)
                loss = criterion(out_decode, labels.view(-1))
                loss.backward()
                optimizer.step()
            super().train(False)
            a = next(iter(train_loader))
            if torch.cuda.is_available():
                a = a[0].cuda(), a[1].cuda()
            b = self(a[0], pos_tag)
            x = ((a[1] == b.argmax(dim=2))).view(-1)
            print('Epoch %d:\nLoss: %f\nAccuracy: %f\n-----------\n'%(i,loss.item(),x.sum().item()/x.shape[0]))

    def eval(self, dataloader, criterion):
        score = []
        criterion = nn.BCELoss()
        for stn, label in dataloader:
            score += [criterion(self(stn), label)]
        return sum(score.item())/len(score)
