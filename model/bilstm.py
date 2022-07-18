import torch
import torch.nn as nn


class NER_BiLSTMNet(nn.Module):
    def __init__(self, n_entity, num_classes, input_dim, hidden_dim, seq_len = 20, drop_prob=0.5, batch_size = 1):
        super(NER_BiLSTMNet, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        self.lstm_cell_forward = nn.LSTMCell(input_dim, hidden_dim)
        self.lstm_cell_backward = nn.LSTMCell(input_dim, hidden_dim)
        self.linearlst = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(seq_len)])
        self.softmax = nn.Softmax()
 
    def forward(self, x):
        '''
        x: sequence of encoded words
        shape of x: [batch_size, seq_len]
        hidden: initial hidden
        '''
        assert len(x.size()) == 3, 'Input must have 3 dimension (batch_size, seq_len, dim)'
        assert x.size(1) == self.seq_len, 'Sequence length is not as same as specified: expected %d, but got %d'%(self.seq_len, x.size(1))
        self.batch_size = x.size(0)
        
        hs_forward = torch.zeros(x.size(0), self.hidden_dim)
        cs_forward = torch.zeros(x.size(0), self.hidden_dim)
        hs_backward = torch.zeros(x.size(0), self.hidden_dim)
        cs_backward = torch.zeros(x.size(0), self.hidden_dim)

        forward = []
        backward = []

        # Unfolding Bi-LSTM
        # Forward
        for i in range(self.seq_len):
            hs_forward, cs_forward = self.lstm_cell_forward(out[i], (hs_forward, cs_forward))
            forward.append(hs_forward)
            
        # Backward
        for i in reversed(range(self.seq_len)):
            hs_backward, cs_backward = self.lstm_cell_backward(out[i], (hs_backward, cs_backward))
            backward.append(hs_backward)

        # LSTM
        ret = []
        for fwd, bwd in zip(forward, backward):
            cat_tensor = torch.cat((fwd, bwd), 1)
            ret.append(self.softmax(cat_tensor))
    
     
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
