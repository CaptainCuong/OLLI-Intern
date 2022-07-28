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
        self.batch_size = x.size(0)
        
        pos_tag = 0.5*self.embedding(pos_tag)
        x = 0.2*x+0.8*pos_tag
        # CNN
        x = torch.transpose(x,1,2)
        x = self.cnn(x)
        x = torch.transpose(x,1,2)
        # LSTM
        lstm_out, _ = self.lstm(x)
        class_tensor = self.linear(lstm_out)
        tuned_class = self.softmax(class_tensor)
        return tuned_class
     
    def train(self, train_loader, test_loader, epochs, batch_size, learning_rate, criterion, optimizer, clip):
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
        accuracy = 0
        for i in range(epochs):
            super().train()
            for inputs, pos_tag, labels in train_loader:
                if torch.cuda.is_available():
                    inputs, pos_tag, labels = inputs.cuda(), pos_tag.cuda(), labels.cuda()
                out_decode = self(inputs, pos_tag).view(-1, self.num_classes)
                self.zero_grad()
                labels = labels.to(torch.long)
                loss = criterion(out_decode, labels.view(-1))
                num_prob = out_decode[:,1]
                num_true = (labels.type(torch.int8).view(-1)==1).type(torch.float32)
                # BCE Loss for num
                num_prob = out_decode[:,1]
                num_true = (labels.type(torch.int8).view(-1)==1).type(torch.float32)
                loss2 = torch.nn.BCELoss()(num_prob, num_true)
                loss = loss + 5*loss2
                loss.backward()
                optimizer.step()
            super().train(False)

            # VAL ACCURACY
            a = next(iter(test_loader))
            if torch.cuda.is_available():
                a = a[0].cuda(), a[1].cuda(), a[2].cuda()
            b = self(a[0], a[1]).argmax(dim=2)
            valid_ele = 0
            x = 0
            for d1 in range(b.shape[0]):
              for d2 in range(b.shape[1]):
                if a[2][d1][d2] != 0:
                  valid_ele += 1
                  if a[2][d1][d2] == b[d1][d2]:
                    x += 1
                else:
                  break
            # x = (a[2] == b).view(-1)
            acc = x/valid_ele
            print('Epoch %d:\nLoss: %f\nAccuracy: %f\n-----------\n'%(i,loss.item(),acc))
            if acc >= accuracy:
                torch.save(self.state_dict(), './bilstm_model.pt')
                accuracy = acc

    def eval(self, dataloader, criterion):
        score = []
        criterion = nn.BCELoss()
        for stn, label in dataloader:
            score += [criterion(self(stn), label)]
        return sum(score.item())/len(score)
