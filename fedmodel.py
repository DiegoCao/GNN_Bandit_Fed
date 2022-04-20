import torch as nn
import tensorflow as tf

from tf.keras.layers import LSTM, Embedding
from tf.keras import Sequential


class NLPModel(nn.Module):
    def __init__(self,vocab):
        super(NLPModel, self).__init__()
        self.lstm_size = 200
        self.embedding_dim = 200
        self.num_layers = 2

        n_vocab = len(vocab)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)


        self.embedding = Embedding(n_vocab, self.embedding_dim)
        self.lstm = Sequential()
        self.lstm.add(LSTM(self.lstm_size, return_sequence=True, input_shape=(200, 1), drop_out = 0.2))
        self.lstm.add(LSTM(self.lstm_size, input_shape=(200, 1), drop_out = 0.2))
    

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length=10):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
