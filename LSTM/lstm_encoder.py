import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence

kmer_count = 1
file_RNA_k_mer = "../kmer_data/{}mer_output.txt".format(kmer_count)

voc = np.load("../kmer_data/rna_dict.npy", allow_pickle=True).item()

# sentence 是一个包含句子列表的 list，其中每个句子都是词汇的列表
def getsentence():
    sentence = []
    with open(file_RNA_k_mer, "r") as f:
        for line in f:
            line = line.strip().split()
            sentence.append(line)
    return sentence

sentences = getsentence() 

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return hn[-1]  # 返回最后一个时间步的隐藏状态作为上下文向量

class LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, embedding_dim, batch_first=True)
        self.fc = nn.Linear(embedding_dim, embedding_dim)  # 将输出映射回嵌入维度

    def forward(self, context, seq_length, max_seq_length):
        # 将上下文向量转换为重复的序列
        context = context.unsqueeze(1).repeat(1, max_seq_length, 1)
        lstm_out, _ = self.lstm(context)
        return self.fc(lstm_out)  # 解码器的输出

class LSTMAutoencoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(embedding_dim, hidden_dim)
        self.decoder = LSTMDecoder(embedding_dim, hidden_dim)

    def forward(self, x, seq_length, max_seq_length):
        context = self.encoder(x)
        reconstructed = self.decoder(context, seq_length, max_seq_length)
        return reconstructed

# 假设 embedding_dim 是你的词向量的维度
embedding_dim = len(next(iter(voc.values())))
hidden_dim = 64  # 可以调整这个参数
device = torch.device("cpu")

# 创建模型实例
model = LSTMAutoencoder(embedding_dim, hidden_dim).to(device)

# 将句子中的每个词汇替换为词向量
sentence_embeddings = []
for sentence in sentences:
    numpy_array = np.array([np.array(voc[word]) for word in sentence])  
    embeddings = torch.tensor(numpy_array).float()
    sentence_embeddings.append(embeddings)

# 将所有句子的 embeddings 打包成一个批量
padded_embeddings = pad_sequence(sentence_embeddings, batch_first=True)
padded_embeddings = padded_embeddings.to(device)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 无监督训练过程
for epoch in range(10):  # 训练多个周期
    optimizer.zero_grad()
    seq_length = (padded_embeddings != 0).sum(dim=1)  # 获取序列长度
    max_seq_length = seq_length.max().item()  # 获取最大序列长度
    reconstructed = model(padded_embeddings, seq_length, max_seq_length)  # 前向传播
    
    # 计算重构损失
    loss = nn.MSELoss()(reconstructed, padded_embeddings)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# 使用模型得到句子的定长向量表示
# context_vectors = model.encoder(padded_embeddings)  # 获取上下文向量
# np.save("./sentence_vector_autoencoder_without_attention.npy", context_vectors.detach().cpu().numpy())
# 使用模型得到句子的定长向量表示
context_vectors = model.encoder(padded_embeddings)  # 获取上下文向量
import os 
if not os.path.exists("../../drive/MyDrive/"):
    os.makedirs("../../drive/MyDrive/")
np.save("../../drive/MyDrive/1mersentence_vector_autoencoder_without_attention.npy", context_vectors.detach().cpu().numpy())
np.save("./sentence_vector_autoencoder_without_attention.npy", context_vectors.detach().cpu().numpy())