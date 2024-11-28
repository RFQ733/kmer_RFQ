import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence

kmer_count = 1
file_RNA_k_mer = "../kmer_data/{}mer_output.txt".format(kmer_count)

voc = np.load("../kmer_data/rna_dict.npy", allow_pickle=True).item()
for word in voc:
    voc[word] = voc[word].astype(np.float32)
# sentence 是一个包含句子列表的 list，其中每个句子都是词汇的列表
def getsentence():
    sentence = []
    with open(file_RNA_k_mer, "r") as f:
        for line in f:
            line = line.strip().split()
            sentence.append(line)
    return sentence

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        x = x.float()  # 将输入转换为 Float 类型
        _, (hn, _) = self.lstm(x)
        return hn[-1]

class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        x = x.float()  # 将输入转换为 Float 类型
        output, _ = self.lstm(x, hidden)
        output = self.fc(output)
        return output

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, seq_length, max_seq_length):
        context = self.encoder(x)
        context = context.unsqueeze(1).repeat(1, max_seq_length, 1)
        output = self.decoder(context, (context, torch.zeros_like(context)))
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
sentences = getsentence()
sentence_embeddings = []
for sentence in sentences:
    numpy_array = np.array([np.array(voc[word]) for word in sentence])
    embeddings = torch.tensor(numpy_array).float()  # 将输入转换为 Float 类型
    sentence_embeddings.append(embeddings)

# 将所有句子的 embeddings 打包成一个批量
padded_embeddings = pad_sequence(sentence_embeddings, batch_first=True)
padded_embeddings = padded_embeddings.to(device)

# 实例化模型
input_dim = padded_embeddings.size(2)
hidden_dim = 128
num_layers = 2
output_dim = input_dim

encoder = LSTMEncoder(input_dim, hidden_dim, num_layers)
decoder = LSTMDecoder(hidden_dim, hidden_dim, num_layers, output_dim)
model = Seq2Seq(encoder, decoder).to(device)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 无监督训练过程
model.train()
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