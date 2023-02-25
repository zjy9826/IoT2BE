from itertools import chain
from random import random

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from torch import autograd

from data_utils import get_data_roadsegment
from model_network import LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_onehot(lis,num):
    lis = lis.astype(int)
    onehot_encoded = []
    for value in lis:
        letter = [0 for _ in range(num)]
        letter[value[0]] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

from sklearn.utils import shuffle
def get_data_roadsegment():
    # 加载数据

    dataset_emb = np.load("./data/dataset/allfeature/dataset-len5-id-time-embedding-allfea.npy",allow_pickle=True)

    # 打乱数据顺序
    dataset = shuffle(dataset_emb)
    # dataset = dataset_emb

    # 划分数据集，8:2:2
    train_size = int(len(dataset) * 0.8)
    trainlist = dataset[:train_size]  # 训练集
    validationlist = dataset[:int(len(trainlist) * 0.2)]  # 验证集
    testlist = dataset[train_size:]  # 测试集

    # 为了获得 targets
    #  id_emb(0:117)+idonehot[118:235)+id(236)+time(237)+druration(238)+since_start(239)+angle(240)+type(241)+64time_emb(242:305)
    L = [c for c in range(242, 306)]  #
    # L = []
    L.append(238)
    L.append(239)

    # 预处理
    length = 5  # 每个样本的长度
    look_back = length - 1

    trainX = trainlist[:, :look_back, (L)]
    trainY = trainlist[:, look_back, 238]  # onehot

    # print(train_segment_id.shape)

    validationX = validationlist[:, :look_back, (L)]
    validationY = validationlist[:, look_back, 238]

    #
    testX = testlist[:, :look_back, (L)]
    testY = testlist[:, look_back, 238]

    return trainX, trainY, validationX, validationY, testX, testY



class Encoder(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=in_features, hidden_size=hidden_size, dropout=0.5, num_layers=2)  # encoder

    def forward(self, enc_input):
        enc_input = enc_input.to(device)
        seq_len, batch_size, embedding_size = enc_input.size()
        h_0 = torch.rand(2, batch_size, self.hidden_size).to(device)
        c_0 = torch.rand(2, batch_size, self.hidden_size).to(device)
        # en_ht:[num_layers * num_directions,Batch_size,hidden_size]
        encode_output, (encode_ht, decode_ht) = self.encoder(enc_input, (h_0, c_0))
        return encode_output, (encode_ht, decode_ht)


class Decoder(nn.Module):
    def __init__(self, in_features,output_size, enc_hid_size, dec_hid_size, Attn):
        super().__init__()
        self.in_features = in_features
        self.Attn = Attn
        self.enc_hid_size = enc_hid_size
        self.dec_hid_size = dec_hid_size
        self.crition = nn.CrossEntropyLoss()
        self.fc = nn.Linear(in_features + enc_hid_size, in_features)
        self.decoder = nn.LSTM(input_size=in_features, hidden_size=dec_hid_size, dropout=0.5, num_layers=2)  # encoder
        self.linear = nn.Linear(dec_hid_size,output_size)

    def forward(self, enc_output, dec_input, s):
        # s : [1, Batch_size , enc_hid_size ] s表示解码器的某一个隐含层的输出
        # enc_output : [seq_len, Batch_size,enc_hid_size]   对应于整个解码器的某一个输入
        # dec_input : [1, Batch_size, embed_size]  对应于解码器的某一个输入
        # dec_input = dec_input.unsqueeze(1)
        seq_len, Batch_size, embed_size = enc_output.size()
        atten = self.Attn(s, enc_output)  # atten : [Batch_size, seq_len]

        atten = atten.unsqueeze(2)  # atten : [Batch_size, seq_len, 1]
        atten = atten.transpose(1, 2)  # atten : [Batch_size, 1, seq_len]
        enc_output = enc_output.transpose(0, 1)
        ret = torch.bmm(atten, enc_output)  # ret : [Batch_size, 1, enc_hid_size]
        ret = ret.transpose(0, 1)  # ret : [1, Batch_size, enc_hid_size]
        # dec_input = dec_input.transpose(0, 1)  # dec_input : [1, Batch_size, embed_size]

        # print(ret.shape)
        # print(dec_input.shape)

        dec_input_t = torch.cat((ret, dec_input), dim=2)  # dec_input_t : [1, Batch_size, enc_hid_size+embed_size]
        dec_input_tt = self.fc(dec_input_t)  # dec_input_tt : [1, Batch_size, embed_size]
        c0 = torch.zeros(2, Batch_size, embed_size)
        s = s.to(device)
        c0 = c0.to(device)
        de_output, (s, _) = self.decoder(dec_input_tt, (s, c0))  # de_output:[1, Batch_size, dec_hid_size]
        de_output = de_output.transpose(0, 1)
        pre = self.linear(de_output.view(de_output.shape[0],-1))
        pre = F.softmax(pre)
        return pre, s


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dec_hid_dim + enc_hid_dim, dec_hid_dim)
        self.fc2 = torch.nn.Linear(dec_hid_dim, 1)

    def forward(self, s, enc_output):
        # 将解码器的输出S和编码器的隐含层输出求相似性
        # s: [1, Batch_size, dec_hid_size]
        # enc_output: [seq_len, Batch_size, enc_hid_size ]
        seq_len, Batch_size, enc_hid_size = enc_output.size()
        # s = s.unsqueeze(1)  # s: [Batch_size,1, dec_hid_size]
        # print(s.shape)
        s = s.repeat(2, 1, 1)  # s: [seq_len, Batch_size, dec_hid_size]

        # print(s.shape)
        # print(enc_output.shape)

        a = torch.tanh(torch.cat((s, enc_output), 2))  # a: [Batch_size, seq_len, dec_hid_size + enc_hid_size ]
        a = self.fc1(a)  # a :  [Batch_size, seq_len, dec_hid_dim]
        a = self.fc2(a)  # a :  [Batch_size, seq_len, 1]
        a = a.squeeze(2)  # a :  [Batch_size, seq_len]
        return F.softmax(a, dim=1).transpose(0, 1)  # softmax 只进行归一化，不改变张量的维度


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, in_features, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, in_features)
        self.crition = nn.CrossEntropyLoss()

    def forward(self, enc_input, dec_input):
        enc_input = enc_input.to(device)
        dec_input = dec_input.unsqueeze(1).to(device)
        # dec_output = dec_output.to(device)

        # print(enc_input.shape)
        # print(dec_input.unsqueeze(1).shape)

        enc_input = enc_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]
        dec_input = dec_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]
        seq_len, Batch_size, embedding_size = dec_input.size()
        outputs = torch.zeros(seq_len, Batch_size, self.hidden_size)  # 初始化一个张量，用来存储解码器每一步的输出
        target_len, _, _ = dec_input.size()
        # 首先通过编码器的最后一步输出得到 解码器的第一个隐含层 ， 以及将编码器的所有的输出层作为后续提取注意力
        enc_output, (s, _) = self.encoder(enc_input)  # s : [1, Batch_size, enc_hid_size ]
        # for i in range(1, target_len):
        #     dec_output_i, s = self.decoder(enc_output, dec_input[i, :, :], s)
        #     outputs[i] = dec_output_i
        # # output:[seq_len,Batch_size,hidden_size]
        # outputs = outputs.to(device)
        # output = self.fc(outputs)
        # output = output.permute(1, 0, 2)
        # loss = 0
        # for i in range(len(output)):  # 对seq的每一个输出进行二分类损失计算
        #     loss += self.crition(output[i], dec_output[i])
        dec_output_i, s = self.decoder(enc_output, dec_input[:, :, :], s)
        return dec_output_i

def test_model(model,testX,map_dic):
    model.eval()
    permutation = torch.randperm(testX.shape[0])  # 返回一个 0到 shape[0] 随机数 的数组
    # print(len(permutation))
    rightNum = 0
    all_num = 0

    test_loss = 0
    num_test = 0

    for index in range(0, testX.shape[0], 32):  # 开始 结束 步长

        # 清除网络先前的梯度值
        with torch.no_grad():
            # 初始化隐藏层数据 GRU需要注释掉
            # model.hidden_cell = (torch.zeros(2, 4, model.hidden_layer_size).to("cuda"),
            #                      torch.zeros(2, 4, model.hidden_layer_size).to("cuda"))


            indices = permutation[index:index + 32]
            X_batch, y_batch = testX[indices], testY[indices]
            X_batch = torch.tensor(X_batch, dtype=torch.float).to("cuda")
            y_batch = torch.tensor(y_batch, dtype=torch.float).to("cuda")
            y_test = model(X_batch,X_batch[:,3,:])
            loss_test = loss_function(y_test, y_batch)
            test_loss += loss_test.item()
            num_test += 1

    avg_test_loss = test_loss / num_test
    avg_test_loss = avg_test_loss / 86400
    # print('测试集的正确率：', rate)
    return avg_test_loss


def train_model(model,trainX,trainY,testX,map_dic,optimizer,loss_function):
    epochs = 20
    batch_size = 32

    all_eva_list = []

    eva_acc = []
    eva_precision = []
    eva_recall = []
    eva_f1 = []

    for i in range(epochs):
        rightNum = 0
        all_num = 0
        # rate = 0

        training_loss = 0
        num_train = 0

        permutation = torch.randperm(trainX.shape[0])  # 返回一个 0到 shape[0] 随机数 的数组
        for index in range(0, trainX.shape[0], batch_size):  # 开始 结束 步长
            model.train()
            # 清除网络先前的梯度值
            optimizer.zero_grad()
            # 初始化隐藏层数据
            # model.hidden_cell = (torch.zeros(1, 4, model.hidden_layer_size).to("cuda"),
            #                      torch.zeros(1, 4, model.hidden_layer_size).to("cuda"))
            # model.hidden_cell = torch.tensor(model.hidden_cell).to("cuda")

            indices = permutation[index:index + batch_size]

            # print("y_batch_id:{}".format(train_segment_id[0]))
            # print("train_y:{}".format(trainY[0].index(1)))
            # print("train_y:{}".format(indices))

            X_batch = trainX[indices]
            y_batch = np.array(trainY)[indices]

            X_batch = torch.tensor(X_batch, dtype=torch.float).to("cuda")
            y_batch = torch.tensor(y_batch, dtype=torch.float).to("cuda")

            # seq = np.array(trainX[index])
            # print(X_batch.shape)
            # label = np.array(trainY[index])
            # 实例化模型
            y_pred = model(X_batch,X_batch[:,3,:])  #

            # print(y_pred.shape)
            # print(y_batch.shape)

            # 训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值
            single_loss = loss_function(y_pred, y_batch)  # 预测 128 与 emb128进行计算损失值
            single_loss.backward()  # 调用backward()自动生成梯度
            optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络

            training_loss += single_loss.item()
            num_train += 1

        avg_training_loss = training_loss / num_train
        avg_training_loss = avg_training_loss / 86400

        test_loss = test_model(model, testX, map_dic)  # 测试集

        print('训练集 epoch：{} loss : {},test_loss:{}'.format(i, avg_training_loss, test_loss))

        all_eva_list.append(test_loss)
    np.save("Evaluation_result_Data/rate0-001/batchsize32/emb/dif-layer/roadnet_time_eva_seq2seq+att-2layers.npy", all_eva_list)

if __name__ == '__main__':
    # 获取相关data trainX,trainY,validationX,validationY,testX,testY
    trainX, trainY, validationX, validationY, testX, testY = get_data_roadsegment()

    input_size = trainX.shape[2]
    output_size = 1

    # 定义相关模型
    attn = Attention(128, 128)  # enc_hid_dim, dec_hid_dim
    enc = Encoder(input_size, 128)  # input_dim, enc_hid_dim, dec_hid_dim
    dec = Decoder(input_size,output_size, 128, 128, attn)  # output_dim, enc_hid_dim, dec_hid_dim, attention

    device = "cuda"

    model = Seq2seq(enc, dec, input_size,128).to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数
    loss_function = nn.L1Loss()

    map_dic = np.load("data/map_dic/map_dir_118.npy", allow_pickle=True)
    # #
    train_model(model,trainX,trainY,testX,map_dic,optimizer,loss_function)


