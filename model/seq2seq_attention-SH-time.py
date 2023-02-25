from itertools import chain
from random import random

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

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

    dataset_emb = np.load("./data/dataset/allfeature/SH-len5-id-onehot-time-emb-allfea.npy",allow_pickle=True)

    # 打乱数据顺序
    dataset = shuffle(dataset_emb)
    # dataset = dataset_emb

    # 划分数据集，8:2:2
    train_size = int(len(dataset) * 0.8)
    trainlist = dataset[:train_size]  # 训练集
    validationlist = dataset[:int(len(trainlist) * 0.2)]  # 验证集
    testlist = dataset[train_size:]  # 测试集

    # 为了获得 targets
    # id_emb(0:127),onehot(128:178),id(179),time(180),since_last(181),since_start(182),midnight(183) weekday(184) weakday_second(185) 类别(186) caseid(187) time_emb(188:251)

    # L = [c for c in range(188, 252)]  #
    L = []
    L.append(181)
    L.append(182)
    # L1.append(240)
    # L1.append(241)
    #



    # 预处理
    length = 5  # 每个样本的长度
    look_back = length - 1

    trainX = trainlist[:, :look_back, (L)]
    trainY = trainlist[:, look_back, 181]  # onehot

    # print(train_segment_id.shape)

    validationX = validationlist[:, :look_back, (L)]
    validationY = validationlist[:, look_back, 181]

    #
    testX = testlist[:, :look_back, (L)]
    testY = testlist[:, look_back, 181]

    return trainX,trainY,validationX,validationY,testX,testY

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional=True,batch_first=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim) # 单层双向GRU
        # self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        '''
        src = [batch_size,src_len,features]
        emb_dim = features_dim
        '''
        # embedded = self.dropout(self.embedding(src)).transpose(0, 1)  # embedded = [src_len, batch_size, emb_dim]

        enc_output, enc_hidden = self.rnn(src)  # if h_0 is not give, it will be set 0 acquiescently

        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))

        return enc_output, s


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [batch_size,src_len, enc_hid_dim * 2]

        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        # enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, input_size,output_dim, enc_hid_dim, dec_hid_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + input_size, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + input_size, output_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input_embedded, s, enc_output):
        # dec_input_embedded = [batch_size,1 , emb_dim]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [batch_size,src_len, enc_hid_dim * 2]

        # dec_input = dec_input.unsqueeze(1) # dec_input = [batch_size, 1]

        # embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1) # embedded = [1, batch_size, emb_dim]

        dec_input_embedded = dec_input_embedded.unsqueeze(1)

        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1)

        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        # enc_output = enc_output.transpose(0, 1)

        # c = [1, batch_size, enc_hid_dim * 2]
        # c = torch.bmm(a, enc_output).transpose(0, 1)

        # print(a.shape)
        # print(enc_output.shape)

        c = torch.bmm(a, enc_output)  # c = [batch_size,1 , enc_hid_dim * 2]

        # print(dec_input_embedded.shape)
        # print(c.shape)
        # rnn_input = [batch_size,1 , (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((dec_input_embedded, c), dim=2)

        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))

        # dec_input_embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        dec_input_embedded = dec_input_embedded.squeeze(1)
        dec_output = dec_output.squeeze(1)
        c = c.squeeze(1)

        # print(dec_input_embedded.shape)
        # print(dec_output.shape)
        # print(c.shape)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, dec_input_embedded), dim=1))
        # pred = F.softmax(pred)

        return pred, dec_hidden.squeeze(0)


import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

    # def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    def forward(self, src):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing

        # batch_size = src.shape[1]
        # trg_len = trg.shape[0]
        # trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        # outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        # dec_input = trg[0,:]

        # for t in range(1, trg_len):

        #     # insert dec_input token embedding, previous hidden state and all encoder hidden states
        #     # receive output tensor (predictions) and new hidden state
        #     dec_output, s = self.decoder(dec_input, s, enc_output)

        #     # place predictions in a tensor holding predictions for each token
        #     outputs[t] = dec_output

        #     # decide if we are going to use teacher forcing or not
        #     teacher_force = random.random() < teacher_forcing_ratio

        #     # get the highest predicted token from our predictions
        #     top1 = dec_output.argmax(1)

        #     # if teacher forcing, use actual next token as next input
        #     # if not, use predicted token
        #     dec_input = trg[t] if teacher_force else top1

        '''
        改动
        '''
        dec_input = src[:, 3, :]
        dec_output, s = self.decoder(dec_input, s, enc_output)
        # top1 = dec_output.argmax(1)

        return dec_output

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
            y_test = model(X_batch)
            loss_test = loss_function(y_test, y_batch)
            test_loss += loss_test.item()
            num_test += 1

    avg_test_loss = test_loss / num_test
    avg_test_loss = avg_test_loss / 86400
    # print('测试集的正确率：', rate)
    return avg_test_loss


def train_model(model,trainX,trainY,testX,map_dic,optimizer,loss_function):
    epochs = 100
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
            y_pred = model(X_batch)  #

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
    np.save("Evaluation_result_Data/rate0-001/batchsize32/emb/SH_time_eva_seq2seq+att.npy", all_eva_list)

if __name__ == '__main__':
    # 获取相关data trainX,trainY,validationX,validationY,testX,testY
    trainX, trainY, validationX, validationY, testX, testY = get_data_roadsegment()

    input_size = trainX.shape[2]
    output_size = 1

    # 定义相关模型
    attn = Attention(128, 128)  # enc_hid_dim, dec_hid_dim
    enc = Encoder(input_size, 128, 128)  # input_dim, enc_hid_dim, dec_hid_dim
    dec = Decoder(input_size,output_size, 128, 128, attn)  # output_dim, enc_hid_dim, dec_hid_dim, attention

    device = "cuda"

    model = Seq2Seq(enc, dec, device).to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数
    loss_function = nn.L1Loss()
    # loss_function = nn.MSELoss()
    # loss_function = nn.L1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)

    map_dic = np.load("data/map_dic/map_dir_118.npy", allow_pickle=True)
    # #
    train_model(model,trainX,trainY,testX,map_dic,optimizer,loss_function)


