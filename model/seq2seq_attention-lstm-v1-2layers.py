from itertools import chain

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
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

    dataset = np.load("./data/dataset/allfeature/dataset-len5_GCN_121fea.npy",allow_pickle=True)

    rng = np.random.default_rng(12345)
    # 打乱数据顺序
    rng.shuffle(dataset)
    # dataset = dataset_emb

    # 划分数据集，8:2:2
    train_size = int(len(dataset) * 0.8)

    trainlist = dataset[:train_size]  # 训练集
    validationlist = dataset[:int(len(trainlist) * 0.2)]  # 验证集
    testlist = dataset[train_size:]  # 测试集
    # [n,j=5,x]
    # 为了获得 targets
    #  id_emb(0:117)+idonehot[118:235)+id(236)+time(237)+druration(238)+since_start(239)+angle(240)+type(241)+64time_emb(242:305)

    # id_emb(0:117) + id 118,+ 119 方向角 + 120 船舶类型
    L = [c for c in range(0, 118)]  #
    # L = []
    L.append(119) # 方向角
    L.append(120) # 船舶类型
    # L.append(241)

    # 预处理
    length = 5  # 每个样本的长度
    look_back = length - 1

    trainX = trainlist[:, :look_back, (L)]
    train_segment_id = trainlist[:, look_back:, 118].astype(int)
    trainY = get_onehot(train_segment_id, 118)  # onehot


    # trainX = trainlist[:, :look_back, :]
    # train_segment_id = trainlist[:, look_back:look_back + 1,0].astype(int)
    # trainY = get_onehot(train_segment_id, 118)  # onehot



    # print(train_segment_id.shape)

    validationX = validationlist[:, :look_back, (L)]
    validation_segment_id = validationlist[:, look_back:, 118].astype(int)
    validationY = get_onehot(validation_segment_id, 118)  # onehot

    # validationX = validationlist[:, :look_back, :]
    # validation_segment_id = validationlist[:, look_back:look_back + 1, 0].astype(int)
    # validationY = get_onehot(validation_segment_id, 118)  # onehot

    #
    testX = testlist[:, :look_back, (L)]
    test_segment_id = testlist[:, look_back:, 118]
    testY = get_onehot(test_segment_id, 118)  # onehot

    # testX = testlist[:, :look_back, :]
    # test_segment_id = testlist[:, look_back:look_back + 1, 0].astype(int)
    # testY = get_onehot(test_segment_id, 118)  # onehot

    return trainX,trainY,validationX,validationY,testX,testY,train_segment_id,validation_segment_id,test_segment_id


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

def test_model(model,testX,test_segment_id,map_dic):
    model.eval()
    permutation = torch.randperm(testX.shape[0])  # 返回一个 0到 shape[0] 随机数 的数组
    # print(len(permutation))
    rightNum = 0
    all_num = 0

    y_list = []
    pre_list = []

    for index in range(0, testX.shape[0], 32):  # 开始 结束 步长

        # 清除网络先前的梯度值
        with torch.no_grad():
            # 初始化隐藏层数据 GRU需要注释掉
            model.encoder.hidden_cell = (torch.zeros(1, 32, 128).to("cuda"),
                                 torch.zeros(1, 32, 128).to("cuda"))


            indices = permutation[index:index + 32]
            X_batch, y_batch = testX[indices], test_segment_id[indices]
            y_batch = y_batch.astype(np.int)
            X_batch = torch.tensor(X_batch, dtype=torch.float).to("cuda")

            # seq = np.array(trainX[index])
            # print(X_batch.shape)
            # label = np.array(trainY[index])
            # 实例化模型
            # print(X_batch)
            y_pred = model(X_batch,X_batch[:,3,:]) #
            # print(y_pred.shape)
            testYPredict_segmentID = np.argmax(y_pred.cpu().detach(), axis=1).tolist() #
            # testYPredict_segmentID = get_y_pre_id(y_pred, y_batch, map_dic)
            # print(testYPredict_segmentID)
            # print(y_batch)
            # print(list(chain.from_iterable(y_batch)))

            y_list = y_list + list(chain.from_iterable(y_batch))
            pre_list = pre_list + testYPredict_segmentID

            for i in range(len(testYPredict_segmentID)):
                if testYPredict_segmentID[i] == y_batch[i][0]:
                    rightNum = rightNum + 1
            all_num = all_num + len(testYPredict_segmentID)

    # 计算正确率
    rate = rightNum / all_num
    acc = accuracy_score(y_list,pre_list)
    precision = precision_score(y_list,pre_list,average='weighted')
    recall = recall_score(y_list,pre_list,average='weighted')
    f1 = f1_score(y_list,pre_list,average='weighted')
    # print('测试集的正确率：', rate)
    return rate,acc,precision,recall,f1


def train_model(model,trainX,trainY,train_segment_id,testX, test_segment_id,map_dic,optimizer,loss_function):

    epochs = 50
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

        permutation = torch.randperm(trainX.shape[0])  # 返回一个 0到 shape[0] 随机数 的数组
        for index in range(0, trainX.shape[0], batch_size):  # 开始 结束 步长
            model.train()
            # 清除网络先前的梯度值
            optimizer.zero_grad()
            # 初始化隐藏层数据
            # model.hidden_cell = (torch.zeros(1, 4, model.hidden_layer_size).to("cuda"),
            #                      torch.zeros(1, 4, model.hidden_layer_size).to("cuda"))
            # model.hidden_cell = torch.tensor(model.hidden_cell).to("cuda")
            hidden_cell = (torch.zeros(1, 32, 128).to("cuda"),
                                torch.zeros(1, 32, 128).to("cuda"))

            indices = permutation[index:index + batch_size]

            # print("y_batch_id:{}".format(train_segment_id[0]))
            # print("train_y:{}".format(trainY[0].index(1)))
            # print("train_y:{}".format(indices))

            X_batch = trainX[indices]
            y_batch = np.array(trainY)[indices]
            y_batch_id = train_segment_id[indices]

            X_batch = torch.tensor(X_batch, dtype=torch.float).to("cuda")
            y_batch = torch.tensor(y_batch, dtype=torch.float).to("cuda")

            # seq = np.array(trainX[index])
            # print(X_batch.shape)
            # label = np.array(trainY[index])
            # 实例化模型
            y_pred = model(X_batch,X_batch[:,3,:])  #

            # print(y_pred.shape)
            # print(y_batch.shape)
            # # print(y_batch_id)
            # # print(exit())
            # print(y_batch)
            # y_pred = np.array(y_pred.detach().cpu()) # GRU
            # 计算损失，反向传播梯度以及更新模型参数
            # 训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值
            single_loss = loss_function(y_pred, y_batch)  # 预测 118 与 118进行计算损失值
            single_loss.backward()  # 调用backward()自动生成梯度
            optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络

            # print(y_pred)
            # 获得预测 id
            # id_list = get_y_pre_id(y_pred,y_batch_id,map_dic)

            Predict_segmentID = np.array(np.argmax(y_pred.detach().cpu(), axis=1).tolist())  # one-hot解码
            # Predict_segmentID = get_y_pre_id(y_pred, y_batch_id, map_dic)
            for k in range(len(Predict_segmentID)):
                if Predict_segmentID[k] == y_batch_id[k]:
                    rightNum = rightNum + 1
            all_num = all_num + y_pred.shape[0]

        # 计算正确率
        rate = rightNum / all_num  # 训练集
        test_rate,acc,precision,recall,f1 = test_model(model, testX, test_segment_id, map_dic)  # 测试集

        print('训练集 epoch：{} 的(train)正确率：{},loss : {},test_acc1:{},test_acc2:{},precision:{},recall:{},f1:{}'.format(i, rate, single_loss.item(),test_rate,acc,precision,recall,f1))

        # 保存 eva
        eva_acc.append(test_rate)
        eva_precision.append(precision)
        eva_recall.append(recall)
        eva_f1.append(f1)

    all_eva_list.append(eva_acc)
    all_eva_list.append(eva_precision)
    all_eva_list.append(eva_recall)
    all_eva_list.append(eva_f1)
    np.save("./newemb_result/roadnet_eva_seq2seq+att-lstm-2layers.npy", all_eva_list)

if __name__ == '__main__':
    # 获取相关data trainX,trainY,validationX,validationY,testX,testY
    trainX, trainY, validationX, validationY, testX, testY, train_segment_id, validation_segment_id, test_segment_id = get_data_roadsegment()

    input_size = trainX.shape[2]
    output_size = 118

    # 定义相关模型
    attn = Attention(128, 128)  # enc_hid_dim, dec_hid_dim
    enc = Encoder(input_size, 128)  # input_dim, enc_hid_dim, dec_hid_dim
    dec = Decoder(input_size,output_size, 128, 128, attn)  # output_dim, enc_hid_dim, dec_hid_dim, attention

    device = "cuda"

    model = Seq2seq(enc, dec, input_size,128).to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数
    loss_function = nn.MSELoss()

    map_dic = np.load("data/map_dic/map_dir_118.npy", allow_pickle=True)  # 没有使用
    # #
    train_model(model,trainX,trainY,train_segment_id,testX, test_segment_id,map_dic,optimizer,loss_function)


