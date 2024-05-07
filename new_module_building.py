import torch.nn as nn
import torch
from loss_func import *
import loss_func
from function1 import ReverseLayerF


def data_in_one(inputdata):
    min = np.nanmin(inputdata.cpu())

    max = np.nanmax(inputdata.cpu())

    outputdata = (inputdata.cpu() - min) / (max - min)

    #return outputdata.cuda()
    return outputdata


class target_encoder_model(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(target_encoder_model, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (7, 7), stride=(3, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(latent_dim, n_classes, bias=False)

    def forward(self, x, mold):
        #print("x1.device:",x.device)
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        #print("x1.device:",x.device)
        x = data_in_one(x.detach())
        #print("x1.device:",x.device)
        #print("22")

        x = x.to('cuda')
        x = self.encoder(x)
        #print("33")
        #print(x.shape)
        x = x.view(-1, 96 * 19 * 9)
        #print('x.shape',x.shape)       
        x2 = self.classifier(x)
        #print('x2.shape',x2.shape)
        # print('x.shape',x.shape)
        if mold == 'classifier':
            return x, x2
        return x


class source_encoder_model(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(source_encoder_model, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (7, 7), stride=(3, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(latent_dim, n_classes, bias=False)

    def forward(self, x, mold):
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = data_in_one(x.detach())
        x = x.to('cuda')
        x = self.encoder(x)
        # print(x.shape)
        x = x.view(-1, 96 * 19 * 9)
        x2 = self.classifier(x)
        #print('x2.shape',x2.shape)
        if mold == 'classifier':
            return x, x2
        return x


class target_classifier(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(target_classifier, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x2 = self.fc1(x)
        return x2


class source1_classifier(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(source1_classifier, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x2 = self.fc1(x)
        return x2


class source2_classifier(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(source2_classifier, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x2 = self.fc1(x)
        return x2


class cnn_model_domain(nn.Module):
    def __init__(self, n_domain, latent_dim):
        super(cnn_model_domain, self).__init__()
        self.n_classes = n_domain
        self.latent_dim = latent_dim
        self.fc2 = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, n_domain),
            nn.Dropout(0.5),
            nn.Softmax(dim=1)
        )

    def forward(self, x, alpha):
        x1 = ReverseLayerF.apply(x, alpha)
        domain_output = self.fc2(x1)
        return domain_output


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def __multi_time(self, size):
        size_temp = list(size)
        size_temp = [size_temp[0] * size_temp[1]] + size_temp[2:]
        return tuple(size_temp)

    def __dist_time(self, size, batch, time_dim):
        size_temp = list(size)
        size_temp = [batch, time_dim] + size_temp[1:]
        return tuple(size_temp)

    def forward(self, x):
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(self.__multi_time(x.size()))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        y = y.contiguous().view(self.__dist_time(y.size(), x.size(0), x.size(1)))  # (samples, timesteps, output_size)

        return y

class UT_HAR_CNN_GRU(nn.Module):
    def __init__(self):
        super(UT_HAR_CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (250,90)
            nn.Conv1d(250,250,12,3),
            nn.ReLU(True),
            nn.Conv1d(250,250,5,2),
            nn.ReLU(True),
            nn.Conv1d(250,250,5,1)
            # 250 x 8
        )
        self.gru = nn.GRU(8,128,num_layers=1)
    def forward(self,x):
        # batch x 1 x 250 x 90
        x = x.view(-1,250,90)
        x = self.encoder(x)
        # batch x 250 x 8
        x = x.permute(1,0,2)
        # 250 x batch x 8
        _, ht = self.gru(x)
        outputs = ht[-1]
        return outputs


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
class UT_HAR_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=7):
        super(UT_HAR_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, 7, stride=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, kernel_size=(10, 11), stride=1),
            nn.ReLU()
        )
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def UT_HAR_ResNet18():
    return UT_HAR_ResNet(Block, [2, 2, 2, 2])
def UT_HAR_ResNet50():
    return UT_HAR_ResNet(Bottleneck, [3,4,6,3])
def UT_HAR_ResNet101():
    return UT_HAR_ResNet(Bottleneck, [3,4,23,3])

class target_classifier_for_ResNet18(nn.Module):
    def __init__(self):
        super(target_classifier_for_ResNet18, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 7),
        )

    def forward(self, x):
        outputs = self.classifier(x)
        return outputs


class source_classifier_for_ResNet18(nn.Module):
    def __init__(self):
        super(source_classifier_for_ResNet18, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512,7),
        )

    def forward(self, x):
        outputs = self.classifier(x)
        return outputs


class source2_classifier_for_ResNet18(nn.Module):
    def __init__(self):
        super(source2_classifier_for_ResNet18, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512,7),
        )

    def forward(self, x):
        outputs = self.classifier(x)
        return outputs


#cnn+gru一组 ----------------------
class CNN_GRU_for_target(nn.Module):
    def __init__(self):
        super(CNN_GRU_for_target, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1000, 180, 9, 3),  # 12 4
            nn.ReLU(True),
            nn.Conv1d(180, 100, 6, 2),  # 7 3
            nn.ReLU(True),
            nn.Conv1d(100, 70, 3, 2),  # 100 60 3 1
            nn.ReLU(True),
            nn.Conv1d(70, 70, 2, 1),
            nn.ReLU(True),
        )
        self.lstm = nn.LSTM(12, 180, num_layers=1)
        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(12, 128),  # Adjust the input size of Linear layer
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(180, 5, bias=False)

    def forward(self, x,mold):
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = x.view(-1, 1000, 180)
        #x = self.linear(x)
        x = self.encoder(x)
        x = x.permute(1,0,2)
        #x = self.linear(x)
        _, (ht,ct) = self.lstm(x)
        x = nn.Dropout(p=0.3)(ht[-1])
        x2 = self.classifier(x)
        #print('x3.shape',x2.shape)
        if mold == 'classifier':
            return x, x2
        return x


class CNN_GRU_for_source(nn.Module):
    def __init__(self):
        super(CNN_GRU_for_source, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1000, 180, 9, 3),  # 12 4
            nn.ReLU(True),
            nn.Conv1d(180, 100, 6, 2),  # 7 3
            nn.ReLU(True),
            nn.Conv1d(100, 70, 3, 2),  # 100 60 3 1
            nn.ReLU(True),
            nn.Conv1d(70, 70, 2, 1),
        )
        self.lstm = nn.LSTM(12, 180, num_layers=1)
        self.linear = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(12, 128),  # Adjust the input size of Linear layer
            #nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(180, 5, bias=False)

    def forward(self, x,mold):
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = x.view(-1, 1000, 180)
        #x = self.linear(x)
        x = self.encoder(x)
        x = x.permute(1,0,2)
        #x = self.linear(x)
        _, (ht,ct) = self.lstm(x)
        x = nn.Dropout(p=0.3)(ht[-1])
        x2 = self.classifier(x)
        if mold == 'classifier':
            return x, x2
        return x

class target_classifier_for_CNN_GRU(nn.Module):
    def __init__(self):
        super(target_classifier_for_CNN_GRU, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(180, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        outputs = self.classifier(x)
        return outputs

class source_classifier_for_CNN_GRU(nn.Module):
    def __init__(self):
        super(source_classifier_for_CNN_GRU, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(180,5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        outputs = self.classifier(x)
        return outputs


class source2_classifier_for_CNN_GRU(nn.Module):
    def __init__(self):
        super(source2_classifier_for_CNN_GRU, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(180,5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        outputs = self.classifier(x)
        return outputs

#-----------

class feature_generator(nn.Module):
    def __init__(self, n_classes):
        super(feature_generator, self).__init__()
        self.n_classes = n_classes
        self.fc2 = nn.Sequential(
            nn.Linear(2 * n_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 96 * 4 * 4),
            nn.Tanh(),
        )
        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)

    def forward(self, noise, labels):
        # labels = self.label_emb(labels)
        inputs = torch.cat((noise, labels), -1)
        # inputs = inputs.type(torch.FloatTensor)
        generator_feature = self.fc2(inputs)
        return generator_feature


# 输入是编码器特征和标签的concat
class feature_discriminator(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(feature_discriminator, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.fc2 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )
        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)

    def forward(self, features, labels):
        # labels = self.label_emb(labels)
        inputs = torch.concat([features, labels], -1)
        inputs = inputs.type(torch.FloatTensor).cuda()
        discriminator_output = self.fc2(inputs)

        return discriminator_output


class feature_discriminator_DIFA(nn.Module):
    def __init__(self, latent_dim):
        super(feature_discriminator_DIFA, self).__init__()
        self.latent_dim = latent_dim
        self.fc2 = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, features):
        discriminator_output = self.fc2(features)
        return discriminator_output


class class_model_for_SA_GAN(nn.Module):
    def __init__(self, n_classes, latent_dim):
        super(class_model_for_SA_GAN, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        def discriminator_block1(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, (7, 7), (3, 1)), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]  # Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))  # 如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block

        def discriminator_block2(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, (5, 4), (2, 2), (1, 0)), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]  # Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))  # 如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block

        def discriminator_block3(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, (3, 3), (1, 1)), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]  # Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))  # 如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block

        self.model = nn.Sequential(
            *discriminator_block1(1, 32, bn=False),
            nn.MaxPool2d(2),
            *discriminator_block2(32, 64),
            nn.MaxPool2d(2),
            *discriminator_block3(64, 96),
            nn.MaxPool2d(2),
        )
        self.adv_layer = nn.Sequential(nn.Linear(latent_dim, n_classes), nn.Softmax(dim=1))  # 先进行线性变换，再进行激活函数激活
        # 上一句中 128是指model中最后一个判别模块的最后一个参数决定的，ds_size由model模块对单张图片的卷积效果决定的，而2次方是整个模型是选取的长宽一致的图片

    def forward(self, inputs):
        out = self.model(inputs)
        out = out.view(out.shape[0], -1)  # 将处理之后的数据维度变成batch * N的维度形式
        output = self.adv_layer(out)
        return output


class feature_generator_for_SA_GAN(nn.Module):
    def __init__(self, latent_dim, noise_shape, input_dim):
        super(feature_generator_for_SA_GAN, self).__init__()
        self.latent_dim = latent_dim
        self.noise_shape = noise_shape
        self.input_dim = input_dim
        self.l1 = nn.Sequential(nn.Linear(input_dim + noise_shape, 96 * 5 * 3))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 64, (1, 1), stride=(1, 1)),
            nn.Upsample(scale_factor=10),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),  # relu激活函数
            nn.Conv2d(64, 32, (1, 1), stride=(1, 1)),  # 二维卷积
            nn.Upsample(scale_factor=(5, 3)),
            nn.BatchNorm2d(32, 0.8),  # BN
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, (1, 1), stride=(1, 1)),
            nn.Tanh(),  # Tanh激活函数
        )

    def forward(self, noise, inputs):
        inputs = inputs.view(-1, 1 * 250 * 90)
        inputs = torch.cat((noise, inputs), -1)
        # inputs=torch.FloatTensor(inputs)
        out = self.l1(inputs)  # l1函数进行的是Linear变换 （第50行定义了）
        out = out.view(out.shape[0], 96, 5,
                       3)  # view是维度变换函数，可以看到out数据变成了四维数据，第一个是batch_size(通过整个的代码，可明白),第二个是channel，第三,四是单张图片的长宽
        generator_output = self.conv_blocks(out)
        # print(generator_output.shape)
        return generator_output


class feature_discriminator_for_SA_GAN(nn.Module):
    def __init__(self, latent_dim):
        super(feature_discriminator_for_SA_GAN, self).__init__()
        self.latent_dim = latent_dim

        def discriminator_block1(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, (7, 7), stride=(3, 1)), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]  # Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))  # 如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block

        def discriminator_block2(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, (5, 4), stride=(2, 2), padding=(1, 0)),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]  # Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))  # 如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block

        def discriminator_block3(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, (3, 3), stride=(1, 1)), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]  # Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))  # 如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block

        self.model = nn.Sequential(
            *discriminator_block1(1, 32, bn=False),
            nn.MaxPool2d(2),
            *discriminator_block2(32, 64),
            nn.MaxPool2d(2),
            *discriminator_block3(64, 96),
            nn.MaxPool2d(2),
        )
        self.adv_layer = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())  # 先进行线性变换，再进行激活函数激活
        # 上一句中 128是指model中最后一个判别模块的最后一个参数决定的，ds_size由model模块对单张图片的卷积效果决定的，而2次方是整个模型是选取的长宽一致的图片

    def forward(self, inputs):
        # print(inputs.shape)
        out = self.model(inputs)
        # print(out.shape)
        out = out.view(out.shape[0], -1)  # 将处理之后的数据维度变成batch * N的维度形式
        # print(out.shape)
        validity = self.adv_layer(out)
        return validity


    def forward(self, features):
        discriminator_output = self.fc2(features)
        return discriminator_output

class feature_discriminator_for_WiADG(nn.Module):
    def __init__(self, latent_dim):
        super(feature_discriminator_for_WiADG, self).__init__()
        self.latent_dim = latent_dim
        self.fc2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 2),
            nn.LogSoftmax()
        )

    def forward(self, features):
        discriminator_output = self.fc2(features)
        return discriminator_output