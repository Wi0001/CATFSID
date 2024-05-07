import torch
import torch.nn as nn
import argparse
from model_util_for_train_for_new import load_data_n_model
import numpy as np
from sklearn.model_selection import KFold
from loss_func import *
import loss_func
import itertools
from itertools import cycle
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

def gaussian_noise(csi, epsilon):
    noise = torch.normal(0, 1, size=csi.shape)
    perturbed_csi = csi + epsilon * noise
    return perturbed_csi


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten() 
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def train(train_loader, unlabel_train_loader, epoch, xent, triplet, Smooth_L1_loss, criterion2, encoder_for_target,
          classifier_for_target, classifier_for_source1, classifier_for_source2, discriminator, rand_sdomain_1,
          rand_sdomain_2, device):
    encoder_for_target.to('cuda')
    optimizer = torch.optim.Adam(encoder_for_target.parameters(), lr=0.000691)
    optimizer1 = torch.optim.Adam(classifier_for_target.parameters(), lr=0.000691)
    optimizer2 = torch.optim.Adam(classifier_for_source1.parameters(), lr=0.0003)
    optimizer3 = torch.optim.Adam(classifier_for_source2.parameters(), lr=0.0003)
    optimizer4 = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    target_iter = iter(unlabel_train_loader)
    for i, inputs in enumerate(train_loader):
        try:
            inputs_target = next(target_iter)
        except:
            target_iter = iter(unlabel_train_loader)
            inputs_target = next(target_iter)
        input_s, label_s, domain_s = inputs
        input_t, label_t, domain_t = inputs_target
        label_s = label_s.to(device)
        label_s = label_s.type(torch.LongTensor)
        input_s = input_s.to(device)
        input_t = input_t.to(device)
        #input_s = input_s.to(device)
        encoder_feat = encoder_for_target(input_s, 'no')
        label_s1 = np.asarray(label_s)
        domain_s1 = np.asarray(domain_s)
        #print("label_s_shape:",label_s.shape)
        #print("encoder_feat_shape:",encoder_feat.shape)
        idx = []
        for c_id in rand_sdomain_1:
            if len(np.where(c_id == domain_s1)[0]) == 0:
                continue
            else:
                idx.append(np.where(c_id == domain_s1)[0])
        if idx == [] or len(idx[0]) == 1:
            print('error')
            exit()
            idx = [np.asarray([a]) for a in range(self.cfg.SOLVER.IMS_PER_BATCH)]
        idx = np.concatenate(idx)
        label_1 = torch.LongTensor(label_s1[idx])
        feat_1 = encoder_feat[idx]
        idx = []
        for c_id in rand_sdomain_2:
            if len(np.where(c_id == domain_s1)[0]) == 0:
                continue
            else:
                idx.append(np.where(c_id == domain_s1)[0])
        if idx == [] or len(idx[0]) == 1:
            print('error')
            exit()
            idx = [np.asarray([a]) for a in range(self.cfg.SOLVER.IMS_PER_BATCH)]
        idx = np.concatenate(idx)
        label_2 = torch.LongTensor(label_s1[idx])
        feat_2 = encoder_feat[idx]

       
        if epoch < 30:
            #print("--1------------------")
            #encoder_feat = encoder_feat
            class_1 = classifier_for_target(encoder_feat)
            #print("--a------------------")
            label_s = label_s.to(device)
            class_1 = class_1
            #print("label_s.device:",label_s.device)
            #print("class_1.device",class_1.device)
            #print("Current device:", torch.cuda.current_device())
            class_loss_1 = xent(class_1, label_s)
            #print("--b------------------")
            total_loss = class_loss_1
            predict = torch.argmax(class_1, dim=1).to('cuda')
            #print("----1")
            epoch_accuracy = (predict == label_s).sum().item() / label_s.size(0)
            #print("----2")
            #epoch_accuracy=1
            
            optimizer.zero_grad()
            optimizer1.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer1.step()
            print('Epoch: [{}][{}/{}]\t''class_loss_1: {:.3f}' 'Acc:{:.4f}'.format(epoch, i + 1, len(train_loader),
                                                                       class_loss_1.item(),epoch_accuracy))
        
      
        if (30 <= epoch < 45) or (60 <= epoch < 75):
            class_1 = classifier_for_source1(feat_1)
            class_2 = classifier_for_source2(feat_2)
            class_loss_1 = xent(class_1, label_1)
            class_loss_2 = xent(class_2, label_2)
            #print("-----------111---------")
            tri_loss = triplet(encoder_feat, label_s)
            label_s = label_s.to('cuda')
            #print("-----------------22222------------")
            #total_loss = 0.2*class_loss_1 + 0.2*class_loss_2 + 0.6*tri_loss[0]
            class_1 = class_1.to('cuda')
            class_2 = class_2.to('cuda')
            predict1 = torch.argmax(class_1, dim=1).to('cuda')
            predict2 = torch.argmax(class_2, dim=1).to('cuda')
            label_1 = label_1.to('cuda')
            label_2 = label_2.to('cuda')
            epoch_accuracy1 = (predict1 == label_1).sum().item() / label_1.size(0)
            epoch_accuracy2 = (predict2 == label_2).sum().item() / label_2.size(0)
            total_loss = class_loss_1 + class_loss_2 + tri_loss[0]
          
            optimizer.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer2.step()
            optimizer3.step()
            print('Epoch: [{}][{}/{}]\t'
                  'total_loss: {:.3f}  class_loss_1: {:.3f}  class_loss_2: {:.3f}   tri_loss: {:.3f}  Acc1:{:.4f} Acc2:{:.4f} '
                  .format(epoch, i + 1, len(train_loader),
                          total_loss.item(), class_loss_1.item(), class_loss_2.item(), tri_loss[0].item(),epoch_accuracy1,epoch_accuracy2
                          ))
        elif (45 <= epoch < 60) or (75 <= epoch < 90):
            class_1 = classifier_for_source2(feat_1)
            class_2 = classifier_for_source1(feat_2)
            class_loss_1 = xent(class_1, label_1)
            class_loss_2 = xent(class_2, label_2)
            tri_loss = triplet(encoder_feat, label_s)
            #total_loss = 0.2*class_loss_1 + 0.2*class_loss_2 + 0.6*tri_loss[0]
            class_1 = class_1.to('cuda')
            class_2 = class_2.to('cuda')
            predict1 = torch.argmax(class_1, dim=1).to('cuda')
            predict2 = torch.argmax(class_2, dim=1).to('cuda')
            label_1 = label_1.to('cuda')
            label_2 = label_2.to('cuda')
            epoch_accuracy1 = (predict1 == label_1).sum().item() / label_1.size(0)
            epoch_accuracy2 = (predict2 == label_2).sum().item() / label_2.size(0)
            total_loss = class_loss_1 + class_loss_2 + tri_loss[0]
       
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print('Epoch: [{}][{}/{}]\t'
                  'total_loss: {:.3f}  class_loss_1: {:.3f}  class_loss_2: {:.3f}   tri_loss: {:.3f}  Acc1:{:.4f} Acc2:{:.4f}'
                  .format(epoch, i + 1, len(train_loader),
                          total_loss.item(), class_loss_1.item(), class_loss_2.item(), tri_loss[0].item(),epoch_accuracy1,epoch_accuracy2
                          ))

        elif 90 <= epoch:
            class_1 = classifier_for_source2(feat_1)
            class_2 = classifier_for_source1(feat_2)
            class_loss_1 = xent(class_1, label_1)
            class_loss_2 = xent(class_2, label_2)
            #print("----------------------------1---------------")
            encoder_feat_t, encoder_class_t = encoder_for_target(input_s, 'classifier')
            #print("encoder_feat_t.shape:",encoder_feat_t.shape)
            #print("label_s.shape:",label_s.shape)
            #print("label_t.shape:",label_t.shape)
            label_t = label_t.to(label_1.dtype)
            if 90 <= epoch < 100:
                feat_concat = torch.cat((feat_1.cuda(), encoder_feat_t.cuda()), 0)
                environment_concat = discriminator(feat_concat.detach())
                #label_t.to(torch.long)
                label_t = label_t.to(label_1.dtype)
                label_concat = torch.cat((torch.ones_like(label_1), torch.zeros_like(label_t)), 0).cuda()
                #label_concat = torch.round(label_concat).to(torch.long)
                d_loss = criterion2(environment_concat, label_concat)
                optimizer4.zero_grad()
                d_loss.backward()
                optimizer4.step()
            if 100 <= epoch < 110:
                feat_concat = torch.cat((feat_2.cuda(), encoder_feat_t.cuda()), 0)
                environment_concat = discriminator(feat_concat.detach())
                label_t = label_t.to(label_2.dtype)
                label_concat = torch.cat((torch.ones_like(label_2), torch.zeros_like(label_t)), 0).cuda()
                d_loss = criterion2(environment_concat, label_concat)
                optimizer4.zero_grad()
                d_loss.backward()
                optimizer4.step()    
                                
            environment_output_t = discriminator(encoder_feat_t)
            target_discriminator_loss = criterion2(environment_output_t, torch.ones_like(label_t).cuda())
            tar_class_1 = classifier_for_source2(encoder_feat_t)
            tar_class_2 = classifier_for_source1(encoder_feat_t)
            # mmd_loss=MMDLoss(feat_1,feat_2)
            #print("----------------------------12---------------")
            tar_L1_loss = Smooth_L1_loss(tar_class_1, tar_class_2)
            tri_loss = triplet(encoder_feat, label_s)

            arg_c1 = torch.argmax(tar_class_1, dim=1)
            arg_c2 = torch.argmax(tar_class_2, dim=1)
            arg_idx = []
            fake_id = []

         
            for i_dx, data in enumerate(arg_c1):
                if (data == arg_c2[i_dx]) and (((tar_class_1[i_dx][data] + tar_class_2[i_dx][arg_c2[i_dx]]) / 2) > 0.8):
                    arg_idx.append(i_dx)
                    fake_id.append(data)
            if 90 <= epoch < 110:
                if arg_idx != []:
                    loss_fake = xent(encoder_class_t[arg_idx], torch.tensor(fake_id))
                    # total_loss = class_loss_1 + class_loss_2 + 0.5 * tar_L1_loss + tri_loss[0] +0.2*mmd_loss
                    total_loss = class_loss_1 + class_loss_2 + 0.4 * tar_L1_loss + tri_loss[
                        0] + 0.2 * target_discriminator_loss
                else:
                    loss_fake = torch.tensor([0])
                    total_loss = class_loss_1 + class_loss_2 + 0.4 * tar_L1_loss + tri_loss[
                        0] + 0.2 * target_discriminator_loss
            if 110 <= epoch:
                if arg_idx != []:
                    loss_fake = xent(encoder_class_t[arg_idx], torch.tensor(fake_id))
                    total_loss = class_loss_1 + class_loss_2 + 0.08 * loss_fake + tri_loss[
                        0] + 0.4 * tar_L1_loss + 0.2 * target_discriminator_loss
                else:
                    loss_fake = torch.tensor([0])
                    total_loss = class_loss_1 + class_loss_2 + tri_loss[
                        0] + 0.4 * tar_L1_loss + 0.2 * target_discriminator_loss
          
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'total_loss: {:.3f}  class_loss_1: {:.3f}  class_loss_2: {:.3f}  tar_L1_loss: {:.3f}  tri_loss: {:.3f}  loss_fake:  {:.6f}'
                      .format(epoch, i + 1, len(train_loader),
                              total_loss.item(), class_loss_1.item(), class_loss_2.item(), tar_L1_loss.item(),
                              tri_loss[0].item(), loss_fake.item()))

    torch.save(encoder_for_target, './model/dataset426&3-428/encoder_for_target.pth')
    torch.save(classifier_for_target, "./model/dataset426&3-428/classifier_for_target.pth")
    return
 

def main():
    root = "/opt/cross_domain/dataset/dataset426&428-3-3/"
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['dataset_for_widar', 'dataset_for_csi'])
    parser.add_argument('--model', choices=['cnn_model', 'cnn_gru_model','cnn_gru','ResNet18','ResNet50','ResNet101'])
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--num_class', type=int, default=5, help='Number of class.')
    parser.add_argument('--latent_dim', type=int, default=180, help='Number of latent_dim.')
    args = parser.parse_args()

    train_loader, unlabel_train_loader, test_loader, encoder_for_target, classifier_for_target, classifier_for_source1, \
    classifier_for_source2, discriminator = load_data_n_model(args.num_class, args.latent_dim,
                                                              'dataset_for_csi', 'cnn_gru', root)
    print("---------------data load-------------------")
    device = torch.device("cuda")

    MMDLoss = loss_func.MMDLoss()
    xent = loss_func.CrossEntropyLabelSmooth(args.num_class)
    triplet = loss_func.TripletLoss()
    Smooth_L1_loss = nn.SmoothL1Loss(reduction='mean')
    criterion2 = nn.CrossEntropyLoss()
    xent = xent.to('cuda')

    encoder_for_target.to('cuda')
    classifier_for_target.to('cuda')
    classifier_for_source1.to('cuda')
    classifier_for_source2.to('cuda')
    discriminator.to('cuda')


    rand_sdomain_1 = np.asarray([0,2])
    rand_sdomain_2 = np.asarray([1,2])


    for epoch in range(0, 30):
        encoder_for_target.train()
        classifier_for_target.train()
        train(train_loader, unlabel_train_loader, epoch, xent, triplet, Smooth_L1_loss, criterion2, encoder_for_target,
              classifier_for_target, classifier_for_source1, classifier_for_source2, discriminator, rand_sdomain_1,
              rand_sdomain_2, device)
    test1( test_loader, xent, device)
    for epoch in range(30, 90):

        encoder_for_target.train()
        classifier_for_source1.train()
        classifier_for_source2.train()

        train(train_loader, unlabel_train_loader, epoch, xent, triplet, Smooth_L1_loss, criterion2, encoder_for_target,
              classifier_for_target, classifier_for_source1, classifier_for_source2, discriminator, rand_sdomain_1,
              rand_sdomain_2, device)

    for epoch in range(90, 120):
        encoder_for_target.train()
        discriminator.train()
        train(train_loader, unlabel_train_loader, epoch, xent, triplet, Smooth_L1_loss, criterion2, encoder_for_target,
              classifier_for_target, classifier_for_source1, classifier_for_source2, discriminator, rand_sdomain_1,
              rand_sdomain_2, device)

    test( test_loader, xent, device)


def test(test_loader, xent, device):
    encoder_for_target = torch.load('./model/dataset426&3-428/encoder_for_target.pth')
    classifier_for_target = torch.load('./model/dataset426&3-428/classifier_for_target.pth')
    encoder_for_target.eval()
    classifier_for_target.eval()
    test_acc = 0
    test_loss = 0
    prob_all = []
    label_all = []
    conf_matrix = torch.zeros(5, 5)
    for data in test_loader:
        inputs, labels, domains = data

        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        encoder_output = encoder_for_target(inputs, 'no')
        class_output = classifier_for_target(encoder_output)
        loss1 = xent(class_output, labels)
        predict_y = torch.argmax(class_output, dim=1).to(device)

        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss1.item() * inputs.size(0)

        prob_all.extend(predict_y.cpu().numpy()) 
        label_all.extend(labels.cpu().numpy())
        conf_matrix = confusion_matrix1(predict_y, labels, conf_matrix=conf_matrix)

    test_acc = test_acc / len(test_loader)
    test_loss = test_loss / len(test_loader.dataset)
    print("test accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
    print("F1-Score:{:.4f}".format(f1_score(label_all, prob_all, average='macro')))
    attack_types = ['A', 'B', 'C', 'D', 'E']
    plot_confusion_matrix(conf_matrix.numpy(), labels_name=attack_types, title='Confusion matrix')
    return

def test1(test_loader, xent, device):
    encoder_for_target = torch.load('./model/dataset426&3-428/encoder_for_target.pth')
    classifier_for_target = torch.load('./model/dataset426&3-428/classifier_for_target.pth')
    encoder_for_target.eval()
    classifier_for_target.eval()
    test_acc = 0
    test_loss = 0
    prob_all = []
    label_all = []
    conf_matrix = torch.zeros(5, 5)
    for data in test_loader:
        inputs, labels, domains = data

        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        encoder_output = encoder_for_target(inputs, 'no')
        class_output = classifier_for_target(encoder_output)
        loss1 = xent(class_output, labels)
        predict_y = torch.argmax(class_output, dim=1).to(device)

        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss1.item() * inputs.size(0)

        prob_all.extend(predict_y.cpu().numpy()) 
        label_all.extend(labels.cpu().numpy())
        conf_matrix = confusion_matrix1(predict_y, labels, conf_matrix=conf_matrix)

    test_acc = test_acc / len(test_loader)
    test_loss = test_loss / len(test_loader.dataset)
    print("test accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
    print("F1-Score:{:.4f}".format(f1_score(label_all, prob_all, average='macro')))
    return



def confusion_matrix1(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_confusion_matrix(cm, labels_name, title, colorbar=False, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap) 
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)
    plt.yticks(num_local, labels_name)
    plt.title(title) 
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('test1.jpg', dpi=300)


if __name__ == "__main__":
    main()