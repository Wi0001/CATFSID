from dataset import *
from new_module_building import *
import torch


def weights_init_normal(m):  # 自定义初始化参数
    classname = m.__class__.__name__  # 获得类名
    if classname.find("Conv") != -1:  # 在类classname中检索到了Conv
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def load_data_n_model(num_class, latent_dim, dataset_name, model_name, root):
    if dataset_name == 'dataset_for_widar':
        print('using dataset: dataset_for_widar')
        data = data_for_dfs(root)
        train_set = torch.utils.data.TensorDataset(data['data\\data-train'], data['label\\label-train'],
                                                   data['domain\\domain-train'])
        unlabel_train_set = torch.utils.data.TensorDataset(data['data\\data-test'], data['label\\label-test'],
                                                           data['domain\\domain-test'])
        test_set = torch.utils.data.TensorDataset(data['data\\data-test'], data['label\\label-test'],
                                                  data['domain\\domain-test'])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
        unlabel_train_loader = torch.utils.data.DataLoader(unlabel_train_set, batch_size=32, shuffle=True,
                                                           drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
        if model_name == 'cnn_model':
            encoder_for_target = target_encoder_model(num_class, latent_dim)
            classifier_for_target = target_classifier(num_class, latent_dim)
            classifier_for_source1 = source1_classifier(num_class, latent_dim)
            classifier_for_source2 = source2_classifier(num_class, latent_dim)
            discriminator = feature_discriminator_for_WiADG(latent_dim)
        elif model_name == 'cnn_gru_model':
            encoder_for_target = CNN_GRU_for_target()
            classifier_for_target = target_classifier_for_CNN_GRU()
            classifier_for_source1 = source_classifier_for_CNN_GRU()
            classifier_for_source2 = source2_classifier_for_CNN_GRU()
            discriminator = feature_discriminator_for_WiADG(latent_dim)

        return train_loader, unlabel_train_loader, test_loader, encoder_for_target, classifier_for_target, classifier_for_source1, \
               classifier_for_source2,discriminator

    elif dataset_name == 'dataset_for_csi':
        print('using dataset: dataset_for_csi')
        data = data_for_csi(root)
        train_set = torch.utils.data.TensorDataset(data['data-train'], data['label-train'],
                                                   data['domain-train'])
        unlabel_train_set = torch.utils.data.TensorDataset(data['data-test'], data['label-test'],
                                                           data['domain-test'])
        test_set = torch.utils.data.TensorDataset(data['data-test'], data['label-test'],
                                                  data['domain-test'])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
        unlabel_train_loader = torch.utils.data.DataLoader(unlabel_train_set, batch_size=64, shuffle=True,
                                                           drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        if model_name == 'cnn_model':
            encoder_for_target = target_encoder_model(num_class, latent_dim)
            classifier_for_target = target_classifier(num_class, latent_dim)
            classifier_for_source1 = source1_classifier(num_class, latent_dim)
            classifier_for_source2 = source2_classifier(num_class, latent_dim)
            discriminator = feature_discriminator_for_WiADG(latent_dim)

        elif model_name == 'cnn_gru':
            encoder_for_target = CNN_GRU_for_target()

            classifier_for_target = target_classifier_for_CNN_GRU()
            classifier_for_source1 = source_classifier_for_CNN_GRU()
            classifier_for_source2 = source2_classifier_for_CNN_GRU()
            discriminator = feature_discriminator_for_WiADG(latent_dim)

        elif model_name == 'ResNet18':
            encoder_for_target = UT_HAR_ResNet18()

            classifier_for_target = target_classifier_for_ResNet18()
            classifier_for_source1 = source_classifier_for_ResNet18()
            classifier_for_source2 = source2_classifier_for_ResNet18()
            discriminator = feature_discriminator_for_WiADG(latent_dim)

        elif model_name == 'ResNet50':
            encoder_for_target = UT_HAR_ResNet50()
            classifier_for_target = target_classifier_for_ResNet18()
            classifier_for_source1 = source_classifier_for_ResNet18()
            classifier_for_source2 = source2_classifier_for_ResNet18()
            discriminator = feature_discriminator_for_WiADG(latent_dim)

        elif model_name == 'ResNet101':
            encoder_for_target = UT_HAR_ResNet101()

            classifier_for_target = target_classifier_for_ResNet18()
            classifier_for_source1 = source_classifier_for_ResNet18()
            classifier_for_source2 = source2_classifier_for_ResNet18()
            discriminator = feature_discriminator_for_WiADG(latent_dim)

        return train_loader, unlabel_train_loader, test_loader, encoder_for_target, classifier_for_target, classifier_for_source1, \
               classifier_for_source2, discriminator


