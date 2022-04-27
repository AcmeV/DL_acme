import models

def config(args):
    '''
        Generate model package dict
    :param
        args: system param
    :return:
        a dict from model name to model package
    '''
    model_type = args.model_type
    if model_type == 'cnn':
        models = ['CNN', 'NN', 'AlexNet', 'NiN', 'VGG',
             'GoogLeNet', 'ResNet', 'SEResNet']

        datasets = ['MNIST', 'CIFAR10', 'TinyImageNet']
    else:
        models = ['MyRNN', 'RNN']
        datasets = ['TimeMachine', 'FunctionValue']

    resnet_versions = [18, 34, 50, 101, 152]
    vgg_versions = [11, 13, 16, 19]

    config_dict = {}

    model_package = f'{args.model_package}.{model_type}_models'

    models_absolute = []
    relations = {}

    for model in models:
        if model == 'ResNet' or model == 'SEResNet':
            for res_v in resnet_versions:
                models_absolute.append(f'{model}{res_v}')
                relations[f'{model}{res_v}'] = 'ResNet'
        elif model == 'VGG':
            for vgg_v in vgg_versions:
                models_absolute.append(f'{model}{vgg_v}')
                relations[f'{model}{vgg_v}'] = model
        else:
            models_absolute.append(f'{model}')
            relations[model] = model

    for model in models_absolute:
        tmp_dict = {}
        config_dict[model] = tmp_dict

        for dataset in datasets:
            path = f'{model_package}.{relations[model]}'
            func = f'{path}.{model}_{dataset}'
            tmp_dict[dataset] = func
    return config_dict
