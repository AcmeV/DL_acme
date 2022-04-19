def config(args):
    '''
        Generate model package dict
    :param
        args: system param
    :return:
        a dict from model name to model package
    '''
    models = ['CNN', 'NN', 'AlexNet', 'NiN', 'VGG',
             'GoogLeNet', 'ResNet', 'SEResNet']

    resnet_versions = [18, 34, 50, 101, 152]
    vgg_versions = [11, 13, 16, 19]

    datasets = ['MNIST', 'CIFAR10', 'TinyImageNet']

    config_dict = {}

    model_package = args.model_package

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
