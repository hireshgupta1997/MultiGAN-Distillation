from torch.utils import data


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# Utility to calculate moving average of generator weights
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def sample_data_test(dataset, batch_size, image_size=4, drop_last=True):
    dataset.resolution = image_size
    loader = data.DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=1, drop_last=drop_last)
    return loader