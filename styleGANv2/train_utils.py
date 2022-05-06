from torch.utils import data


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def requires_grad_multiple(models, flag=True):
    for model in models:
        if model is not None:
            requires_grad(model, flag)


# Utility to calculate moving average of generator weights
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch