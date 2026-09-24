import torchvision.transforms as T


def stats_transform():

    return T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
        T.ToTensor()
    ])



def base_transform(mean, std):

    return T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
