import torchvision.transforms as T


def transform_wout_normalize():

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


def augmentation_transform(mean, std):

    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(20),
        T.Resize((64,64)),
        T.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02
        ),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])


def noise_transform_train():

    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(20),
        T.Resize((64,64)),
        T.ToTensor(),
    ])


def masked_jitter(img_tensor, mask):

    jitter = T.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02
    )

    img_jitter = jitter(img_tensor)

    img_final = (img_jitter * mask + img_tensor * (1 - mask))

    return img_final
