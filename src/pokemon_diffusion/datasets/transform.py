from torchvision import transforms

def get_normalize_transform(normalization_range):
    if normalization_range == "minus_one_to_one":
        return transforms.Normalize([0.5] * 3, [0.5] * 3)
    if normalization_range == "zero_to_one":
        return transforms.Lambda(lambda x: x) # ToTensor already scales to [0, 1]
    raise ValueError(f"Unknown normalization_range: {normalization_range}")

def get_train_transform(img_size=96, normalization_range="minus_one_to_one"):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255,),
        transforms.ToTensor(),
        get_normalize_transform(normalization_range),
    ])

def get_eval_transform(img_size=96, normalization_range="minus_one_to_one"):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        get_normalize_transform(normalization_range),
    ])