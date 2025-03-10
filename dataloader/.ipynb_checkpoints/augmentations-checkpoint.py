import numpy as np
import torch


def DataTransform(sample, config):
    # weak_aug = scaling(sample, sigma=config.augmentation.jitter_scale_ratio_weak/10)
    weak_aug = jitter(sample, sigma=config.augmentation.jitter_ratio_weak/10)
    # weak_aug = sample
    # strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    # strong_aug = scaling(sample, sigma=config.augmentation.jitter_scale_ratio_strong/10)
    # strong_aug = permutation(sample, max_segments=config.augmentation.max_seg)
    strong_aug = jitter(sample, sigma=config.augmentation.jitter_ratio_strong/10)
    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    # print(sigma)
    return x + torch.normal(mean=0., std=sigma, size=x.shape).to(x.device)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    x_detached = x.detach()  # 利用了detach()函数，因为aug不需要梯度返回
    x = x_detached.cpu().numpy()
    # factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    # print(sigma)
    # print(factor)
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    x_detached = x.detach()  # 利用了detach()函数，因为aug不需要梯度返回
    x = x_detached.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            idx = np.random.permutation(len(splits))
            warp = np.concatenate([splits[i] for i in idx]).ravel()
            # warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)
