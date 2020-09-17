import torch
import numpy as np
import timeit
from random import sample
from sklearn.metrics import pairwise_distances


def spearman(a, b):

    order_penalty = (a - b) ** 2
    weight = a * b

    distance = 1e4 * np.sum(order_penalty * weight)
    return distance


def spearman_pairwise_distance(X, centers=None):

    if (centers is not None):
        return torch_spearman_pairwise_distance(X, centers)

    else:
        return torch_spearman_pairwise_distance(X, X)


def torch_spearman_pairwise_distance(X, centers):

    max_calculation_size = 15000000
    calculation_size = len(X) * len(centers)
    size_ratio = max_calculation_size / calculation_size

    if (size_ratio >= 1):
        return torch_spearman_pairwise_distance_core(X, centers)

    else:
        print("size_ratio > 1")

        nb_center_per_step = int(size_ratio * len(centers))
        if (nb_center_per_step <= 0):
            nb_center_per_step = 1

        print("nb_center_per_step: " + str(nb_center_per_step))
        all_column_distance = []
        for i in range(0, len(centers), nb_center_per_step):
            all_column_distance += [torch_spearman_pairwise_distance_core(X, centers[i:i+nb_center_per_step])]

        return np.concatenate(all_column_distance, axis=1)


def torch_spearman_pairwise_distance_core(X, centers):

    nb_data = X.shape[0]
    nb_cluster = centers.shape[0]
    vectSize = centers.shape[1]

    torch_x = torch.from_numpy(X).cuda()
    torch_x = torch_x.unsqueeze(1)
    torch_x = torch_x.expand(nb_data, nb_cluster, vectSize)

    torch_centers = torch.from_numpy(centers).cuda()
    torch_centers = torch_centers.expand(nb_data, nb_cluster, vectSize)

    order_penalty = (torch_x - torch_centers) ** 2
    weight = torch_x * torch_centers

    distance = 1e4 * torch.sum(order_penalty * weight, dim=2)

    return distance.data.cpu().numpy()


if __name__ == '__main__':

    nb_row = 341
    nb_feature = 20
    nb_cluster = 11
    nb_run = 1

    a = np.random.uniform(0, 10, (nb_row,nb_feature))

    cluster_index = sample(list(range(nb_row)), nb_cluster)
    b = a[cluster_index, :]


    print("\n\nSK Dist")

    sk_dist = None
    all_time = []
    for i in range(nb_run):

        start = timeit.default_timer()

        sk_dist = pairwise_distances(a, b, metric=spearman, n_jobs=12)

        stop = timeit.default_timer()
        time_in_second = stop - start
        all_time += [time_in_second]

    print(sk_dist)
    print(sk_dist.shape)
    print(np.mean(all_time))
    print(np.sum(all_time))
    


    print("\nTorch Dist")

    torch_dist = None
    all_time = []
    for i in range(nb_run):
        start = timeit.default_timer()

        torch_dist = spearman_pairwise_distance(a, b)

        stop = timeit.default_timer()
        time_in_second = stop - start
        all_time += [time_in_second]

    print(torch_dist)
    print(torch_dist.shape)
    print(np.mean(all_time))
    print(np.sum(all_time))

    print(np.allclose(sk_dist, torch_dist, atol=0.1))


    print("\nTorch Dist 2")

    torch_dist = None
    all_time = []
    for i in range(nb_run):
        start = timeit.default_timer()

        torch_dist = spearman_pairwise_distance(a)

        stop = timeit.default_timer()
        time_in_second = stop - start
        all_time += [time_in_second]

    print(torch_dist)
    print(torch_dist.shape)
    print(np.mean(all_time))
    print(np.sum(all_time))


    print("\n\nSK Dist 2")

    sk_dist = None
    all_time = []
    for i in range(nb_run):
        start = timeit.default_timer()

        sk_dist = pairwise_distances(a, metric=spearman, n_jobs=12)

        stop = timeit.default_timer()
        time_in_second = stop - start
        all_time += [time_in_second]

    print(sk_dist)
    print(sk_dist.shape)
    print(np.mean(all_time))
    print(np.sum(all_time))

    print(np.allclose(sk_dist, torch_dist, atol=0.1))


