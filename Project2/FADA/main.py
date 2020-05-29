from FADA.training import pretrain, train_discriminator, train
from FADA.data import sample_groups, test_data
import torch

n_target_samples = 100
n_source_samples = 1000
plot = True

if __name__ == '__main__':
    female_train_path = "../Data/FEMALE_train.csv"
    female_test_path = "../Data/FEMALE_test.csv"
    male_train_path = "../Data/MALE_train.csv"
    male_test_path = "../Data/MALE_test.csv"
    mixed_train_path = "../Data/MIXED_train.csv"
    mixed_test_path = "../Data/MIXED_test.csv"

    cuda = False

    source_domain_path_1 = male_train_path
    source_test_path_1 = male_test_path
    source_domain_path_2 = mixed_train_path
    source_test_path_2 = mixed_test_path

    target_domain_path = female_train_path
    target_test = female_test_path

    groups, data = sample_groups(source_domain_path_1, source_domain_path_2, target_domain_path, n_target_samples=n_target_samples, n_source_samples=n_source_samples)

    source_domain_test = test_data(source_test_path_1, source_test_path_2)

    encoder, classifier = pretrain(data, source_domain_test, cuda=cuda, epochs=20)

    discriminator = train_discriminator(encoder, groups, n_target_samples=n_target_samples, epochs=50, cuda=cuda)

    min_mse = train(target_test, encoder, discriminator, classifier, data, groups, n_target_samples=n_target_samples, cuda=cuda, epochs=150, plot=plot)

    print(min_mse)
