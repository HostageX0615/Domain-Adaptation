import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from FADA.models import Classifier, Encoder, DCD
from FADA.util import eval_on_test, into_tensor
from FADA.data import test_data_loader, get_test_XY
import random
import matplotlib.pyplot as plt


def model_fn(encoder, classifier):
    return lambda x: classifier(encoder(x))


''' Pre-train the encoder and classifier as in (a) in figure 2. '''


def pretrain(all_data, src_test, epochs=5, batch_size=128, cuda=False):
    global loss
    X_s, y_s, _, _ = all_data
    src_test_X, src_test_Y = get_test_XY(src_test)

    classifier = Classifier(20)
    encoder = Encoder()

    if cuda:
        classifier.cuda()
        encoder.cuda()

    ''' Jointly optimize both encoder and classifier '''
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()))
    loss_fn = nn.MSELoss()

    for e in range(epochs):

        for _ in range(len(X_s) // batch_size):
            inds = torch.randperm(len(X_s))[:batch_size]

            x, y = Variable(X_s[inds]), Variable(y_s[inds])
            optimizer.zero_grad()

            if cuda:
                x, y = x.cuda(), y.cuda()

            y_pred = model_fn(encoder, classifier)(x.float())

            loss = loss_fn(y_pred, y.float())

            loss.backward()

            optimizer.step()

        print("Epoch", e, "Loss", loss.item(), "Test_Loss",
              eval_on_test(src_test_X, src_test_Y, model_fn(encoder, classifier)))

    return encoder, classifier


''' Train the discriminator while the encoder is frozen '''


def train_discriminator(encoder, groups, n_target_samples=2, cuda=False, epochs=20):
    global loss

    discriminator = DCD()  # Takes in concatenated hidden representations
    loss_fn = nn.CrossEntropyLoss()

    # Only train DCD
    optimizer = optim.Adam(discriminator.parameters())

    # Size of group G2, the smallest one, times the amount of groups
    n_iters = 4 * n_target_samples

    if cuda:
        discriminator.cuda()

    print("Training DCD")
    for e in range(epochs):

        for _ in range(n_iters):

            # Sample a pair of samples from a group
            group = random.choice([0, 1, 2, 3])

            x1, x2 = groups[group][random.randint(0, len(groups[group]) - 1)]
            x1, x2 = Variable(x1), Variable(x2)

            if cuda:
                x1, x2 = x1.cuda(), x2.cuda()

            # Optimize the DCD using sample drawn
            optimizer.zero_grad()

            # Concatenate encoded representations
            x_cat = torch.cat([encoder(x1.unsqueeze(0).float()), encoder(x2.unsqueeze(0).float())], 1)
            y_pred = discriminator(x_cat)
            # y_pred = Variable(y_pred.squeeze())

            # Label is the group
            y = Variable(torch.tensor([group]).long())
            if cuda:
                y = y.cuda()

            loss = -loss_fn(y_pred, y)

            loss.backward()

            optimizer.step()

        print("Epoch", e, "Loss", loss.item())

    return discriminator


''' FADA Loss, as given by (4) in the paper. The minus sign is shifted because it seems to be wrong '''


def fada_loss(y_pred_g2, g1_true, y_pred_g4, g3_true, gamma=0.2):
    return -gamma * torch.mean(g1_true * torch.log(y_pred_g2) + g3_true * torch.log(y_pred_g4))


''' Step three of the algorithm, train everything except the DCD '''


def train(target_test, encoder, discriminator, classifier, data, groups, n_target_samples=2, cuda=False, epochs=20,
          batch_size=256, plot=False):
    # For evaluation only
    global loss, mses
    test_data = test_data_loader(target_test)
    test_x, test_y = get_test_XY(test_data)

    X_s, Y_s, X_t, Y_t = data

    G1, G2, G3, G4 = groups

    ''' Two optimizers, one for DCD (which is frozen) and one for class training '''
    class_optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()))
    dcd_optimizer = optim.Adam(encoder.parameters())

    loss_fn = nn.MSELoss()
    n_iters = 4 * n_target_samples

    if plot:
        mses = []
    print("Final Training")
    for e in range(epochs):

        # Shuffle data at each epoch
        inds = torch.randperm(X_s.shape[0])
        X_s, Y_s = X_s[inds], Y_s[inds]

        inds = torch.randperm(X_t.shape[0])
        X_t, Y_t = X_t[inds], Y_t[inds]

        g2_one, g2_two = into_tensor(G2, into_vars=True)
        g4_one, g4_two = into_tensor(G4, into_vars=True)

        inds = torch.randperm(g2_one.shape[0])
        if cuda:
            inds = inds.cuda()
        g2_one, g2_two, g4_one, g4_two = g2_one[inds], g2_two[inds], g4_one[inds], g4_two[inds]

        for _ in range(n_iters):

            class_optimizer.zero_grad()
            dcd_optimizer.zero_grad()

            # Evaluate source predictions
            inds = torch.randperm(X_s.shape[0])[:batch_size]
            x_s, y_s = Variable(X_s[inds]), Variable(Y_s[inds])
            if cuda:
                x_s, y_s = x_s.cuda(), y_s.cuda()
            y_pred_s = model_fn(encoder, classifier)(x_s.float())

            # Evaluate target predictions
            ind = random.randint(0, X_t.shape[0] - 1)
            x_t, y_t = Variable(X_t[ind].unsqueeze(0)), Variable(Y_s[ind].clone().detach())
            if cuda:
                x_t, y_t = x_t.cuda(), y_t.cuda()

            y_pred_t = model_fn(encoder, classifier)(x_t.float())

            # Evaluate groups 

            x1, x2 = encoder(g2_one.float()), encoder(g2_two.float())
            y_pred_g2 = discriminator(torch.cat([x1, x2], 1))
            g1_true = 1

            x1, x2 = encoder(g4_one.float()), encoder(g4_two.float())
            y_pred_g4 = discriminator(torch.cat([x1, x2], 1))
            g3_true = 3

            # Evaluate loss
            # This is the full loss given by (5) in the paper
            loss = fada_loss(y_pred_g2, g1_true, y_pred_g4, g3_true) + loss_fn(y_pred_s, y_s.float()) + loss_fn(y_pred_t, y_t.float())

            loss.backward()

            class_optimizer.step()
        mse = eval_on_test(test_x, test_y, model_fn(encoder, classifier))
        print("Epoch", e, "Loss", loss.item(), "Test Loss", mse)

        if plot:
            mses.append(mse)

    if plot:
        plt.plot(range(len(mses)), mses)
        plt.title("Test MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.show()

    mses = mses.sort()
    return mses[0]