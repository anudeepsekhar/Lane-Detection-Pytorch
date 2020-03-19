import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def plot_label_mask(model, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    ins_pred_= model(images)
    ins_pred = ins_pred_.cpu().data.numpy()
    ins_pred = np.concatenate(ins_pred)
    images_ = images.cpu()
    grid_img = torchvision.utils.make_grid(images_, nrow=1)
    images_ = torch.squeeze(images_)
    # print(images_.shape)
    images_ = images_.permute(1, 2, 0)
    print(images_.shape)
    # images_.sh/ape
    # images_ = np.concatenate(images_)[:,0].astype(np.uint8)
    # images_ = np.transpose(images_,(1,2,0))
    labels_ = labels.cpu().data.numpy()
    # labels_ = images.cpu().data.numpy()
    labels_ = np.squeeze(labels_)
    # labels_ = np.squeeze(labels_)

    # labels_ = np.transpose(labels_,(2,1,0))
    # images = np.concatenate(images)[:, 0].astype(np.uint8)
    # preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig, ax = plt.subplots(1, 5, figsize=(48,12))

    # for i, ax_ in enumerate(axes):
    #     # color_img = gen_color_img(sem_pred[i], ins_pred[i], 4)

    ax[0].imshow(images_.numpy())
    # axes2[0][1].imshow(sem_pred[0])
    # print(sem_pred[i].shape)
    # for k in range(1):
    ax[1].imshow(labels_[0])
    ax[2].imshow(ins_pred[0])
    ax[3].imshow(labels_[1])
    ax[4].imshow(ins_pred[1])

    return fig

    # ax[0].imshow(images[0])
    # axes2[1][1].imshow(sem_labels[0])
    # print(sem_pred[i].shape)
    # for k in range(1):
    # fig = plt.figure(figsize=(12, 48))
    #     for idx in np.arange(3):
    #         ax = fig.add_subplot(1, 3, idx+1, xticks=[], yticks=[])
    #         matplotlib_imshow(images[idx], one_channel=True)
    #     return fig