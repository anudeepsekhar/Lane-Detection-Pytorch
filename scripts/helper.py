import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np
import torch.nn.functional as F

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

def plot_label_mask(model, images, labels, grayscale):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction.
    '''
    ins_pred_= model(images)
    ins_pred = ins_pred_.cpu().data.numpy()
    # ins_pred = ins_pred[0]
    ins_pred = np.concatenate(ins_pred)
    images_ = images.cpu()
    # images_ = images_[0]
    grid_img = torchvision.utils.make_grid(images_, nrow=1)
    images_ = torch.squeeze(images_)
    if not grayscale:
        images_ = images_.permute(1, 2, 0)
    labels_ = labels.cpu().data.numpy()
    labels_ = np.squeeze(labels_)
    
    # plot the images in the batch, along with predicted and true labels
    fig, ax = plt.subplots(1, 5, figsize=(48,12))

    ax[0].imshow(images_.numpy()/255)
    ax[1].imshow(labels_[0])
    ax[2].imshow(ins_pred[0])
    ax[3].imshow(labels_[1])
    ax[4].imshow(ins_pred[1])

    return fig

    def plot_label_mask2(model, images, labels, grayscale):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction.
        '''
        ins_pred_= model(images)
        ins_pred = ins_pred_.cpu().data.numpy()
        ins_pred = np.concatenate(ins_pred)
        images_ = images.cpu()
        images_ = images_[0]
        grid_img = torchvision.utils.make_grid(images_, nrow=1)
        images_ = torch.squeeze(images_)
        if not grayscale:
            images_ = images_.permute(1, 2, 0)
        labels_ = labels.cpu().data.numpy()
        labels_ = np.squeeze(labels_)
        
        # plot the images in the batch, along with predicted and true labels
        fig, ax = plt.subplots(1, 5, figsize=(48,12))

        ax[0].imshow(images_.numpy()/255)
        ax[1].imshow(labels_[0])
        ax[2].imshow(ins_pred[0])
        ax[3].imshow(labels_[1])
        ax[4].imshow(ins_pred[1])

    return fig

def plot_tu_data(images, labels, predicts):
    images = images.cpu()
    labels = labels.cpu()
    predicts = predicts.cpu().detach()
    predicts = F.sigmoid(predicts)
    image = torch.squeeze(images[0])
    image = image.permute(1, 2, 0)
    label = torch.squeeze(labels[0])
    predict = torch.squeeze(predicts[0])
    fig = plt.figure(figsize=(30,10))
    plt.subplot(1,5,1)
    plt.imshow(image)
    plt.subplot(1,5,2)
    plt.imshow(image)
    plt.imshow(label, cmap='jet', alpha=0.5)
    plt.subplot(1,5,3)
    plt.imshow(image)
    plt.imshow(predict, cmap='jet', alpha=0.5)
    plt.subplot(1,5,4)
    plt.imshow(label)
    plt.subplot(1,5,5)
    plt.imshow(predict)
    return fig


def plot_tu_data_2(images, labels, predicts):
    images = images.cpu()
    labels = labels.cpu()
    predicts = predicts.cpu().detach()
    predicts = F.sigmoid(predicts)
    image = torch.squeeze(images[0])
    image = image.permute(1, 2, 0)
    label1 = labels[0][0]
    label2 = labels[0][1]
    predict1 = predicts[0][0]
    predict2 = predicts[0][1]
    fig = plt.figure(figsize=(30,10))
    plt.subplot(1,5,1)
    plt.imshow(image)
    # plt.subplot(1,7,2)
    # plt.imshow(image)
    # plt.imshow(label, cmap='jet', alpha=0.5)
    # plt.subplot(1,7,3)
    # plt.imshow(image)
    # plt.imshow(predict, cmap='jet', alpha=0.5)
    plt.subplot(1,5,2)
    plt.imshow(label1)
    plt.subplot(1,5,3)
    plt.imshow(predict1)
    plt.subplot(1,5,4)
    plt.imshow(label2)
    plt.subplot(1,5,5)
    plt.imshow(predict2)
    return fig