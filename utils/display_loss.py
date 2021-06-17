import numpy as np
import matplotlib.pyplot as plt


def display_loss(loss_disc, loss_gen, loss_enc,interval):
    epoch = [k*interval for k in range(len(loss_gen))]

    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.ylim([0, 12000])
    plt.plot(epoch, loss_gen)
    plt.plot(epoch, loss_disc)
    plt.plot(epoch, loss_enc)
    plt.legend(['generator', 'discriminator', 'encoder'], loc='upper left')
    plt.show()

def display_loss_norm(loss,interval):
    epoch = [k*interval for k in range(len(loss))]

    plt.ylabel('valid_loss')
    plt.xlabel('epoch')
    #plt.ylim([0, 12000])
    plt.plot(epoch, loss)
    #plt.legend(['generator', 'discriminator'], loc='upper left')
    plt.show()
