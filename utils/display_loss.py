import numpy as np
import matplotlib.pyplot as plt


def display_loss(loss_disc, loss_gen):
    epoch = np.arange(len(loss_gen))

    plt.ylabel('losses of modules')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.ylim([0, 12000])
    plt.plot(epoch, loss_gen)
    plt.plot(epoch, loss_disc)
    plt.legend(['generator', 'discriminator'], loc='upper left')
    plt.show()
