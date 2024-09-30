# A utility function that plots the training loss and validation loss from
# a Keras history object.

import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import seaborn as sns


def plot(history, fig_name):
    plt.clf()
    plt.plot(history.history['loss'], label='Training set',
             color='blue', linestyle='-')
    plt.plot(history.history['val_loss'], label='Validation set',
             color='green', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(0, len(history.history['loss']))
    plt.legend()
    plt.savefig(f'/home/aman/Desktop/Learn/DeepLearning/Echidna/plots/{fig_name}')



def plot_histories(histories, fig_name):
    colors = plt.cm.get_cmap('tab20', 2*len(histories.keys()))  # Choose a colormap with a large number of colors
    # markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', '+', 'x', '|', '_'] 
    plt.clf()
    for i, (model_name, history) in enumerate(histories.items()):
        color_tr, color_val=colors(i), colors(i+len(histories.keys()))  # Get color from the colormap
        plt.plot(history.history['loss'], label=f'{model_name}_Training set',
                color=color_tr, linestyle='-')
        plt.plot(history.history['val_loss'], label=f'{model_name}_Validation set',
                color=color_val, linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(0, len(history.history['loss']))
    plt.legend()
    plt.savefig(f'/home/aman/Desktop/Learn/DeepLearning/Echidna/plots/{fig_name}')