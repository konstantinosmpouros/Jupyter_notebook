# Import libraries
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
import numpy as np

sns.set_style('whitegrid')

def walk_through_dir(dir_path):
    """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdirectories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """

    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def plot_loss_accuracy_curves(history):
    """
  Returns separate loss and accuracy curves for training and validation metrics.

  Args:
    history: TensorFlow model History object
  """

    # Assuming 'history' is your training history object
    epochs = range(len(history.history['loss']))
    data = {
        'Epochs': epochs,
        'Training Accuracy': history.history['accuracy'],
        'Validation Accuracy': history.history['val_accuracy'],
        'Training Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss']
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Melt the DataFrame to long format
    df_melted_acc = pd.melt(df, id_vars=['Epochs'], value_vars=['Training Accuracy', 'Validation Accuracy'],
                            var_name='Type', value_name='Accuracy')
    df_melted_loss = pd.melt(df, id_vars=['Epochs'], value_vars=['Training Loss', 'Validation Loss'], var_name='Type',
                             value_name='Loss')

    # Set up the figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Accuracy
    sns.lineplot(data=df_melted_acc, x='Epochs', y='Accuracy', hue='Type', ax=axs[0])
    axs[0].set_title('Accuracy over Epochs')
    axs[0].set_xticks(epochs)
    axs[0].set(ylim=(0, 1))

    # Plot Loss
    sns.lineplot(data=df_melted_loss, x='Epochs', y='Loss', hue='Type', ax=axs[1])
    axs[1].set_title('Loss over Epochs')
    axs[1].set_xticks(epochs)
    axs[1].set(ylim=(0, df_melted_loss['Loss'].max() + 0.2))

    plt.tight_layout()
    plt.show()


def random_img_show(train_data, class_names):
    fig, axis = plt.subplots(1, 4, figsize=(14, 9))

    counter = 0
    for image, label in train_data.take(4):
        axis[counter].imshow(image.numpy())
        axis[counter].set_title(f'{class_names[label.numpy()]}')
        axis[counter].axis(False)

        counter += 1

    plt.show()


# Make a function for preprocessing images
def preprocess_img(image, label, img_shape=227):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape])  # reshape to img_shape
    return tf.cast(image, tf.float32), label  # return (float32_image, label) tuple


def check_models_have_close_weights(model1, model2):
    # Check if both models have the same number of layers
    if len(model1.layers) != len(model2.layers):
        print("Models have different number of layers")
    else:
        for layer1, layer2 in zip(model1.layers, model2.layers):
            weights1 = layer1.get_weights()
            weights2 = layer2.get_weights()
            
            # Check if the layers have the same number of weight matrices
            if len(weights1) != len(weights2):
                print("Layers have different number of weight matrices")
            
            # Compare each weight matrix
            flag = True
            for w1, w2 in zip(weights1, weights2):
                if not np.allclose(w1, w2, atol=1e-7):
                    print("Weights in a layer are not equal")
                    flag = False

        if flag:
            print("Both models have the same weights in all layers")
