from tensorflow import keras

class CNN:
  """
  CNN architecture
  """

  def prepare_single_CNN_layer(self, input_shape, n_classes):
    """CNN - FC(10)"""
    # Build the model using the functional API
    # input layer
    i = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(i)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPooling2D((2, 2))(x)

    # x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Flatten()(x)
    # x = keras.layers.Dropout(0.2)(x)

    # Hidden layer
    x = keras.layers.Dense(1024, activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)

    # last hidden layer i.e.. output layer
    x = keras.layers.Dense(n_classes, activation='softmax')(x)

    return keras.Model(i, x)



  