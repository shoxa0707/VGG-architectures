import tensorflow.keras as keras

class VGG16:
    def __init__(self, include_top=True):
        # Pooling
        self.maxpool = keras.layers.MaxPooling2D(2)
        # Convolution part
        # BLOCK1
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        # BLOCK2
        self.conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.conv4 = keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
        # BLOCK3
        self.conv5 = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.conv6 = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.conv7 = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')
        # BLOCK4
        self.conv8 = keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')
        self.conv9 = keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')
        self.conv10 = keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')
        # BLOCK5
        self.conv11 = keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')
        self.conv12 = keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')
        self.conv13 = keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')
        # Fully Connected part
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(4096, activation='relu')
        self.fc2 = keras.layers.Dense(4096, activation='relu')
        self.out = keras.layers.Dense(1000, activation='softmax')
        # Take fully connected layer, if False, FC layer is taken, else FC layer isn't taken
        self.include_top = include_top

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxpool(x)

        if self.include_top:
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.out(x)

        return x
    
    def get_model(self):
        x = keras.layers.Input(shape=(224, 224, 3))
        return keras.models.Model(inputs=[x], outputs=self.__call__(x))
