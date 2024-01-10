from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from models.custom_cnn import cnn_model


def cnn_train(model, train_images, train_labels):
    history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    print(f'Training accuracy: {train_acc[-1]}')
    print(f'Validation accuracy: {val_acc[-1]}')

    return model

if __name__ == "__main__":
    model = cnn_model()
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    trained_model = cnn_train(model, train_images, train_labels)