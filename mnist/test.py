from tensorflow.keras.datasets import mnist
from mnist.train import cnn_train
from mnist.models.custom_cnn import cnn_model
from tensorflow.keras.utils import to_categorical

def cnn_test(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

if __name__ == "__main__":
    model = cnn_model()
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    trained_model = cnn_train(model, train_images, train_labels)
    cnn_test(trained_model, test_images, test_labels)
