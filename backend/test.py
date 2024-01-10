# backend/test.py
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from models.cnn_1 import cnn_1

def test_cnn_1():
    # Load MNIST test dataset
    (_, _), (x_test, y_test) = mnist.load_data()

    # Preprocess the test data
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    y_test = to_categorical(y_test)

    # Load the CNN model architecture
    model = cnn_1()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the trained weights
    weights_path = 'backend/weights/cnn_1_weights.h5'
    model.load_weights(weights_path)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    test_cnn_1()
