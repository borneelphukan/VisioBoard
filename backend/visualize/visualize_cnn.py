import matplotlib.pyplot as plt

def training_accuracy(history):
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('static/images/visualization/training_accuracy.png')

def training_loss(history):
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('static/images/visualization/training_loss.png')

def test_accuracy(accuracy):
    plt.figure(figsize=(6, 6))
    plt.bar(['Test'], [accuracy], color=['blue'])
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])  # Set y-axis limits to match the scale of accuracy (0 to 1)
    plt.savefig('static/images/visualization/test_accuracy.png')

def test_loss(loss):
    plt.figure(figsize=(6, 6))
    plt.bar(['Test'], [loss], color=['red'])
    plt.title('Test Loss')
    plt.ylabel('Loss')
    plt.ylim([0, 1])  # Set y-axis limits to match the scale of accuracy (0 to 1)
    plt.savefig('static/images/visualization/test_loss.png')
