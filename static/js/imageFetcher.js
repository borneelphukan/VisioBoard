function fetchDatasetImage() {
  // Make a request to the Flask endpoint
  fetch("/get_image")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.blob();
    })
    .then((blob) => {
      // Create a data URL from the blob
      const imageUrl = URL.createObjectURL(blob);

      // Update the image source
      document.querySelector("#imageContainer img").src = imageUrl;
    })
    .catch((error) => console.error("Error fetching image:", error));
}

function fetchTrainingAccuracy() {
  fetch("/training_accuracy")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.blob();
    })
    .then((blob) => {
      // Create a data URL from the blob
      const imageUrl = URL.createObjectURL(blob);

      // Update the image source
      document.querySelector("#trainingAccuracy img").src = imageUrl;
    })
    .catch((error) => console.error("Error fetching image:", error));
}

function fetchTrainingLoss() {
  fetch("/training_loss")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.blob();
    })
    .then((blob) => {
      // Create a data URL from the blob
      const imageUrl = URL.createObjectURL(blob);

      // Update the image source
      document.querySelector("#trainingLoss img").src = imageUrl;
    })
    .catch((error) => console.error("Error fetching image:", error));
}

function fetchTestAccuracy() {
  fetch("/test_accuracy")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.blob();
    })
    .then((blob) => {
      // Create a data URL from the blob
      const imageUrl = URL.createObjectURL(blob);

      // Update the image source
      document.querySelector("#testAccuracy img").src = imageUrl;
    })
    .catch((error) => console.error("Error fetching image:", error));
}

function fetchTestLoss() {
  fetch("/test_loss")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.blob();
    })
    .then((blob) => {
      // Create a data URL from the blob
      const imageUrl = URL.createObjectURL(blob);

      // Update the image source
      document.querySelector("#testLoss img").src = imageUrl;
    })
    .catch((error) => console.error("Error fetching image:", error));
}
