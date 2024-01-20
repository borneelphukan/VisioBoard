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

function fetchTrainingPlot() {
  fetch("/get_tplot")
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
      document.querySelector("#trainingPlot img").src = imageUrl;
    })
    .catch((error) => console.error("Error fetching image:", error));
}
