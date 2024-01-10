function loadDataset() {
  var dropdown = document.getElementById("datasetDropdown");
  var selectedDataset = dropdown.options[dropdown.selectedIndex].value;

  if (selectedDataset === "mnist") {
    // Send AJAX request to Flask API
    fetch("/load_mnist", {
      method: "POST",
    })
      .then((response) => response.text())
      .then((data) => {
        // Handle the response as needed
        console.log(data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  } else if (selectedDataset === "fashion-mnist") {
    // Send AJAX request to Flask API
    fetch("/load_fashion_mnist", {
      method: "POST",
    })
      .then((response) => response.text())
      .then((data) => {
        // Handle the response as needed
        console.log(data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }
}
