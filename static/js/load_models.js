function loadSelectedModel() {
  var selectedModel = document.getElementById("modelDropdown").value;
  if (selectedModel !== "Select Model") {
    fetch("/load_model", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: "selected_model=" + selectedModel,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        // Display the response message in the console
      })
      .catch((error) => console.error("Error:", error));
  } else {
    console.error("Please select a model");
  }
}
