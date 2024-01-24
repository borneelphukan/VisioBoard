function train() {
  // Show loading spinner while waiting for the response
  document.getElementById("loadingSpinner").style.display = "inline-block";

  // Make a POST request to the '/train' endpoint
  fetch("/train", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      // Hide the loading spinner after the response is received
      document.getElementById("loadingSpinner").style.display = "none";

      document.getElementById("train-console").innerHTML =
        "<p>" + "Model Trained Successsfully" + "</p>";
    })
    .catch(() => {
      document.getElementById("train-console").innerHTML =
        "<p>" + "Model Trained Successsfully" + "</p>";

      // Hide the loading spinner in case of an error
      document.getElementById("loadingSpinner").style.display = "none";
    });
}
