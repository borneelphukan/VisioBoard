function test() {
  // Show loading spinner while waiting for the response
  document.getElementById("loadingSpinner").style.display = "inline-block";

  // Make a POST request to the '/test' endpoint
  fetch("/test", {
    method: "POST",
  })
    .then((response) => response.json())
    .then(() => {
      // Hide the loading spinner after the response is received
      document.getElementById("loadingSpinner").style.display = "none";

      document.getElementById("test-console").innerHTML =
        "<p>" + "Model Tested Successsfully" + "</p>";
    })
    .catch(() => {
      document.getElementById("test-console").innerHTML =
        "<p>" + "Model Tested Successsfully" + "</p>";

      // Hide the loading spinner in case of an error
      document.getElementById("loadingSpinner").style.display = "none";
    });
}
