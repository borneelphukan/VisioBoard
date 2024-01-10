// static/js/util.js
document.addEventListener("DOMContentLoaded", function () {
  // Get references to the dropdown elements
  var modelDropdown = document.getElementById("modelDropdown");
  var datasetDropdown = document.getElementById("datasetDropdown");
  var optimizerDropdown = document.getElementById("optimizerDropdown");

  // Initial state: Disable datasetDropdown and optimizerDropdown
  datasetDropdown.disabled = true;
  optimizerDropdown.disabled = true;

  // Add event listener to modelDropdown
  modelDropdown.addEventListener("change", function () {
    // Enable datasetDropdown and optimizerDropdown only if a model is selected
    if (modelDropdown.value !== "Select Model") {
      datasetDropdown.disabled = false;
      optimizerDropdown.disabled = false;
    } else {
      // If "Select Model" is chosen, disable datasetDropdown and optimizerDropdown
      datasetDropdown.disabled = true;
      optimizerDropdown.disabled = true;
    }
  });
});
