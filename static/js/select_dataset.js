// Select figure8 or lorenz
$(document).ready(function () {
  $("#datasetDropdown").change(function () {
    var selectedDataset = $(this).val();

    // Trigger an AJAX request to your Flask endpoint based on the selected dataset
    if (selectedDataset === "figure8" || selectedDataset === "lorenz") {
      $.ajax({
        type: "POST",
        url: "/load_dataset",
        data: { dataset_type: selectedDataset },
        success: function (response) {
          // Handle the response from the server if needed
          console.log(response);
        },
        error: function (error) {
          console.error("Error:", error);
        },
      });
    }
  });
});
