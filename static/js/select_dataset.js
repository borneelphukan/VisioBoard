// Select figure8
$(document).ready(function () {
  $("#datasetDropdown").change(function () {
    var selectedDataset = $(this).val();
    if (selectedDataset === "figure8") {
      // Trigger an AJAX request to your Flask endpoint
      $.ajax({
        type: "POST",
        url: "/load_dataset",
        data: { dataset_type: "figure8" },
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


