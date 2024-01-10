// Load RNNModel
$(document).ready(function () {
  $("#modelDropdown").change(function () {
    var selectedModel = $(this).val();
    if (selectedModel === "custom_cnn") {
      // Trigger an AJAX request to your Flask endpoint
      $.ajax({
        type: "POST",
        url: "/custom_cnn",
        data: { model_type: "custom_cnn" },
        success: function (response) {
          // Handle the response from the server if needed
          console.log(response);
        },
        error: function (error) {
          console.error("Fehler: ", error);
        },
      });
    }
  });
});
