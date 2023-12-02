// Load RNNModel
$(document).ready(function () {
  $("#modelDropdown").change(function () {
    var selectedModel = $(this).val();
    if (selectedModel === "rnn_model") {
      // Trigger an AJAX request to your Flask endpoint
      $.ajax({
        type: "POST",
        url: "/rnn_model",
        data: { model_type: "rnn_model" },
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
