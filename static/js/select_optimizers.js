// Select Optimizer
$(document).ready(function () {
  $("#optimizerDropdown").change(function () {
    var selectedOptimizer = $(this).val();
    if (selectedOptimizer === "dopamine") {
      // Trigger an AJAX request to your Flask endpoint
      $.ajax({
        type: "POST",
        url: "/optimize",
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
