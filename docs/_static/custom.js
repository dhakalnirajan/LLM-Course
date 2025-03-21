// Example: Add a "Copy to Clipboard" button to code blocks
document.addEventListener ('DOMContentLoaded', function () {
  var codeBlocks = document.querySelectorAll ('div.highlight pre');

  codeBlocks.forEach (function (codeBlock) {
    var copyButton = document.createElement ('button');
    copyButton.className = 'copy-button';
    copyButton.textContent = 'Copy';
    copyButton.style.float = 'right'; // Float the button to the right
    copyButton.style.marginBottom = '5px'; // Add some spacing.

    codeBlock.parentNode.insertBefore (copyButton, codeBlock);

    copyButton.addEventListener ('click', function () {
      var code = codeBlock.innerText;
      navigator.clipboard.writeText (code).then (
        function () {
          copyButton.textContent = 'Copied!';
          setTimeout (function () {
            copyButton.textContent = 'Copy';
          }, 2000); // Reset after 2 seconds
        },
        function (err) {
          console.error ('Could not copy text: ', err);
        }
      );
    });
  });
});
