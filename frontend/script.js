async function sendMessage() {
  const input = document.getElementById("userInput");
  const message = input.value.trim();
  if (!message) return;

  // Display user message
  const chatbox = document.getElementById("chatbox");
  chatbox.innerHTML += `<p><b>You:</b> ${message}</p>`;

  // Send to backend
  const response = await fetch("http://127.0.0.1:8000/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ message })
  });
  const data = await response.json();

  // Display bot reply
  chatbox.innerHTML += `<p><b>Bot:</b> ${data.reply || data.error}</p>`;
  input.value = "";
}