// Grab elements
const form = document.getElementById("uploadForm");
const loader = document.getElementById("loader");
const resultSection = document.getElementById("resultSection");
const errorSection = document.getElementById("errorSection");

// Handle form submit
form.addEventListener("submit", async (e) => {
  e.preventDefault(); // Prevent page reload

  // UI: show loader, hide results & error
  loader.classList.remove("d-none");
  resultSection.classList.add("d-none");
  errorSection.classList.add("d-none");

  // Get the file
  const formData = new FormData(form);

  try {
    // Send file to backend
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    loader.classList.add("d-none");

    if (response.ok && data.cdr !== null) {
      // ✅ Set result values
      document.getElementById("cdrVal").textContent = data.cdr;
      document.getElementById("diagnosis").textContent = data.diagnosis;
      document.getElementById("risk").textContent = data.risk ?? "N/A";

      // ✅ Style tag based on risk
      const tag = document.getElementById("tagLabel");
      tag.classList.remove("bg-success", "bg-warning", "bg-danger");

      if (data.risk === 0) {
        tag.textContent = "🟢 Healthy";
        tag.classList.add("bg-success", "text-white");
      } else if (data.risk < 100) {
        tag.textContent = "🟡 Risky";
        tag.classList.add("bg-warning", "text-dark");
      } else {
        tag.textContent = "🔴 Glaucoma Detected";
        tag.classList.add("bg-danger", "text-white");
      }

      resultSection.classList.remove("d-none");
    } else {
      // ❌ Handle error
      throw new Error(data.error || data.diagnosis || "Unknown error");
    }
  } catch (err) {
    loader.classList.add("d-none");
    document.getElementById("errorMsg").textContent = err.message;
    errorSection.classList.remove("d-none");
  }
});
