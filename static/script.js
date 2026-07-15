// ---- Tab switching ----
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---- Poll live state ----
async function pollState() {
  try {
    const res = await fetch("/get_state");
    const data = await res.json();

    document.getElementById("currentWord").textContent = data.current_word || "—";
    document.getElementById("confidenceText").textContent = `${data.confidence}%`;
    document.getElementById("confidenceFill").style.width = `${data.confidence}%`;
    document.getElementById("sentenceText").textContent = data.sentence || "—";

    document.getElementById("lowLightBadge").classList.toggle("hidden", !data.low_light);

    const uncertain = data.hand_detected && data.confidence < 70;
    document.getElementById("invalidBadge").classList.toggle("hidden", !uncertain);
  } catch (e) {
    // server might be mid-restart; ignore
  }
}
setInterval(pollState, 400);

// ---- Reset sentence ----
document.getElementById("resetBtn").addEventListener("click", async () => {
  await fetch("/reset_sentence", { method: "POST" });
});

// ---- Text to speech ----
async function speak(lang) {
  const text = document.getElementById("sentenceText").textContent.trim();
  if (!text || text === "—") return;
  const res = await fetch("/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, lang }),
  });
  const data = await res.json();
  if (data.audio_url) {
    const player = document.getElementById("audioPlayer");
    player.src = data.audio_url;
    player.classList.remove("hidden");
    player.play();
  } else if (data.error) {
    alert(data.error);
  }
}
document.getElementById("speakEnBtn").addEventListener("click", () => speak("en"));
document.getElementById("speakTaBtn").addEventListener("click", () => speak("ta"));

// ---- Save to history ----
document.getElementById("saveHistoryBtn").addEventListener("click", async () => {
  const sentence = document.getElementById("sentenceText").textContent.trim();
  if (!sentence || sentence === "—") return;
  await fetch("/history", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentence }),
  });
  loadHistory();
});

// ---- History list ----
async function loadHistory() {
  const res = await fetch("/history");
  const items = await res.json();
  const list = document.getElementById("historyList");
  list.innerHTML = "";
  items.forEach(item => {
    const li = document.createElement("li");
    li.innerHTML = `${item.sentence}<span class="ts">${item.timestamp}</span>`;
    list.appendChild(li);
  });
}
document.getElementById("clearHistoryBtn").addEventListener("click", async () => {
  await fetch("/history", { method: "DELETE" });
  loadHistory();
});
loadHistory();

// ---- Video upload ----
document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("videoFile");
  const resultBox = document.getElementById("uploadResult");
  if (!fileInput.files.length) {
    resultBox.textContent = "Please choose a video file first.";
    return;
  }
  resultBox.textContent = "Analyzing video... please wait.";
  const formData = new FormData();
  formData.append("video", fileInput.files[0]);

  const res = await fetch("/upload", { method: "POST", body: formData });
  const data = await res.json();
  if (data.error) {
    resultBox.textContent = `Error: ${data.error}`;
  } else {
    resultBox.textContent = `Recognized sentence: "${data.sentence}" (from ${data.frame_count} analyzed frames)`;
  }
});

// ---- Emergency mode ----
const emergencyOverlay = document.getElementById("emergencyOverlay");
document.getElementById("emergencyBtn").addEventListener("click", async () => {
  await fetch("/toggle_emergency", { method: "POST" });
  emergencyOverlay.classList.remove("hidden");
});
document.getElementById("closeEmergencyBtn").addEventListener("click", async () => {
  await fetch("/toggle_emergency", { method: "POST" });
  emergencyOverlay.classList.add("hidden");
});
document.querySelectorAll(".emergency-phrase").forEach(btn => {
  btn.addEventListener("click", async () => {
    const text = btn.dataset.phrase;
    const res = await fetch("/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang: "en" }),
    });
    const data = await res.json();
    if (data.audio_url) {
      const player = document.getElementById("audioPlayer");
      player.src = data.audio_url;
      player.classList.remove("hidden");
      player.play();
    }
  });
});
