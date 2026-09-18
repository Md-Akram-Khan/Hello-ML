const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

function mapPrediction(response) {
  const prediction = String(response.prediction || "").toLowerCase();
  const isCat = prediction === "cat" || prediction === "1";
  const rawConfidence = Number(response.confidence);
  const confidence = rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence;

  return {
    label: isCat ? "CAT" : "NOT A CAT",
    isCat,
    confidence: Math.max(0, Math.min(100, confidence)),
  };
}

export async function classifyImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.error || "The image could not be classified.");
  }

  return mapPrediction(payload);
}
