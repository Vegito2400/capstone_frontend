// Configure VITE_API_BASE_URL for a deployed API. The local default matches
// the FastAPI server used during development.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api').replace(/\/$/, '');

/** Sends an image to the API and returns only values used by the results UI. */
export async function analyzeImage(file, onProgress = () => {}) {
	if (!file) throw new Error('Select an image before starting an analysis.');

	onProgress(0, 'Uploading image');
	const formData = new FormData();
	formData.append('file', file);

	let response;
	try {
		response = await fetch(`${API_BASE_URL}/analyze`, { method: 'POST', body: formData });
	} catch {
		throw new Error('Unable to reach the analysis service. Please ensure the backend is running.');
	}

	const payload = await response.json().catch(() => ({}));
	if (!response.ok) {
		const detail = typeof payload.detail === 'string' ? payload.detail : null;
		throw new Error(detail || `The analysis service returned HTTP ${response.status}.`);
	}

	const { confidence, probabilities } = payload;
	if (!Number.isFinite(confidence) || !probabilities || !Number.isFinite(probabilities.fake) || !Number.isFinite(probabilities.real)) {
		throw new Error('The analysis service returned an invalid result.');
	}

	onProgress(1, 'Analysis complete');
	return { confidence, probabilities: { fake: probabilities.fake, real: probabilities.real } };
}

