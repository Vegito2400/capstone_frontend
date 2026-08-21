// Mock analysis service for local development and testing.
// Exports a named `analyzeImage(file, onProgress)` function that
// simulates progress and returns a fake analysis result.

export async function analyzeImage(file, onProgress = () => {}) {
	const stages = [
		'Uploading',
		'Preprocessing',
		'ELA Analysis',
		'Dual-ViT Analysis',
		'Generating Explainability',
		'Verifying Integrity'
	];

	for (let i = 0; i < stages.length; i++) {
		try { onProgress(i, stages[i]); } catch (e) { /* ignore errors from callback */ }
		// small delay to simulate work
		// eslint-disable-next-line no-await-in-loop
		await new Promise((res) => setTimeout(res, 350));
	}

	const result = {
		prediction: Math.random() > 0.35 ? 'Authentic' : 'Tampered',
		confidence: +(0.85 + Math.random() * 0.14).toFixed(2),
		probabilities: {
			authentic: +(0.7 + Math.random() * 0.25).toFixed(3),
			tampered: +(0.05 + Math.random() * 0.15).toFixed(3),
			ai_generated: +(0.01 + Math.random() * 0.05).toFixed(3),
		},
		verificationId: `MT-VER-${Math.floor(1000 + Math.random() * 9000)}`,
		timestamp: new Date().toISOString(),
		summary: 'Simulated forensic analysis (local mock). Use the real backend integration for production results.'
	};

	return result;
}

