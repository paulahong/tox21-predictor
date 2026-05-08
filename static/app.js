// app.js – Handles UI interactions and API communication

// Utility to show/hide elements
function toggleVisibility(id, show) {
    const el = document.getElementById(id);
    if (show) {
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

// Show loading spinner
function showLoading() {
    toggleVisibility('loading', true);
    toggleVisibility('results', false);
    toggleVisibility('error', false);
}

// Hide loading spinner
function hideLoading() {
    toggleVisibility('loading', false);
}

// Display error message
function showError(message) {
    const errEl = document.getElementById('error');
    errEl.textContent = message;
    toggleVisibility('error', true);
}

// Update result UI
function displayResult(toxicity, probability) {
    const labelEl = document.getElementById('toxicity-label');
    const barEl = document.getElementById('probability-bar');
    const probText = document.getElementById('probability-text');

    // Set label text and style
    labelEl.textContent = toxicity;
    if (toxicity === 'Toxic') {
        labelEl.classList.remove('nontoxic');
        labelEl.classList.add('toxic');
    } else {
        labelEl.classList.remove('toxic');
        labelEl.classList.add('nontoxic');
    }

    // Update probability bar (percentage)
    const percent = (probability * 100).toFixed(1);
    barEl.style.width = `${percent}%`;
    probText.textContent = `Probability of toxicity: ${percent}%`;
    toggleVisibility('results', true);
}

// Main prediction function
async function predictToxicity() {
    const smilesInput = document.getElementById('smiles-input');
    const smiles = smilesInput.value.trim();

    if (!smiles) {
        showError('Please enter a SMILES string.');
        return;
    }

    showLoading();

    try {
        const response = await fetch('/api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles })
        });

        if (!response.ok) {
            const errMsg = await response.text();
            throw new Error(errMsg || 'Server returned an error.');
        }

        const data = await response.json();
        const toxicity = data.toxicity;
        const probability = data.probability; // value between 0 and 1

        displayResult(toxicity, probability);
    } catch (err) {
        console.error('Prediction error:', err);
        showError(err.message || 'An unexpected error occurred.');
    } finally {
        hideLoading();
    }
}
