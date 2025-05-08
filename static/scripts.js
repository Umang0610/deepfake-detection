function previewImage(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('imagePreview');
    const fileInfo = document.getElementById('imageFileInfo');
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';
        fileInfo.textContent = `Selected file: ${file.name}`;
    }
}

function previewAudio(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('audioPreview');
    const fileInfo = document.getElementById('audioFileInfo');
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';
        fileInfo.textContent = `Selected file: ${file.name}`;
    }
}

function previewVideo(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('videoPreview');
    const fileInfo = document.getElementById('videoFileInfo');
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';
        fileInfo.textContent = `Selected file: ${file.name}`;
    }
}

async function predictImage() {
    const input = document.getElementById('imageInput');
    const fileInfo = document.getElementById('imageFileInfo');
    const resultDiv = document.getElementById('imageResult');
    const spinner = document.getElementById('imageSpinner');

    if (!input.files || input.files.length === 0) {
        resultDiv.innerHTML = '<p style="color: red;">Please select an image file.</p>';
        return;
    }

    const file = input.files[0];
    fileInfo.textContent = `Selected file: ${file.name}`;
    spinner.style.display = 'block';
    resultDiv.innerHTML = '';

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/predict/image', {
            method: 'POST',
            body: formData
        });

        spinner.style.display = 'none';

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        if (data.error) {
            resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `
                <p>Result: <span style="color: ${data.result === 'Deepfake' ? 'red' : 'green'}">${data.result}</span></p>
                <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
            `;
        }
    } catch (error) {
        spinner.style.display = 'none';
        resultDiv.innerHTML = '<p style="color: red;">An error occurred while processing the image.</p>';
        console.error('Error:', error);
    }
}
async function predictAudio() {
    const input = document.getElementById('audioInput');
    const fileInfo = document.getElementById('audioFileInfo');
    const resultDiv = document.getElementById('audioResult');
    const spinner = document.getElementById('audioSpinner');
    const preview = document.getElementById('audioPreview');

    // Reset previous results
    resultDiv.innerHTML = '';
    spinner.style.display = 'block';
    
    if (!input.files || input.files.length === 0) {
        resultDiv.innerHTML = '<div class="error">Please select an audio file.</div>';
        spinner.style.display = 'none';
        return;
    }

    const file = input.files[0];
    fileInfo.textContent = `Selected file: ${file.name}`;
    
    // Show audio player
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';

    const formData = new FormData();
    formData.append('audio', file);

    try {
        const response = await fetch('/predict/audio', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        spinner.style.display = 'none';

        if (!response.ok) {
            throw new Error(data.error || 'Server error');
        }

        // Display results
        resultDiv.innerHTML = `
            <div class="result-row">
                <span class="result-label">Result:</span>
                <span class="result-value ${data.result === 'Deepfake' ? 'fake' : 'real'}">
                    ${data.result}
                </span>
            </div>
            <div class="result-row">
                <span class="result-label">Confidence:</span>
                <span class="result-value">
                    ${(data.confidence * 100).toFixed(2)}%
                </span>
            </div>
            ${data.load_method ? `<div class="debug-info">Method: ${data.load_method}</div>` : ''}
        `;
    } catch (error) {
        spinner.style.display = 'none';
        resultDiv.innerHTML = `
            <div class="error">${error.message}</div>
            ${error.details ? `<div class="error-details">${JSON.stringify(error.details)}</div>` : ''}
        `;
        console.error('Audio processing error:', error);
    }
}

async function predictVideo() {
    const input = document.getElementById('videoInput');
    const fileInfo = document.getElementById('videoFileInfo');
    const resultDiv = document.getElementById('videoResult');
    const spinner = document.getElementById('videoSpinner');

    if (!input.files || input.files.length === 0) {
        resultDiv.innerHTML = '<p style="color: red;">Please select a video file.</p>';
        return;
    }

    const file = input.files[0];
    fileInfo.textContent = `Selected file: ${file.name}`;
    spinner.style.display = 'block';
    resultDiv.innerHTML = '';

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch('/predict/video', {
            method: 'POST',
            body: formData
        });

        spinner.style.display = 'none';

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        if (data.error) {
            resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `
                <p>Result: <span style="color: ${data.result === 'Deepfake' ? 'red' : 'green'}">${data.result}</span></p>
                <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
            `;
        }
    } catch (error) {
        spinner.style.display = 'none';
        resultDiv.innerHTML = '<p style="color: red;">An error occurred while processing the video.</p>';
        console.error('Error:', error);
    }
}