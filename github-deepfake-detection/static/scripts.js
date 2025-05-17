let currentImageUrl = null;
let currentAudioUrl = null;
let currentVideoUrl = null;

function previewImage(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('imagePreview');
    const fileInfo = document.getElementById('imageFileInfo');
    const resultDiv = document.getElementById('imageResult');

    resultDiv.innerHTML = '';
    preview.style.display = 'none';

    if (file) {
        if (currentImageUrl) {
            URL.revokeObjectURL(currentImageUrl);
        }
        currentImageUrl = URL.createObjectURL(file);
        preview.src = currentImageUrl;
        preview.style.display = 'block';
        fileInfo.textContent = `Selected file: ${file.name}`;
    } else {
        fileInfo.textContent = '';
    }
}

function previewAudio(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('audioPreview');
    const fileInfo = document.getElementById('audioFileInfo');
    const resultDiv = document.getElementById('audioResult');

    resultDiv.innerHTML = '';
    preview.style.display = 'none';

    if (file) {
        if (currentAudioUrl) {
            URL.revokeObjectURL(currentAudioUrl);
        }
        currentAudioUrl = URL.createObjectURL(file);
        preview.src = currentAudioUrl;
        preview.style.display = 'block';
        fileInfo.textContent = `Selected file: ${file.name}`;
    } else {
        fileInfo.textContent = '';
    }
}

function previewVideo(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('videoPreview');
    const fileInfo = document.getElementById('videoFileInfo');
    const resultDiv = document.getElementById('videoResult');

    resultDiv.innerHTML = '';
    preview.style.display = 'none';

    if (file) {
        if (currentVideoUrl) {
            URL.revokeObjectURL(currentVideoUrl);
        }
        currentVideoUrl = URL.createObjectURL(file);
        preview.src = currentVideoUrl;
        preview.style.display = 'block';
        fileInfo.textContent = `Selected file: ${file.name}`;
    } else {
        fileInfo.textContent = '';
    }
}

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const SUPPORTED_IMAGE_TYPES = ['image/jpeg', 'image/png'];
const SUPPORTED_AUDIO_TYPES = ['audio/mpeg', 'audio/wav', 'audio/mp3'];
const SUPPORTED_VIDEO_TYPES = ['video/mp4', 'video/avi', 'video/mov'];

async function validateFile(file, supportedTypes, maxSize, resultDiv) {
    if (!file) {
        resultDiv.innerHTML = '<div class="error" role="alert">Please select a file.</div>';
        return false;
    }
    if (!supportedTypes.includes(file.type)) {
        resultDiv.innerHTML = `<div class="error" role="alert">Unsupported file type. Please upload a ${supportedTypes.join(', ')} file.</div>`;
        return false;
    }
    if (file.size > maxSize) {
        resultDiv.innerHTML = `<div class="error" role="alert">File too large. Maximum size is ${maxSize / (1024 * 1024)}MB.</div>`;
        return false;
    }
    try {
        await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve();
            reader.onerror = () => reject(new Error('File is unreadable'));
            reader.readAsArrayBuffer(file);
        });
        return true;
    } catch (error) {
        resultDiv.innerHTML = `<div class="error" role="alert">File is corrupted or unreadable.</div>`;
        return false;
    }
}

function clearSection(inputId, previewId, fileInfoId, resultId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const fileInfo = document.getElementById(fileInfoId);
    const resultDiv = document.getElementById(resultId);

    input.value = '';
    preview.style.display = 'none';
    if (previewId === 'imagePreview' && currentImageUrl) {
        URL.revokeObjectURL(currentImageUrl);
        currentImageUrl = null;
    } else if (previewId === 'audioPreview' && currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
        currentAudioUrl = null;
    } else if (previewId === 'videoPreview' && currentVideoUrl) {
        URL.revokeObjectURL(currentVideoUrl);
        currentVideoUrl = null;
    }
    preview.src = '';
    fileInfo.textContent = '';
    resultDiv.innerHTML = '';
}

async function predictImage() {
    const input = document.getElementById('imageInput');
    const fileInfo = document.getElementById('imageFileInfo');
    const resultDiv = document.getElementById('imageResult');
    const spinner = document.getElementById('imageSpinner');

    resultDiv.innerHTML = '';
    spinner.style.display = 'block';

    if (!input.files || input.files.length === 0) {
        resultDiv.innerHTML = '<div class="error" role="alert">Please select an image file.</div>';
        spinner.style.display = 'none';
        return;
    }

    const file = input.files[0];
    if (!(await validateFile(file, SUPPORTED_IMAGE_TYPES, MAX_FILE_SIZE, resultDiv))) {
        spinner.style.display = 'none';
        return;
    }

    fileInfo.textContent = `Selected file: ${file.name}`;
    resultDiv.innerHTML = '';

    const formData = new FormData();
    formData.append('image', file);

    try {
        console.debug('Sending image prediction request:', { fileName: file.name, size: file.size, type: file.type });
        const response = await fetch('/predict/image', {
            method: 'POST',
            body: formData
        });

        console.debug('Image prediction response status:', response.status);
        spinner.style.display = 'none';

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Image prediction failed:', errorText);
            throw new Error(`Network response was not ok: ${errorText}`);
        }

        const data = await response.json();
        console.debug('Image prediction response data:', data);
        if (data.error) {
            resultDiv.innerHTML = `<div class="error" role="alert">Error: ${data.error}</div>`;
            return;
        }

        let evidenceHtml = '';
        if (data.evidence && typeof data.evidence === 'object') {
            const annotatedImage = data.evidence.annotated_image && typeof data.evidence.annotated_image === 'string' && data.evidence.annotated_image.length > 0 ? `
                <h4>Annotated Image</h4>
                <img src="${data.evidence.annotated_image}" class="evidence-image" alt="Image with highlighted regions for deepfake detection">
            ` : '<p class="evidence-text">Annotated image unavailable</p>';

            const heatmap = data.evidence.heatmap && typeof data.evidence.heatmap === 'string' && data.evidence.heatmap.length > 0 ? `
                <h4>Gradient Heatmap</h4>
                <img src="${data.evidence.heatmap}" class="evidence-image" alt="Heatmap showing pixel gradient intensities">
            ` : '<p class="evidence-text">Heatmap unavailable</p>';

            const gradientScore = typeof data.evidence.gradient_score === 'number' ? `
                <p class="evidence-text"><strong>Gradient Score:</strong> ${data.evidence.gradient_score.toFixed(2)}</p>
            ` : '<p class="evidence-text">Gradient score unavailable</p>';

            const explanation = data.evidence.explanation && Array.isArray(data.evidence.explanation) && data.evidence.explanation.length > 0 ? `
                <ul class="evidence-text">
                    ${data.evidence.explanation.map(item => `<li>${item}</li>`).join('')}
                </ul>
            ` : '<p class="evidence-text">Explanation unavailable</p>';

            evidenceHtml = `
                <div class="evidence-container">
                    ${annotatedImage}
                    ${heatmap}
                    ${gradientScore}
                    ${explanation}
                </div>
            `;
        } else {
            evidenceHtml = '<div class="evidence-container"><p class="evidence-text">No evidence available</p></div>';
        }

        resultDiv.innerHTML = `
            <div class="result-row">
                <span class="result-label">Result:</span>
                <span class="result-value ${data.result === 'Deepfake' ? 'fake' : 'real'}">
                    ${data.result || 'Unknown'}
                </span>
            </div>
            <div class="result-row">
                <span class="result-label">Confidence:</span>
                <span class="result-value">
                    ${(typeof data.confidence === 'number' ? data.confidence * 100 : 0).toFixed(2)}%
                </span>
            </div>
            ${evidenceHtml}
            <button class="inline" onclick="clearSection('imageInput', 'imagePreview', 'imageFileInfo', 'imageResult')" style="margin-top: 10px;">Clear</button>
        `;
    } catch (error) {
        console.error('Image prediction error:', error);
        spinner.style.display = 'none';
        resultDiv.innerHTML = `
            <div class="error" role="alert">An error occurred while processing the image.</div>
            <div class="error-details">${error.message}</div>
        `;
    }
}

async function predictAudio() {
    const input = document.getElementById('audioInput');
    const fileInfo = document.getElementById('audioFileInfo');
    const resultDiv = document.getElementById('audioResult');
    const spinner = document.getElementById('audioSpinner');
    const preview = document.getElementById('audioPreview');

    resultDiv.innerHTML = '';
    spinner.style.display = 'block';

    if (!input.files || input.files.length === 0) {
        resultDiv.innerHTML = '<div class="error" role="alert">Please select an audio file.</div>';
        spinner.style.display = 'none';
        return;
    }

    const file = input.files[0];
    if (!(await validateFile(file, SUPPORTED_AUDIO_TYPES, MAX_FILE_SIZE, resultDiv))) {
        spinner.style.display = 'none';
        return;
    }

    fileInfo.textContent = `Selected file: ${file.name}`;
    preview.src = currentAudioUrl || URL.createObjectURL(file);
    preview.style.display = 'block';

    const formData = new FormData();
    formData.append('audio', file);

    try {
        console.debug('Sending audio prediction request:', { fileName: file.name, size: file.size, type: file.type });
        const response = await fetch('/predict/audio', {
            method: 'POST',
            body: formData
        });

        console.debug('Audio prediction response status:', response.status);
        spinner.style.display = 'none';

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Audio prediction failed:', errorText);
            throw new Error(`Network response was not ok: ${errorText}`);
        }

        const data = await response.json();
        console.debug('Audio prediction response data:', data);
        if (data.error) {
            resultDiv.innerHTML = `<div class="error" role="alert">Error: ${data.error}</div>`;
            return;
        }

        let evidenceHtml = '';
        if (data.evidence && typeof data.evidence === 'object') {
            const spectrogram = data.evidence.spectrogram && typeof data.evidence.spectrogram === 'string' && data.evidence.spectrogram.length > 0 ? `
                <h4>Mel Spectrogram</h4>
                <img src="${data.evidence.spectrogram}" class="evidence-image" alt="Mel spectrogram showing frequency patterns over time">
            ` : '<p class="evidence-text">Mel spectrogram unavailable</p>';

            const fluxPlot = data.evidence.flux_plot && typeof data.evidence.flux_plot === 'string' && data.evidence.flux_plot.length > 0 ? `
                <h4>Spectral Flux Plot</h4>
                <img src="${data.evidence.flux_plot}" class="evidence-image" alt="Plot showing spectral flux variations over time">
            ` : '<p class="evidence-text">Spectral flux plot unavailable</p>';

            const spectralFlux = (typeof data.evidence.flux_mean === 'number' && typeof data.evidence.flux_std === 'number') ? `
                <p class="evidence-text"><strong>Spectral Flux:</strong> Mean=${data.evidence.flux_mean.toFixed(2)}, Std=${data.evidence.flux_std.toFixed(2)}</p>
            ` : '<p class="evidence-text">Spectral flux metrics unavailable</p>';

            const zcr = (typeof data.evidence.zcr_mean === 'number' && typeof data.evidence.zcr_std === 'number') ? `
                <p class="evidence-text"><strong>Zero-Crossing Rate:</strong> Mean=${data.evidence.zcr_mean.toFixed(4)}, Std=${data.evidence.zcr_std.toFixed(4)}</p>
            ` : '<p class="evidence-text">Zero-crossing rate metrics unavailable</p>';

            const explanation = data.evidence.explanation && Array.isArray(data.evidence.explanation) && data.evidence.explanation.length > 0 ? `
                <ul class="evidence-text">
                    ${data.evidence.explanation.map(item => `<li>${item}</li>`).join('')}
                </ul>
            ` : '<p class="evidence-text">Explanation unavailable</p>';

            evidenceHtml = `
                <div class="evidence-container">
                    ${spectrogram}
                    ${fluxPlot}
                    ${spectralFlux}
                    ${zcr}
                    ${explanation}
                </div>
            `;
        } else {
            evidenceHtml = '<div class="evidence-container"><p class="evidence-text">No evidence available</p></div>';
        }

        resultDiv.innerHTML = `
            <div class="result-row">
                <span class="result-label">Result:</span>
                <span class="result-value ${data.result === 'Deepfake' ? 'fake' : 'real'}">
                    ${data.result || 'Unknown'}
                </span>
            </div>
            <div class="result-row">
                <span class="result-label">Confidence:</span>
                <span class="result-value">
                    ${(typeof data.confidence === 'number' ? data.confidence * 100 : 0).toFixed(2)}%
                </span>
            </div>
            ${evidenceHtml}
            <button class="inline" onclick="clearSection('audioInput', 'audioPreview', 'audioFileInfo', 'audioResult')" style="margin-top: 10px;">Clear</button>
        `;
    } catch (error) {
        console.error('Audio prediction error:', error);
        spinner.style.display = 'none';
        resultDiv.innerHTML = `
            <div class="error" role="alert">An error occurred while processing the audio.</div>
            <div class="error-details">${error.message}</div>
        `;
    }
}

async function predictVideo() {
    const input = document.getElementById('videoInput');
    const fileInfo = document.getElementById('videoFileInfo');
    const resultDiv = document.getElementById('videoResult');
    const spinner = document.getElementById('videoSpinner');

    resultDiv.innerHTML = '';
    spinner.style.display = 'block';

    if (!input.files || input.files.length === 0) {
        resultDiv.innerHTML = '<div class="error" role="alert">Please select a video file.</div>';
        spinner.style.display = 'none';
        return;
    }

    const file = input.files[0];
    if (!(await validateFile(file, SUPPORTED_VIDEO_TYPES, MAX_FILE_SIZE, resultDiv))) {
        spinner.style.display = 'none';
        return;
    }

    fileInfo.textContent = `Selected file: ${file.name}`;
    resultDiv.innerHTML = '';

    const formData = new FormData();
    formData.append('video', file);

    try {
        console.debug('Sending video prediction request:', { fileName: file.name, size: file.size, type: file.type });
        const response = await fetch('/predict/video', {
            method: 'POST',
            body: formData
        });

        console.debug('Video prediction response status:', response.status);
        spinner.style.display = 'none';

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Video prediction failed:', errorText);
            throw new Error(`Network response was not ok: ${errorText}`);
        }

        const data = await response.json();
        console.debug('Video prediction response data:', data);
        if (data.error) {
            resultDiv.innerHTML = `<div class="error" role="alert">Error: ${data.error}</div>`;
            return;
        }

        let evidenceHtml = '';
        if (data.evidence && typeof data.evidence === 'object') {
            const framesHtml = (data.evidence.frames && Array.isArray(data.evidence.frames) && data.evidence.frames.length > 0) ? `
                <h4>Annotated Frames</h4>
                ${data.evidence.frames.map((frame, index) => 
                    frame.url && typeof frame.url === 'string' && frame.url.length > 0 ? `
                        <img src="${frame.url}" class="evidence-image" alt="Annotated video frame ${index + 1} for deepfake detection">
                    ` : `<p class="evidence-text">Frame ${index + 1} unavailable</p>`
                ).join('')}
            ` : '<p class="evidence-text">Annotated frames unavailable</p>';

            const temporal = (data.evidence.temporal_consistency && typeof data.evidence.temporal_consistency === 'string' && typeof data.evidence.flow_score === 'number') ? `
                <p class="evidence-text"><strong>Temporal Consistency:</strong> ${data.evidence.temporal_consistency} (Flow Score: ${data.evidence.flow_score.toFixed(2)}</p>
            ` : '<p class="evidence-text">Temporal consistency unavailable</p>';

            const explanation = data.evidence.explanation && Array.isArray(data.evidence.explanation) && data.evidence.explanation.length > 0 ? `
                <ul class="evidence-text">
                    ${data.evidence.explanation.map(item => `<li>${item}</li>`).join('')}
                </ul>
            ` : '<p class="evidence-text">Explanation unavailable</p>';

            evidenceHtml = `
                <div class="evidence-container">
                    ${framesHtml}
                    ${temporal}
                    ${explanation}
                </div>
            `;
        } else {
            evidenceHtml = '<div class="evidence-container"><p class="evidence-text">No evidence available</p></div>';
        }

        resultDiv.innerHTML = `
            <div class="result-row">
                <span class="result-label">Result:</span>
                <span class="result-value ${data.result === 'Deepfake' ? 'fake' : 'real'}">
                    ${data.result || 'Unknown'}
                </span>
            </div>
            <div class="result-row">
                <span class="result-label">Confidence:</span>
                <span class="result-value">
                    ${(typeof data.confidence === 'number' ? data.confidence * 100 : 0).toFixed(2)}%
                </span>
            </div>
            ${evidenceHtml}
            <button class="inline" onclick="clearSection('videoInput', 'videoPreview', 'videoFileInfo', 'videoResult')" style="margin-top: 10px;">Clear</button>
        `;
    } catch (error) {
        console.error('Video prediction error:', error);
        spinner.style.display = 'none';
        resultDiv.innerHTML = `
            <div class="error" role="alert">An error occurred while processing the video.</div>
            <div class="error-details">${error.message}</div>
        `;
    }
}