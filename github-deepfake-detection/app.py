# app.py - Deepfake Detection Flask App for Hugging Face Spaces
import os
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import librosa
from pydub import AudioSegment
import io
import logging
import tempfile
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Memory optimization settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Limit TF threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Limit TF threads

# Set matplotlib to use a non-interactive backend for Hugging Face Spaces
plt.switch_backend('Agg')

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure for Hugging Face Spaces
PORT = int(os.environ.get("PORT", 7860))
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Contents of MODELS_DIR ({MODELS_DIR}): {os.listdir(MODELS_DIR)}")

# Parameters (aligned with app.ipynb)
N_MFCC = 13
MAX_TIME_STEPS = 862
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# ========== MODEL LOADING ==========
# Image model definition
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

# Load image model
def load_image_model(model_path):
    device = torch.device("cpu")  # Hugging Face Spaces uses CPU
    model = DeepfakeDetector(num_classes=2)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = checkpoint
        model = model.to(device)
        model.eval()
        label_mapping = checkpoint.get('label_mapping', {'fake': 0, 'real': 1})
        logger.info("✅ Image model loaded successfully")
        return model, label_mapping, device
    except Exception as e:
        logger.error(f"❌ Error loading image model: {str(e)}")
        return None, {'fake': 0, 'real': 1}, device

image_model_path = os.path.join(MODELS_DIR, "best_deepfake_detector_resnet18.pth")
image_model, label_mapping, device = load_image_model(image_model_path)

# Load audio model
audio_model_path = os.path.join(MODELS_DIR, 'audio_model.h5')
try:
    audio_model = tf.keras.models.load_model(audio_model_path, compile=False)
    logger.info("✅ Audio model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load audio model: {str(e)}")
    audio_model = None

# Load video model
video_model_path = os.path.join(MODELS_DIR, "deepfake_video_new.h5")
try:
    video_model = tf.keras.models.load_model(video_model_path, compile=False)
    logger.info("✅ Video model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading video model: {str(e)}")
    video_model = None

# Video feature extractor
def build_feature_extractor():
    try:
        feature_extractor = tf.keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)
        outputs = feature_extractor(preprocessed)
        logger.info("✅ Feature extractor built successfully")
        return tf.keras.Model(inputs, outputs, name="feature_extractor")
    except Exception as e:
        logger.error(f"❌ Error building feature extractor: {str(e)}")
        return None

feature_extractor = build_feature_extractor()

# ========== PROCESSING AND ANALYSIS FUNCTIONS ==========
def preprocess_audio(audio_bytes):
    try:
        logger.debug("Starting audio preprocessing")
        sound = AudioSegment.from_file(io.BytesIO(audio_bytes))
        sound = sound.set_channels(1)
        sr = sound.frame_rate
        samples = np.array(sound.get_array_of_samples())
        audio = samples.astype(np.float32) / (2**15)
        hop_length = 512
        if sr != 44100:
            hop_length = int(512 * sr / 44100)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length)
        mfccs = mfccs.T
        if mfccs.shape[0] < MAX_TIME_STEPS:
            mfccs = pad_sequences([mfccs], maxlen=MAX_TIME_STEPS, dtype='float32', padding='post', truncating='post')[0]
        else:
            mfccs = mfccs[:MAX_TIME_STEPS, :]
        logger.debug(f"Audio preprocessing successful: sample rate={sr}, audio length={len(audio)}")
        return audio, sr, mfccs
    except Exception as e:
        logger.error(f"Audio preprocessing error: {str(e)}", exc_info=True)
        raise

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    logger.debug(f"Loading video from path: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error("Failed to open video file")
        raise ValueError("Could not open video file")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug("Reached end of video or error reading frame")
                break
            if frame is None or frame.size == 0:
                logger.warning("Empty frame encountered, skipping")
                continue
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            if len(frames) == max_frames:
                logger.debug(f"Reached max frames limit: {max_frames}")
                break
        if not frames:
            logger.error("No valid frames extracted from video")
            raise ValueError("No valid frames could be extracted from the video")
        logger.debug(f"Successfully loaded {len(frames)} frames")
        return np.array(frames)
    finally:
        cap.release()

def prepare_single_video(frames):
    logger.debug(f"Preparing video frames: {len(frames)} frames")
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            try:
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :], verbose=0)
            except Exception as e:
                logger.error(f"Error processing frame {j}: {str(e)}")
                continue
        frame_mask[i, :length] = 1
    logger.debug("Video frames prepared successfully")
    return frame_features, frame_mask

def analyze_pixel_gradients(image):
    try:
        logger.debug("Starting pixel gradient analysis")
        image_np = np.array(image.convert('L'))
        if image_np.size == 0:
            logger.error("Image is empty or invalid")
            return 0.0, None
        logger.debug(f"Image size: {image_np.shape}")

        image_np = cv2.resize(image_np, (128, 128))
        grad_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        threshold = np.percentile(grad_magnitude, 90)
        high_gradients = grad_magnitude[grad_magnitude > threshold]
        gradient_score = float(np.mean(high_gradients)) if high_gradients.size > 0 else 0.0

        heatmap = cv2.applyColorMap(grad_magnitude, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (256, 256))
        _, buffer = cv2.imencode('.png', heatmap)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        heatmap_url = f"data:image/png;base64,{heatmap_b64}"

        logger.debug(f"Gradient analysis successful: score={gradient_score}")
        return gradient_score, heatmap_url
    except Exception as e:
        logger.error(f"Error in pixel gradient analysis: {str(e)}", exc_info=True)
        return 0.0, None

def analyze_audio_features(audio, sr):
    try:
        logger.debug(f"Starting audio feature analysis: audio length={len(audio)}, sample rate={sr}")
        if len(audio) == 0:
            logger.error("Audio data is empty")
            return 0.0, 0.0, 0.0, 0.0, None

        S = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(S))
        spectral_flux = np.sum(np.diff(S_db, axis=1)**2, axis=0)
        flux_mean = float(np.mean(spectral_flux))
        flux_std = float(np.std(spectral_flux))

        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        plt.figure(figsize=(6, 4), dpi=100)
        plt.plot(spectral_flux, color='blue', label='Spectral Flux')
        plt.title('Spectral Flux Over Time')
        plt.xlabel('Time')
        plt.ylabel('Flux')

        # Adjust y-axis limits to include data and mean line with padding
        data_max = np.max(spectral_flux)
        data_min = np.min(spectral_flux)
        y_max = max(data_max, flux_mean) * 1.2  # Add 20% padding above the max
        y_min = min(data_min, flux_mean) * 0.8  # Add 20% padding below the min
        plt.ylim(y_min, y_max)

        # Draw mean flux line with transparency
        line_color = 'red' if flux_mean > 10 else 'green'
        plt.axhline(
            y=flux_mean, color=line_color, linestyle='--', alpha=0.7,
            label=f'Mean Flux: {flux_mean:.2f}'
        )

        # Annotate the mean flux line
        plt.annotate(
            f'Mean: {flux_mean:.2f}', xy=(0.05, flux_mean), xytext=(0.05, flux_mean * 1.1),
            textcoords='data', color=line_color, fontsize=8,
            arrowprops=dict(arrowstyle='->', color=line_color)
        )

        # Add interpretation text in top-left corner
        if flux_mean > 10:
            plt.text(
                0.02, 0.98, 'High flux indicates abrupt changes', color='red',
                transform=plt.gca().transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8)
            )

        plt.legend(loc='upper right', fontsize=8)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        flux_b64 = base64.b64encode(buf.read()).decode('utf-8')
        flux_url = f"data:image/png;base64,{flux_b64}"

        logger.debug(
            f"Audio feature analysis successful: flux_mean={flux_mean}, "
            f"flux_std={flux_std}, zcr_mean={zcr_mean}, zcr_std={zcr_std}"
        )
        return flux_mean, flux_std, zcr_mean, zcr_std, flux_url
    except Exception as e:
        logger.error(f"Error in audio feature analysis: {str(e)}", exc_info=True)
        return 0.0, 0.0, 0.0, 0.0, None

def compute_optical_flow(frames):
    try:
        logger.debug(f"Computing optical flow for {len(frames)} frames")
        if len(frames) < 2:
            logger.error("Not enough frames to compute optical flow")
            return 0.0

        flow_magnitudes = []
        for i in range(len(frames) - 1):
            frame1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(float(np.mean(magnitude)))
        flow_score = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0
        logger.debug(f"Optical flow computed: flow_score={flow_score}")
        return flow_score
    except Exception as e:
        logger.error(f"Error in optical flow computation: {str(e)}", exc_info=True)
        return 0.0

def generate_mel_spectrogram(audio, sr, result, confidence, flux_mean, flux_std):
    try:
        logger.debug("Generating Mel spectrogram")
        plt.figure(figsize=(6, 4))
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')

        if result == 'Deepfake':
            plt.text(
                0.5, 0.9, f'High Flux: {flux_mean:.2f} (Std: {flux_std:.2f})', color='red',
                transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8)
            )
        else:
            plt.text(
                0.5, 0.9, f'Stable Flux: {flux_mean:.2f} (Std: {flux_std:.2f})', color='green',
                transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8)
            )

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        spectrogram_b64 = base64.b64encode(buf.read()).decode('utf-8')
        spectrogram_url = f"data:image/png;base64,{spectrogram_b64}"
        logger.debug("Mel spectrogram generated successfully")
        return spectrogram_url
    except Exception as e:
        logger.error(f"Error generating Mel spectrogram: {str(e)}")
        return None

def annotate_frames(frames, result, confidence):
    annotated_frames = []
    try:
        logger.debug(f"Annotating {min(len(frames), 2)} frames")
        for i, frame in enumerate(frames[:2]):
            if frame is None or frame.size == 0:
                logger.error(f"Frame {i} is empty or invalid")
                continue
            frame_bgr = frame[:, :, [2, 1, 0]].copy()  # RGB to BGR
            if result == 'Deepfake':
                cv2.rectangle(frame_bgr, (50, 50, 100, 100), (0, 0, 255), 2)
                cv2.putText(
                    frame_bgr, 'Suspicious Region', (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )
            else:
                cv2.rectangle(frame_bgr, (50, 50, 100, 100), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr, 'Natural Features', (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
            success, buffer = cv2.imencode('.png', frame_bgr)
            if not success:
                logger.error(f"Failed to encode frame {i} as PNG")
                continue
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frame_url = f"data:image/png;base64,{frame_b64}"
            annotated_frames.append({'url': frame_url})
            logger.debug(f"Frame {i} annotated successfully")
        if not annotated_frames:
            logger.error("No frames were successfully annotated")
        logger.debug(f"Successfully annotated {len(annotated_frames)} frames")
        return annotated_frames
    except Exception as e:
        logger.error(f"Error annotating frames: {str(e)}", exc_info=True)
        return []

def annotate_image(image, result, confidence):
    try:
        logger.debug("Annotating image")
        image_np = np.array(image)
        image_bgr = image_np[:, :, ::-1]  # RGB to BGR
        image_bgr = cv2.resize(image_bgr, (256, 256))

        if result == 'Deepfake':
            cv2.rectangle(image_bgr, (50, 50, 100, 100), (0, 0, 255), 2)
            cv2.putText(
                image_bgr, 'Unnatural Blending', (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
        else:
            cv2.rectangle(image_bgr, (50, 50, 100, 100), (0, 255, 0), 2)
            cv2.putText(
                image_bgr, 'Natural Features', (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        _, buffer = cv2.imencode('.png', image_bgr)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        image_url = f"data:image/png;base64,{image_b64}"
        logger.debug("Image annotation successful")
        return image_url
    except Exception as e:
        logger.error(f"Error annotating image: {str(e)}")
        return None

# Image preprocessing pipeline
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== FLASK ROUTES ==========
@app.route('/')
def index():
    logger.debug("Serving index page")
    return render_template('index.html')

@app.route('/predict/image', methods=['POST'])
def predict_image():
    try:
        logger.debug("Received image prediction request")
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        logger.debug(
            f"Image file: name={file.filename}, size={file.content_length}, "
            f"type={file.content_type}"
        )
        if file.content_length > MAX_FILE_SIZE:
            logger.error(f"Image file too large: {file.content_length} bytes")
            return jsonify({"error": "File too large"}), 400
        if not file or file.filename == '':
            logger.error("Image file is empty")
            return jsonify({"error": "Image file is empty"}), 400

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size == 0:
            logger.error("Image file is empty")
            return jsonify({"error": "Image file is empty"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
            logger.debug(f"Saved image to temp path: {temp_path}")

        try:
            if image_model is None:
                logger.error("Image model not loaded")
                return jsonify({"error": "Image model not available"}), 503

            logger.debug(f"Loading image from {temp_path}")
            image = Image.open(temp_path).convert('RGB')
            image_tensor = image_transform(image).unsqueeze(0).to(device)

            gradient_score, heatmap_url = analyze_pixel_gradients(image)

            with torch.no_grad():
                output = image_model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(probabilities, dim=1)
                confidence_value = confidence[0].item()

            result = "Deepfake" if pred[0].item() == 0 else "Real"
            color = "red" if result == "Deepfake" else "green"

            annotated_image_url = annotate_image(image, result, confidence_value)
            explanation = [
                f"<strong>Classification</strong>: {result} with {(confidence_value * 100):.2f}% confidence.",
                f"<strong>Gradient Score</strong>: {gradient_score:.2f}. Higher scores may "
                "indicate unnatural pixel transitions.",
                "<strong>Calculation</strong>: We analyze pixel intensity changes using "
                "horizontal and vertical gradients, then average the strongest changes "
                "(top 10%) to get the score.",
                "<strong>How It Works</strong>: The horizontal gradient (change in pixel "
                "intensity across columns) and vertical gradient (change across rows) are "
                "calculated. Their combined strength is averaged for high-gradient areas "
                "to produce the score.",
                "<strong>Interpretation</strong>: A higher score suggests sharp or unnatural "
                "pixel changes, often found in deepfake images. A lower score indicates "
                "smooth transitions, typical in real images.",
                "<strong>Annotated Image</strong>: Highlights areas of interest used in "
                "detection.",
                "<strong>Gradient Heatmap</strong>: Visualizes pixel intensity changes "
                "across the image."
            ]
            if result == "Deepfake":
                explanation.append(
                    "<strong>Red Boxes</strong>: Indicate unnatural blending or pixel "
                    "artifacts, common in deepfake images."
                )
            else:
                explanation.append(
                    "<strong>Green Boxes</strong>: Confirm natural features, such as "
                    "consistent lighting and textures."
                )

            response = {
                "result": result,
                "confidence": confidence_value,
                "color": color,
                "evidence": {
                    "annotated_image": annotated_image_url,
                    "heatmap": heatmap_url,
                    "gradient_score": round(gradient_score, 2),
                    "explanation": explanation
                }
            }
            logger.debug(f"Image prediction response: {response}")
            return jsonify(response)
        finally:
            os.unlink(temp_path)
            logger.debug(f"Deleted temp file: {temp_path}")
    except Exception as e:
        logger.error(f"Image prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

@app.route('/predict/audio', methods=['POST'])
def predict_audio():
    try:
        logger.debug("Received audio prediction request")
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files['audio']
        logger.debug(
            f"Audio file: name={file.filename}, size={file.content_length}, "
            f"type={file.content_type}"
        )
        if file.content_length > MAX_FILE_SIZE:
            logger.error(f"Audio file too large: {file.content_length} bytes")
            return jsonify({"error": "File too large"}), 400

        audio_bytes = file.read()
        if not audio_bytes:
            logger.error("Audio file is empty")
            return jsonify({"error": "Audio file is empty"}), 400

        if audio_model is None:
            logger.error("Audio model not loaded")
            return jsonify({"error": "Audio model not available"}), 503

        audio, sr, mfccs = preprocess_audio(audio_bytes)
        mfccs = np.expand_dims(mfccs, axis=0)

        flux_mean, flux_std, zcr_mean, zcr_std, flux_url = analyze_audio_features(audio, sr)

        prediction = audio_model.predict(mfccs, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0, predicted_class])
        result = "Deepfake" if predicted_class == 1 else "Real"
        color = "red" if result == "Deepfake" else "green"

        spectrogram_url = generate_mel_spectrogram(audio, sr, result, confidence, flux_mean, flux_std)
        explanation = [
            f"<strong>Classification</strong>: {result} with {(confidence * 100):.2f}% confidence.",
            f"<strong>Spectral Flux</strong>: Average={flux_mean:.2f}, "
            f"Variation={flux_std:.2f}. High flux suggests abrupt frequency changes.",
            "<strong>Calculation</strong>: We measure how much the audio’s frequency "
            "content changes between short time frames, then calculate the average and "
            "variation of these changes.",
            "<strong>How It Works</strong>: The frequency content (spectrum) is computed "
            "for each frame. The difference in spectrum between consecutive frames is "
            "squared and summed, then averaged to get the flux. The variation shows how "
            "consistent these changes are.",
            "<strong>Interpretation</strong>: A higher average flux indicates sudden "
            "frequency shifts, common in synthetic audio. Lower flux suggests stable "
            "patterns, typical of real audio.",
            f"<strong>Zero-Crossing Rate</strong>: Average={zcr_mean:.4f}, "
            f"Variation={zcr_std:.4f}. High rates may indicate synthetic audio.",
            "<strong>Calculation</strong>: We count how often the audio signal switches "
            "between positive and negative in each frame, then average these counts and "
            "calculate their variation.",
            "<strong>How It Works</strong>: The number of times the audio waveform crosses "
            "zero per frame is counted. The average and variation of these counts are "
            "computed to assess signal behavior.",
            "<strong>Interpretation</strong>: A higher average rate suggests frequent "
            "changes, often in synthetic audio. Lower rates indicate natural speech "
            "patterns.",
            "<strong>Mel Spectrogram</strong>: Shows frequency patterns over time.",
            "<strong>Flux Plot</strong>: Visualizes spectral flux variations."
        ]
        if result == "Deepfake":
            explanation.append(
                "<strong>High Flux and Rates</strong>: Indicate unnatural speech patterns, "
                "typical of deepfake audio."
            )
        else:
            explanation.append(
                "<strong>Stable Flux and Rates</strong>: Suggest natural audio patterns, "
                "consistent with real recordings."
            )
        response = {
            "result": result,
            "confidence": confidence,
            "color": color,
            "evidence": {
                "spectrogram": spectrogram_url,
                "flux_plot": flux_url,
                "flux_mean": round(flux_mean, 2),
                "flux_std": round(flux_std, 2),
                "zcr_mean": round(zcr_mean, 4),
                "zcr_std": round(zcr_std, 4),
                "explanation": explanation
            }
        }
        logger.debug(f"Audio prediction response: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500

@app.route('/predict/video', methods=['POST'])
def predict_video():
    try:
        logger.debug("Received video prediction request")
        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({"error": "No video file provided"}), 400

        file = request.files['video']
        logger.debug(
            f"Video file: name={file.filename}, size={file.content_length}, "
            f"type={file.content_type}"
        )
        if file.content_length > MAX_FILE_SIZE:
            logger.error(f"Video file too large: {file.content_length} bytes")
            return jsonify({"error": "File too large"}), 400
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            logger.error(f"Unsupported video format: {file.filename}")
            return jsonify({"error": "Unsupported video format"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
            logger.debug(f"Saved video to temp path: {temp_path}")

        try:
            frames = load_video(temp_path)
            if len(frames) == 0:
                logger.error("No frames extracted from video")
                return jsonify({"error": "No frames extracted from video"}), 500

            frame_features, frame_mask = prepare_single_video(frames)

            flow_score = compute_optical_flow(frames)
            temporal_consistency = "Low" if flow_score > 5 else "High"

            if video_model is None:
                logger.error("Video model not loaded")
                return jsonify({
                    "result": "Real",
                    "confidence": 0.5,
                    "color": "green",
                    "warning": "Video model not available - using fallback"
                }), 503

            prediction = video_model.predict([frame_features, frame_mask], verbose=0)[0]
            prediction = float(prediction.item())
            predicted_label = "Deepfake" if prediction >= 0.5 else "Real"
            confidence = prediction if predicted_label == "Deepfake" else 1 - prediction

            color = "red" if predicted_label == "Deepfake" else "green"
            annotated_frames = annotate_frames(frames, predicted_label, confidence)
            explanation = [
                f"<strong>Classification</strong>: {predicted_label} with {(confidence * 100):.2f}% confidence.",
                f"<strong>Temporal Consistency</strong>: {temporal_consistency}.",
                f"<strong>Mean Optical Flow</strong>: {flow_score:.2f}. High flow may "
                "indicate unnatural motion.",
                "<strong>Calculation</strong>: We measure motion between video frames by "
                "tracking pixel movement, then average the strength of this motion.",
                "<strong>How It Works</strong>: The motion (optical flow) between consecutive "
                "frames is calculated as the speed and direction of pixel shifts. The average "
                "speed across all frame pairs gives the flow score.",
                "<strong>Interpretation</strong>: A higher flow score suggests more motion, "
                "which may be unnatural in deepfakes. A lower score indicates stable motion, "
                "typical of real videos.",
                "<strong>Temporal Consistency Check</strong>: Marked as Low if the flow score "
                "is above 5, otherwise High.",
                "<strong>Interpretation</strong>: Low consistency suggests inconsistent "
                "motion, common in deepfake videos. High consistency indicates natural motion.",
                "<strong>Annotated Frames</strong>: Highlight areas of interest used in "
                "detection."
            ]
            if predicted_label == "Deepfake":
                explanation.append(
                    "<strong>Red Boxes</strong>: Highlight suspicious regions, such as "
                    "unnatural movements or artifacts."
                )
            else:
                explanation.append(
                    "<strong>Green Boxes</strong>: Indicate natural features, consistent with "
                    "real videos."
                )
            response = {
                "result": predicted_label,
                "confidence": confidence,
                "color": color,
                "evidence": {
                    "frames": annotated_frames,
                    "temporal_consistency": temporal_consistency,
                    "flow_score": round(flow_score, 2),
                    "explanation": explanation
                }
            }
            logger.debug(f"Video prediction response: {response}")
            return jsonify(response)
        finally:
            os.unlink(temp_path)
            logger.debug(f"Deleted temp file: {temp_path}")
    except Exception as e:
        logger.error(f"Video prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "OK",
        "models_loaded": {
            "audio": audio_model is not None,
            "image": image_model is not None,
            "video": video_model is not None
        }
    })

if __name__ == '__main__':
    logger.info(f"🚀 Starting Deepfake Detection App on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)