# app.py - Deepfake Detection Flask App with Memory Optimizations
import os
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import librosa
from pydub import AudioSegment
import io
import logging
import tempfile
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Memory optimization settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Limit TF threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Limit TF threads

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure for Render environment
PORT = int(os.environ.get("PORT", 10000))
MODELS_DIR = "/opt/render/project/src/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# Parameters (further optimized)
N_MFCC = 13
MAX_TIME_STEPS = 400  # Reduced from 862
MAX_FILE_SIZE = 10 * 1024 * 1024  # Reduced to 5MB
IMG_SIZE = 128  # Already optimized
MAX_SEQ_LENGTH = 10  # Reduced from 20
NUM_FEATURES = 2048

# ========== LAZY LOADING IMPLEMENTATION ==========
class ModelManager:
    def __init__(self):
        self.models = {
            'audio_model': None,
            'image_model': None,
            'video_model': None,
            'feature_extractor': None,
            'image_device': None,
            'label_mapping': None
        }
        self.loaded = False
    
    def load_image_model(self):
        if self.models['image_model'] is None:
            try:
                image_model_path = os.path.join(MODELS_DIR, "best_deepfake_detector_resnet18.pth")
                if not os.path.exists(image_model_path):
                    logger.error(f"Image model file not found: {image_model_path}")
                    return
                device = torch.device("cpu")
                model = DeepfakeDetector(num_classes=2)
                checkpoint = torch.load(image_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                model.eval()
                self.models['image_model'] = model
                self.models['image_device'] = device
                self.models['label_mapping'] = checkpoint.get('label_mapping', {'fake': 0, 'real': 1})
                logger.info("✅ Image model loaded")
            except Exception as e:
                logger.error(f"❌ Image model loading failed: {str(e)}", exc_info=True)
    
    def load_audio_model(self):
        if self.models['audio_model'] is None:
            try:
                audio_model_path = os.path.join(MODELS_DIR, 'audio_model.tflite')  # Use TFLite
                if not os.path.exists(audio_model_path):
                    logger.error(f"Audio model file not found: {audio_model_path}")
                    return
                interpreter = tf.lite.Interpreter(model_path=audio_model_path)
                interpreter.allocate_tensors()
                self.models['audio_model'] = interpreter
                logger.info("✅ Audio TFLite model loaded")
            except Exception as e:
                logger.error(f"❌ Audio model loading failed: {str(e)}", exc_info=True)
    
    def load_video_models(self):
        if self.models['video_model'] is None:
            try:
                video_model_path = os.path.join(MODELS_DIR, "deepfake_video_new.tflite")  # Use TFLite
                if not os.path.exists(video_model_path):
                    logger.error(f"Video model file not found: {video_model_path}")
                    return
                interpreter = tf.lite.Interpreter(model_path=video_model_path)
                interpreter.allocate_tensors()
                self.models['video_model'] = interpreter
                if self.models['feature_extractor'] is None:
                    self.models['feature_extractor'] = tf.keras.applications.InceptionV3(
                        weights="imagenet",
                        include_top=False,
                        pooling="avg",
                        input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    )
                logger.info("✅ Video TFLite model loaded")
            except Exception as e:
                logger.error(f"❌ Video model loading failed: {str(e)}", exc_info=True)
    
    def unload_models(self):
        if self.models['audio_model'] is not None:
            del self.models['audio_model']
            self.models['audio_model'] = None
        if self.models['video_model'] is not None:
            del self.models['video_model']
            self.models['video_model'] = None
        if self.models['feature_extractor'] is not None:
            del self.models['feature_extractor']
            self.models['feature_extractor'] = None
        if self.models['image_model'] is not None:
            del self.models['image_model']
            self.models['image_model'] = None
            torch.cuda.empty_cache()
        tf.keras.backend.clear_session()
        logger.info("🔄 Models unloaded to free memory")

model_manager = ModelManager()

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = resnet18(weights=None)  # Load weights manually
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

# ========== PROCESSING FUNCTIONS ==========
def preprocess_audio(audio_bytes):
    try:
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
        return mfccs
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
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = model_manager.models['feature_extractor'].predict(batch[None, j, :])
        frame_mask[i, :length] = 1
    return frame_features, frame_mask

image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== FLASK ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        logger.error("No image file in request")
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if file.content_length > MAX_FILE_SIZE:
        logger.error(f"Image file too large: {file.content_length} bytes")
        return jsonify({"error": "File too large"}), 400
    try:
        model_manager.load_image_model()
        if model_manager.models['image_model'] is None:
            logger.error("Image model not loaded")
            return jsonify({"error": "Image model not available"}), 503
        image = Image.open(file.stream).convert('RGB')
        image = image_transform(image).unsqueeze(0).to(model_manager.models['image_device'])
        with torch.no_grad():
            output = model_manager.models['image_model'](image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, dim=1)
        result = "Deepfake" if pred[0].item() == 0 else "Real"
        model_manager.unload_models()
        return jsonify({
            "result": result,
            "confidence": round(confidence[0].item(), 2),
            "color": "red" if result == "Deepfake" else "green"
        })
    except Exception as e:
        logger.error(f"Image prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

@app.route('/predict/audio', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        logger.error("No audio file in request")
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files['audio']
    if file.content_length > MAX_FILE_SIZE:
        logger.error(f"Audio file too large: {file.content_length} bytes")
        return jsonify({"error": "File too large"}), 400
    try:
        model_manager.load_audio_model()
        if model_manager.models['audio_model'] is None:
            logger.error("Audio model not loaded")
            return jsonify({"error": "Audio model not available"}), 503
        mfccs = preprocess_audio(file.read())
        mfccs = np.expand_dims(mfccs, axis=0)
        interpreter = model_manager.models['audio_model']
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], mfccs)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0, predicted_class])
        model_manager.unload_models()
        return jsonify({
            "result": "Deepfake" if predicted_class == 1 else "Real",
            "confidence": round(confidence, 2),
            "color": "red" if predicted_class == 1 else "green"
        })
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500

@app.route('/predict/video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        logger.error("No video file in request")
        return jsonify({"error": "No video file provided"}), 400
    file = request.files['video']
    if file.content_length > MAX_FILE_SIZE:
        logger.error(f"Video file too large: {file.content_length} bytes")
        return jsonify({"error": "File too large"}), 400
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        logger.error(f"Unsupported video format: {file.filename}")
        return jsonify({"error": "Unsupported video format"}), 400
    try:
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)
        try:
            model_manager.load_video_models()
            if model_manager.models['video_model'] is None:
                logger.error("Video model not loaded")
                return jsonify({
                    "result": "Real",
                    "confidence": 0.5,
                    "color": "green",
                    "warning": "Video model not available - using fallback"
                }), 503
            frames = load_video(temp_path)
            frame_features, frame_mask = prepare_single_video(frames)
            interpreter = model_manager.models['video_model']
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], frame_features)
            interpreter.invoke()
            prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
            prediction = float(prediction.item())
            predicted_label = "Deepfake" if prediction >= 0.5 else "Real"
            model_manager.unload_models()
            return jsonify({
                "result": predicted_label,
                "confidence": round(prediction if predicted_label == "Deepfake" else 1 - prediction, 2),
                "color": "red" if predicted_label == "Deepfake" else "green"
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        logger.error(f"Video prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "OK",
        "models_loaded": {
            "audio": model_manager.models['audio_model'] is not None,
            "image": model_manager.models['image_model'] is not None,
            "video": model_manager.models['video_model'] is not None
        }
    })

if __name__ == '__main__':
    logger.info(f"🚀 Starting Deepfake Detection App on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)