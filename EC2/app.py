from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf # NEW: Import TensorFlow

# Assuming these are imported from process.py or defined locally
from EC2.process import extract_mediapipe_features, standardize_sequence, CLASSES, MAX_SEQUENCE_LENGTH, NUM_FEATURES

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- NEW: Load the Deep Learning Model Globally ---
# IMPORTANT: Replace '/path/to/your/ksl_slr_model.keras' with the actual path
# to your trained Keras model file on your EC2 instance.
# Ensure the model file is accessible (e.g., in your project directory).
try:
    model = tf.keras.models.load_model('195_75.keras') # <<<--- UPDATE THIS PATH
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Re-compile is often good practice
    print("Deep learning model loaded successfully.")
    print(f"Model input shape: {model.input_shape}")
except Exception as e:
    model = None
    print(f"ERROR: Could not load the DL model. Prediction functionality will be disabled. Error: {e}")

@app.route('/extract_features', methods=['POST'])
def extract_and_return_features():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            print(f"Received video: {filepath}. Starting feature extraction...")
            keypoints = extract_mediapipe_features(filepath, display_video=False)

            if keypoints is None or keypoints.shape[0] == 0:
                os.remove(filepath)
                return jsonify({'error': f'Could not extract any features from video: {filename}.'}), 500

            print(f"Raw extracted features shape: {keypoints.shape}")

            standardized_keypoints = standardize_sequence(keypoints, max_len=MAX_SEQUENCE_LENGTH, num_features=NUM_FEATURES)
            final_features_array = np.expand_dims(standardized_keypoints, axis=0)

            print(f"Standardized and reshaped features shape: {final_features_array.shape}")

            # --- NEW: Perform Model Prediction ---
            predicted_class_name = "N/A"
            confidence = "N/A"
            raw_predictions = None

            if model:
                try:
                    # Model expects a batch of sequences: (1, MAX_SEQUENCE_LENGTH, NUM_FEATURES)
                    predictions = model.predict(final_features_array)
                    raw_predictions = predictions[0].tolist() # Convert to list for JSON
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    predicted_class_name = CLASSES[predicted_class_index]
                    confidence = float(np.max(predictions)) # Convert to float for JSON serializability

                    print(f"Prediction for {filename}: {predicted_class_name} with confidence {confidence:.4f}")
                except Exception as pred_e:
                    print(f"Error during model prediction: {pred_e}")
                    predicted_class_name = "Prediction Error"
                    confidence = "Error"
            else:
                print("Skipping prediction: DL model not loaded.")
            # --- END NEW: Model Prediction ---

            os.remove(filepath) # Clean up the uploaded file

            response_data = {
                'message': 'Features extracted, standardized, and prediction performed successfully',
                'video_filename': filename,
                'output_shape_for_model': list(final_features_array.shape),
                'predicted_class': predicted_class_name, # NEW
                'confidence': confidence,                 # NEW
                'raw_predictions': raw_predictions        # NEW: Optionally include raw probabilities
            }

            if request.args.get('include_features', 'false').lower() == 'true':
                response_data['features'] = final_features_array.tolist()
                print("Full features included in response due to 'include_features' parameter.")
            else:
                print("Full features omitted from response for brevity. Use ?include_features=true to get them.")

            return jsonify(response_data)

        except Exception as e:
            print(f"Error during feature extraction or processing: {e}")
            if os.path.exists(filepath):
                os.remove(filepath) # Ensure cleanup even on error
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    return jsonify({'error': 'Unknown error'}), 500

# --- Existing Healthcheck Endpoint ---
@app.route('/healthcheck')
def health_check():
    return render_template('healthcheck.html')

# --- Existing API endpoint to serve the video upload page ---
@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)