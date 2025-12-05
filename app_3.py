import os
import time
import random
import mimetypes
from flask import Flask, request, jsonify, send_from_directory

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# --- ================================================ ---
# ---             MOCK AI ANALYSIS FUNCTION            ---
# --- ================================================ ---

def mock_deepfake_analysis(file_path):
    """
    *** .
    
    """
    print(f"Starting MOCK analysis for: {file_path}...")
    
    # 1. Simulate the time it takes for an AI model to run
    time.sleep(2) 
    
    # 2. Generate a random, fake result
    is_fake = random.choice([True, False])
    confidence = random.uniform(85.0, 99.9) # Give a high confidence
    
    if is_fake:
        message = "Analysis complete: High probability of being a deepfake."
        return {
            "is_deepfake": True,
            "confidence": f"{confidence:.2f}",
            "message": message,
            "filename": os.path.basename(file_path)
        }
    else:
        message = "Analysis complete: Appears to be authentic."
        return {
            "is_deepfake": False,
            "confidence": f"{confidence:.2f}",
            "message": message,
            "filename": os.path.basename(file_path)
        }

# --- ================================================ ---
# ---                 FLASK API ROUTES               ---
# --- ================================================ ---

@app.route('/')
def serve_index():
    # --- THIS IS THE FIX ---
    # It points to your 'index_2.html' file
    return send_from_directory('.', 'index_2.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    
    if file:
        filepath = None
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # We are now calling the free 'mock_deepfake_analysis'
            # function for ALL file types.
            analysis_result = mock_deepfake_analysis(filepath)

            return jsonify(analysis_result)

        except Exception as e:
            print(f"Error during analysis: {e}")
            return jsonify({"message": f"Failed to analyze file: {e}"}), 500
        
        finally:
            # Clean up the uploaded file
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                
# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server... Open http://127.0.0.1:5001 in your browser.")
    # You can set debug=True again since we aren't loading a heavy model
    app.run(debug=True, host='0.0.0.0', port=5001)

