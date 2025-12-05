import os
import time
import random
import requests
import base64
import json
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
# ---             GEMINI API UTILITIES               ---
# --- ================================================ ---

# NOTE: The API key will be injected by the environment.
API_KEY = "" 
# Using the powerful Gemini Pro model for a "stronger" analysis.
GEMINI_MODEL = "gemini-2.5-pro-preview-09-2025"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

def exponential_backoff_call(payload):
    """Handles API call with exponential backoff."""
    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raises an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            # Check for rate limits or server errors before retrying
            status_code = response.status_code if 'response' in locals() else 'Unknown'
            if attempt < MAX_RETRIES - 1 and (status_code in [429, 500, 503] or status_code == 'Unknown'):
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"API Error ({status_code}): {e}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                raise

def analyze_with_gemini(file_path):
    """
    Uses the Gemini API to analyze the file (image or video) for deepfake artifacts.
    """
    print(f"Starting Gemini analysis for: {file_path} using {GEMINI_MODEL}...")
    
    # 1. Read file and encode to base64, and check mime type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if not mime_type or not (mime_type.startswith('video/') or mime_type.startswith('image/')):
        raise ValueError("File is not a recognized image or video type.")

    with open(file_path, 'rb') as media_file:
        media_bytes = media_file.read()
    media_base64 = base64.b64encode(media_bytes).decode('utf-8')

    # 2. Define structured output schema
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "is_deepfake": { "type": "BOOLEAN", "description": "True if the media appears to be manipulated or computer-generated, False if it appears authentic and genuine." },
            "confidence_level": { "type": "STRING", "description": "A percentage string (e.g., '99.90%') representing the model's confidence in its assessment (based on visual cues only)." },
            "reasoning": { "type": "STRING", "description": "A concise explanation of the visual evidence found (or not found) that leads to the conclusion. For images, focus on forensic details like pixel inconsistencies or compression artifacts. For video, focus on motion and sync issues." }
        },
        "required": ["is_deepfake", "confidence_level", "reasoning"]
    }

    # 3. Construct payload
    is_image = mime_type.startswith('image/')
    
    # --- UPDATED SYSTEM PROMPT FOR BALANCE AND ACCURACY ---
    system_prompt = (
        "You are a professional media forensic analyst. Your task is to accurately assess whether the uploaded media "
        "is an authentic image/video or if it contains evidence of sophisticated digital manipulation, such as a deepfake. "
        "Acknowledge that compression artifacts from social media or low-light conditions can mimic manipulation, but focus only "
        "on clear and compelling evidence of digital alteration, especially on human faces. "
        "For still images, look for the following: "
        "1. **Inconsistent Lighting/Shadows:** Clear differences between the subject's lighting and the background's lighting. "
        "2. **Edge Artifacts:** Sharp, pixelated, or unnatural blending lines where a face or object meets the background. "
        "3. **Facial Textures:** Unusually smooth, plastic-like skin texture, or distorted features like ears, teeth, or hands. "
        "4. **Perspective Mismatch:** The subject's head is angled or sized inconsistently with the body or environment. "
        "Only mark as a deepfake if the evidence is strong. Respond strictly with the required JSON schema."
    )
    user_prompt = (
        f"Perform a balanced forensic analysis of the attached {'image' if is_image else 'video'} for deepfake artifacts. "
        f"Based *only* on the visual evidence and the criteria provided, is the media a clear deepfake or is it authentic?"
        "Provide a detailed reasoning and a confidence score based on the thorough visual analysis."
    )
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": media_base64
                        }
                    }
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }

    # 4. Call the API
    api_response = exponential_backoff_call(payload)
    
    # 5. Extract and parse result
    try:
        # Attempt to get the JSON text
        json_text = api_response['candidates'][0]['content']['parts'][0]['text']
        
        # ATTEMPT TO PARSE JSON SAFELY
        try:
            parsed_result = json.loads(json_text)
        except json.JSONDecodeError as json_e:
            # If JSON parsing fails, the model likely returned an error message in plain text.
            error_message = f"Gemini API returned non-JSON text: {json_text[:200]}..."
            print(f"Error: JSON parsing failed. {json_e}. {error_message}")
            raise ValueError(f"Analysis failed: Gemini output was malformed. Message: {json_text[:50]}...")

        # If parsing succeeds, continue extracting data
        is_fake = parsed_result.get("is_deepfake", False)
        confidence_str = parsed_result.get("confidence_level", "50.00%").replace('%', '')
        confidence = f"{float(confidence_str):.2f}"
        
        reasoning = parsed_result.get("reasoning", "Analysis failed to provide a detailed reason.")
        
        message = f"Gemini Analysis Complete: {'Deepfake Detected' if is_fake else 'Appears Authentic'}. Reason: {reasoning}"

        return {
            "is_deepfake": is_fake,
            "confidence": confidence,
            "message": message,
            "filename": os.path.basename(file_path)
        }
        
    except (KeyError, IndexError, ValueError) as e:
        # Catch errors if the JSON structure is missing keys, index is out of range, or the custom ValueError raised above
        print(f"Error processing Gemini response structure: {e}. Raw response: {json.dumps(api_response, indent=2)}")
        
        # Check for a specific 'promptFeedback' error often associated with video processing issues
        safety_error = api_response.get('promptFeedback', {}).get('blockReasonMessage')
        if safety_error:
             # Provide a more specific error message to the user if a safety or video processing error occurred.
             return {
                "is_deepfake": False,
                "confidence": "50.00",
                "message": f"Video analysis error: {safety_error}",
                "filename": os.path.basename(file_path)
             }


        return {
            "is_deepfake": False,
            "confidence": "50.00",
            "message": "API analysis failed to return structured data. Check the console for full error logs.",
            "filename": os.path.basename(file_path)
        }

# --- ================================================ ---
# ---                 FLASK API ROUTES               ---
# --- ================================================ ---

@app.route('/')
def serve_index():
    # This route serves the frontend file (index.html)
    return send_from_directory('.', 'index.html') 

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
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            analysis_result = analyze_with_gemini(filepath)

            return jsonify(analysis_result)

        except requests.exceptions.HTTPError as e:
            error_msg = f"API HTTP Error: {e.response.status_code} - {e.response.text}"
            print(error_msg)
            return jsonify({"message": f"API Error: Failed to analyze file. Status Code: {e.response.status_code}"}), 500
        except Exception as e:
            error_msg = f"General Error during analysis: {e}"
            print(error_msg)
            return jsonify({"message": f"Failed to analyze file: {e}"}), 500
        
        finally:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                
# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server... Open http://127.0.0.1:5001 in your browser.")
    app.run(debug=True, host='0.0.0.0', port=5001)