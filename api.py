from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Reject Oversized: Max file size set to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 

# 2. Reject Invalid Types: Audio formats strictly enforced
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# MAIN ROUTE
# ==========================================
@app.route('/detect-audio', methods=['POST'])
def detect_audio():
    # EDGE CASE 1: No file attached to the request
    if 'file' not in request.files:
        return jsonify({'error': 'Bad Request: No file attached'}), 400

    file = request.files['file']

    # EDGE CASE 2: Empty form submitted
    if file.filename == '':
        return jsonify({'error': 'Bad Request: No file selected'}), 400

    # REJECT INVALID FILE TYPE
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported Media Type: Only .mp3, .wav, .flac, or .m4a allowed'}), 415

    try:
        # ---------------------------------------------------------
        # INTEGRATION POINT: Spectrogram Pipeline & Model go here
        # ---------------------------------------------------------
        
        # Mock response for frontend development and Viva demonstration
        return jsonify({
            'status': 'success',
            'message': 'Audio received and validated successfully.',
            'filename': file.filename,
            'prediction': 'AI_Generated', 
            'confidence_score': 0.94
        }), 200

    except Exception as e:
        # REJECT MALFORMED: Catch processing crashes safely
        return jsonify({'error': f'Internal Server Error: Malformed file or processing failed. Details: {str(e)}'}), 500


# ==========================================
# ERROR HANDLERS
# ==========================================
@app.errorhandler(413)
def request_entity_too_large(error):
    # Sends a clean JSON response if the 5MB limit is breached
    return jsonify({'error': 'Payload Too Large: File exceeds the 5MB limit.'}), 413


if __name__ == '__main__':
    print("🚀 S.A.F.E. API is online and waiting for audio data...")
    app.run(debug=True, port=5000)