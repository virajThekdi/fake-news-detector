from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from utils.predict import predict_fake_news, extract_text_from_pdf

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')  # Serves the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        # Handle PDF file upload
        pdf_file = request.files['file']
        if pdf_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            extracted_text = extract_text_from_pdf(pdf_file)
            if not extracted_text.strip():
                return jsonify({'error': 'No text found in PDF'}), 400
            prediction = predict_fake_news(extracted_text)
            return jsonify({'prediction': prediction, 'source': 'PDF'})
        except Exception as e:
            return jsonify({'error': f'PDF processing failed: {str(e)}'}), 500

    else:
        # Handle direct text input
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        prediction = predict_fake_news(data['text'])
        return jsonify({'prediction': prediction, 'source': 'Text'})

if __name__ == '__main__':
    app.run(debug=True)
