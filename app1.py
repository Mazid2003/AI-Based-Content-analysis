from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from logo_detection_query_model import extract_text_from_pdf, split_text_with_overlap, retrieve_answer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to process PDF and create FAISS index
def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text_with_overlap(text)
    embeddings = model.encode(text_chunks, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings))
    return text_chunks, index

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle PDF upload
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process PDF and create FAISS index
            text_chunks, index = process_pdf(filepath)
            query = request.form['query']
            answers = retrieve_answer(query, text_chunks, model, index)

            return render_template('results.html', query=query, answers=answers)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
