import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Function to split the text into chunks with overlap for context preservation
def split_text_with_overlap(text, chunk_size=300, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Sample PDF file path
pdf_path = "C:\\Users\\USER\\Desktop\\SIH\\LOGO DETECTION.pdf"

# Step 1: Extract text from the PDF
text = extract_text_from_pdf(pdf_path)

# Step 2: Split the text into overlapping chunks
text_chunks = split_text_with_overlap(text)

# Step 3: Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Generate embeddings for each chunk of text
embeddings = model.encode(text_chunks, normalize_embeddings=True)

# Step 5: Use FAISS with inner product (cosine similarity) for efficient search
dimension = embeddings.shape[1]  # Embedding dimensions
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings))

# Now, the index is ready to perform similarity searches
def retrieve_answer(query, text_chunks, model, index, top_k=1):
    # Encode query and normalize
    query_embedding = model.encode([query], normalize_embeddings=True)
    # Perform similarity search
    D, I = index.search(np.array(query_embedding), k=top_k)
    print("Distances:", D)
    print("Indices:", I)
    # Return the top_k relevant chunks
    return [text_chunks[i] for i in I[0]]

query = "references"  # Sample query

# Retrieve the top 3 most relevant text chunks for the query
answers = retrieve_answer(query, text_chunks, model, index)

# Print the answers (the relevant text chunks from the PDF)
for i, answer in enumerate(answers):
    print(f"Answer {i + 1}: {answer}")  
    combined_answers = "\n".join(answers)
    lines = combined_answers.split("\n")  # Split all answers by newline
    for line in lines:
        print(line) 
        print()
