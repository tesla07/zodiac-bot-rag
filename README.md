# ✨ Zodiac RAG Chatbot

A sophisticated RAG (Retrieval-Augmented Generation) chatbot that channels the mystical wisdom of Linda Goodman's astrology books: "Sun Signs" and "Love Signs". Built with FastAPI, OpenAI, and FAISS for intelligent astrological insights.

## 🌟 Features

- **PDF Processing**: Intelligent parsing of Linda Goodman's astrology books using PyMuPDF and unstructured
- **Adaptive Chunking**: Semantic similarity-based text chunking for optimal context retrieval
- **OpenAI Integration**: GPT-4 for generating responses in Linda Goodman's mystical tone
- **FAISS Vector Store**: High-performance similarity search for relevant content
- **FastAPI Backend**: RESTful API with comprehensive endpoints
- **Streamlit Frontend**: Beautiful, mystical-themed chat interface
- **Source Attribution**: Shows which parts of the books were used for each response

## 🏗️ Architecture

```
zodiac-rag/
├── parser_chunker.py    # PDF parsing and adaptive chunking
├── embed_index.py       # OpenAI embeddings and FAISS indexing
├── rag_engine.py        # RAG pipeline and response generation
├── app.py              # FastAPI backend application
├── streamlit_app.py    # Streamlit frontend interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd zodiac-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002
FAISS_INDEX_PATH=./data/faiss_index
UPLOAD_DIR=./uploads
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

### 3. Start the Backend

```bash
# Start FastAPI server
python app.py
```

The API will be available at `http://localhost:8000`

### 4. Start the Frontend (Optional)

```bash
# In a new terminal
streamlit run streamlit_app.py
```

The Streamlit interface will be available at `http://localhost:8501`

## 📚 Usage

### 1. Upload PDFs

First, upload Linda Goodman's books:

- **Via API**: `POST /upload` with PDF file
- **Via Streamlit**: Use the file uploader in the sidebar

### 2. Ask Questions

Ask about:
- Zodiac sign personalities and traits
- Love compatibility between signs
- Linda Goodman's mystical insights
- Astrological relationships and dynamics

### 3. Example Queries

- "Tell me about Aries personality traits"
- "What is the compatibility between Leo and Aquarius?"
- "How does Linda Goodman describe Cancer in relationships?"
- "What are the mystical qualities of Pisces?"

## 🔧 API Endpoints

### Core Endpoints

- `GET /health` - Health check and system status
- `POST /upload` - Upload and index PDF files
- `POST /ask` - Ask questions to the RAG chatbot
- `GET /suggestions` - Get suggested questions
- `GET /stats` - System statistics
- `DELETE /reset` - Reset the entire index

### Example API Usage

```python
import requests

# Upload a PDF
with open('sun_signs.pdf', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})

# Ask a question
response = requests.post('http://localhost:8000/ask', data={
    'query': 'Tell me about Aries personality',
    'top_k': 5
})

print(response.json())
```

## 🎨 Features in Detail

### Adaptive Chunking

The system uses semantic similarity to merge adjacent paragraphs, creating contextually coherent chunks that preserve the mystical flow of Linda Goodman's writing.

### Mystical Tone Generation

Responses are generated in Linda Goodman's distinctive style:
- Poetic and mystical language
- Cosmic metaphors and spiritual insights
- Warm, wise, and spiritually attuned tone
- Practical wisdom wrapped in mystical understanding

### Source Attribution

Each response includes:
- Source file information
- Similarity scores
- Text previews from relevant sections
- Number of chunks retrieved

## 🔍 Technical Details

### Dependencies

- **FastAPI**: Modern web framework for APIs
- **PyMuPDF**: High-performance PDF processing
- **unstructured**: Fallback PDF parser
- **OpenAI**: GPT-4 and embedding models
- **FAISS**: Efficient similarity search
- **Streamlit**: Interactive web interface
- **scikit-learn**: Semantic similarity calculations

### Performance Optimizations

- Batch processing for embeddings
- Efficient FAISS indexing
- Semantic similarity-based chunking
- Configurable chunk sizes and overlap

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | GPT model to use | `gpt-4` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-ada-002` |
| `FAISS_INDEX_PATH` | Path for FAISS index | `./data/faiss_index` |
| `UPLOAD_DIR` | PDF upload directory | `./uploads` |
| `CHUNK_SIZE` | Maximum chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `TOP_K` | Default retrieval count | `5` |

### Tuning Parameters

- **Chunk Size**: Larger chunks preserve more context but may reduce precision
- **Chunk Overlap**: Higher overlap improves continuity but increases storage
- **Top-K**: More retrieved chunks provide richer context but may include irrelevant information

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Not Set**
   - Ensure `OPENAI_API_KEY` is set in your `.env` file
   - Verify the key is valid and has sufficient credits

2. **PDF Processing Errors**
   - Ensure PDFs are not password-protected
   - Try different PDF files if parsing fails
   - Check file permissions

3. **Memory Issues**
   - Reduce `CHUNK_SIZE` for large documents
   - Process PDFs one at a time
   - Monitor system memory usage

4. **API Connection Issues**
   - Verify the FastAPI server is running
   - Check firewall settings
   - Ensure correct port configuration

### Debug Mode

Enable debug logging by setting the log level in `app.py`:

```python
uvicorn.run("app:app", log_level="debug")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Linda Goodman for her mystical astrological wisdom
- OpenAI for providing the GPT and embedding models
- The open-source community for the amazing tools used in this project

## 🌟 Future Enhancements

- [ ] Support for more astrology books
- [ ] Birth chart analysis capabilities
- [ ] Real-time horoscope generation
- [ ] Multi-language support
- [ ] Advanced visualization features
- [ ] Integration with astrological APIs

---

**May the stars guide your journey! ✨** 