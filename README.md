# RAG-IT

A Python library for generating embeddings from various file types (text, PDFs, DOCX, images, and videos) for Retrieval-Augmented Generation (RAG) workflows. RAG-IT processes files, generates embeddings using state-of-the-art models, and saves them as JSON files for easy integration into RAG pipelines.

## Features

- **Supported File Types**:
  - **Text**: `.txt`
  - **Documents**: `.pdf`, `.docx`
  - **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`
  - **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
- **Embedding Models**:
  - Text: `sentence-transformers` (default: `all-MiniLM-L6-v2`)
  - Images/Videos: `CLIP` (default: `openai/clip-vit-base-patch32`)
- **Batch Processing**: Process multiple files in one call with `ragify_batch`.
- **Customizable Output**: Save embeddings to a specified directory.
- **Robust Error Handling**: Validates file existence, size, and content.
- **Verbose Logging**: Detailed logs for debugging and monitoring.
- **GPU Support**: Automatically uses CUDA if available.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TheJoshCode/rag-it.git
   cd ragit
   ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
# Usage
Basic Example
```bash
from ragit import RagIt

# Initialize RAG-IT
ragit = RagIt()

# Process individual files
text_output = ragit.ragit_text("example.txt", output_dir="embeddings")
pdf_output = ragit.ragit_pdf("example.pdf", output_dir="embeddings")
docx_output = ragit.ragit_docx("example.docx", output_dir="embeddings")
image_output = ragit.ragit_image("example.jpg", output_dir="embeddings")
video_output = ragit.ragit_video("example.mp4", max_frames=5, output_dir="embeddings")

print(f"Outputs: {text_output}, {pdf_output}, {docx_output}, {image_output}, {video_output}")

# Batch Processing
files = ["doc1.txt", "doc2.pdf", "img.jpg", "vid.mp4"]
outputs = ragit.ragit_batch(files, output_dir="embeddings", max_frames=5)
print(f"Batch outputs: {outputs}")

```
# Output Format
Each processed file generates a <filename>_embeddings.json file, e.g.:
```bash
{
    "file_path": "example.jpg",
    "embedding": [0.123, -0.456, ..., 0.789],
    "timestamp": "2025-05-29T15:42:00.123456",
    "embedding_shape": [512]
}
```
# Customization
```bash
# Use different models and max file size
ragit = RagIt(
    text_model_name="all-mpnet-base-v2",
    clip_model_name="openai/clip-vit-large-patch14",
    max_file_size_mb=200
)
```

# Notes
- Performance: Use lightweight models for speed or swap to larger models for better embeddings.

- Output Directory: If output_dir is not specified, JSON files are saved alongside input files.

- Extensibility: Add support for more file types by updating supported_extensions in ragify_batch.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs, features, or improvements.
License
MIT License. See LICENSE for details.

