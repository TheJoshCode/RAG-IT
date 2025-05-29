import logging
import json
import numpy as np
import fitz
from PIL import Image
from moviepy.editor import VideoFileClip
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from datetime import datetime
from typing import List, Dict, Optional
import docx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAG-IT:
    def __init__(
        self,
        text_model_name: str = 'all-MiniLM-L6-v2',
        clip_model_name: str = 'openai/clip-vit-base-patch32',
        max_file_size_mb: int = 100
    ):
        logger.info("Initializing RAG-IT...")
        self.max_file_size_mb = max_file_size_mb
        try:
            self.text_model = SentenceTransformer(text_model_name)
            logger.info(f"Text embedding model {text_model_name} loaded successfully.")
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            logger.info(f"CLIP model {clip_model_name} loaded successfully.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.clip_model.to(self.device)
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _validate_file(self, file_path: str) -> None:
        logger.info(f"Validating file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File {file_path} exceeds max size of {self.max_file_size_mb} MB")
        logger.info(f"File {file_path} validated successfully (size: {file_size_mb:.2f} MB)")

    def _extract_text_from_pdf(self, file_path: str) -> str:
        logger.info(f"Extracting text from PDF: {file_path}")
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if not text.strip():
                raise ValueError(f"No text extracted from {file_path}")
            logger.info(f"Extracted {len(text)} characters from {file_path}")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        logger.info(f"Extracting text from DOCX: {file_path}")
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            if not text.strip():
                raise ValueError(f"No text extracted from {file_path}")
            logger.info(f"Extracted {len(text)} characters from {file_path}")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    def _extract_frames_from_video(self, file_path: str, max_frames: int = 10) -> List[Image.Image]:
        logger.info(f"Extracting frames from video: {file_path}")
        try:
            clip = VideoFileClip(file_path)
            duration = clip.duration
            frames = []
            step = max(1, duration / max_frames)
            for t in np.arange(0, duration, step):
                frame = clip.get_frame(t)
                pil_image = Image.fromarray(frame)
                frames.append(pil_image)
            clip.close()
            if not frames:
                raise ValueError(f"No frames extracted from {file_path}")
            logger.info(f"Extracted {len(frames)} frames from {file_path}")
            return frames
        except Exception as e:
            logger.error(f"Error extracting frames from video {file_path}: {e}")
            raise
    
    def _generate_text_embedding(self, text: str) -> np.ndarray:
        logger.info(f"Generating embedding for text (length: {len(text)} characters)")
        try:
            embedding = self.text_model.encode(text, convert_to_numpy=True)
            logger.info(f"Text embedding generated with shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise
    
    def _generate_image_embedding(self, image: Image.Image) -> np.ndarray:
        logger.info("Generating embedding for image")
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs).cpu().numpy()
            logger.info(f"Image embedding generated with shape: {embedding.shape}")
            return embedding.flatten()
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            raise
    
    def _save_embedding(self, file_path: str, embedding: np.ndarray, output_dir: Optional[str] = None) -> str:
        logger.info(f"Saving embedding for {file_path}")
        try:
            output_file = os.path.splitext(file_path)[0] + "_embeddings.json"
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, os.path.basename(output_file))
            embedding_data = {
                "file_path": file_path,
                "embedding": embedding.tolist(),
                "timestamp": datetime.now().isoformat(),
                "embedding_shape": list(embedding.shape)
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embedding_data, f, indent=4)
            logger.info(f"Embedding saved to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error saving embedding for {file_path}: {e}")
            raise
    
    def ragify_text(self, file_path: str, output_dir: Optional[str] = None) -> str:
        logger.info(f"Ragifying text file: {file_path}")
        try:
            self._validate_file(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if not text.strip():
                raise ValueError(f"No text content in {file_path}")
            embedding = self._generate_text_embedding(text)
            return self._save_embedding(file_path, embedding, output_dir)
        except Exception as e:
            logger.error(f"Error ragifying text file {file_path}: {e}")
            raise
    
    def ragify_pdf(self, file_path: str, output_dir: Optional[str] = None) -> str:
        logger.info(f"Ragifying PDF file: {file_path}")
        try:
            self._validate_file(file_path)
            text = self._extract_text_from_pdf(file_path)
            embedding = self._generate_text_embedding(text)
            return self._save_embedding(file_path, embedding, output_dir)
        except Exception as e:
            logger.error(f"Error ragifying PDF file {file_path}: {e}")
            raise
    
    def ragify_docx(self, file_path: str, output_dir: Optional[str] = None) -> str:
        logger.info(f"Ragifying DOCX file: {file_path}")
        try:
            self._validate_file(file_path)
            text = self._extract_text_from_docx(file_path)
            embedding = self._generate_text_embedding(text)
            return self._save_embedding(file_path, embedding, output_dir)
        except Exception as e:
            logger.error(f"Error ragifying DOCX file {file_path}: {e}")
            raise
    
    def ragify_image(self, file_path: str, output_dir: Optional[str] = None) -> str:
        logger.info(f"Ragifying image file: {file_path}")
        try:
            self._validate_file(file_path)
            image = Image.open(file_path).convert('RGB')
            embedding = self._generate_image_embedding(image)
            return self._save_embedding(file_path, embedding, output_dir)
        except Exception as e:
            logger.error(f"Error ragifying image file {file_path}: {e}")
            raise
    
    def ragify_video(self, file_path: str, max_frames: int = 10, output_dir: Optional[str] = None) -> str:
        logger.info(f"Ragifying video file: {file_path}")
        try:
            self._validate_file(file_path)
            frames = self._extract_frames_from_video(file_path, max_frames)
            embeddings = [self._generate_image_embedding(frame) for frame in frames]
            avg_embedding = np.mean(embeddings, axis=0)
            return self._save_embedding(file_path, avg_embedding, output_dir)
        except Exception as e:
            logger.error(f"Error ragifying video file {file_path}: {e}")
            raise
    
    def ragify_batch(self, file_paths: List[str], output_dir: Optional[str] = None, max_frames: int = 10) -> List[str]:
        logger.info(f"Processing batch of {len(file_paths)} files")
        output_files = []
        supported_extensions = {
            'text': ['.txt'],
            'pdf': ['.pdf'],
            'docx': ['.docx'],
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        }
        
        for file_path in file_paths:
            try:
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in supported_extensions['text']:
                    output = self.ragify_text(file_path, output_dir)
                elif file_ext in supported_extensions['pdf']:
                    output = self.ragify_pdf(file_path, output_dir)
                elif file_ext in supported_extensions['docx']:
                    output = self.ragify_docx(file_path, output_dir)
                elif file_ext in supported_extensions['image']:
                    output = self.ragify_image(file_path, output_dir)
                elif file_ext in supported_extensions['video']:
                    output = self.ragify_video(file_path, max_frames, output_dir)
                else:
                    logger.warning(f"Unsupported file extension {file_ext} for {file_path}")
                    continue
                output_files.append(output)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        logger.info(f"Batch processing completed: {len(output_files)} files processed successfully")
        return output_files