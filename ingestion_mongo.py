import os
import pymongo
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pypdf
from bs4 import BeautifulSoup
import markdown
import logging
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.operations import SearchIndexModel
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def get_config():
    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI not found in environment variables")
    return MONGODB_URI

# Initialize MongoDB with error handling
try:
    MONGODB_URI = get_config()
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    # Test connection
    client.admin.command('ping')
    db = client['hcl']
    collection = db['hcl-web']
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    raise

# Initialize Sentence Transformer model
def initialize_embeddings_model():
    try:
        model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
        #model = SentenceTransformer('all-mpnet-base-v2')
        logger.info("Successfully initialized Sentence Transformer model")
        return model
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        raise

def get_embeddings(text: str, model) -> List[float]:
    try:
        return model.encode(text).tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def create_chunker():
    return RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        length_function=len,
    )

def get_file_metadata(file_path: str) -> Dict:
    return {
        "filename": os.path.basename(file_path),
        "file_type": os.path.splitext(file_path)[1][1:],
        "file_size": os.path.getsize(file_path),
        "file_path": file_path
    }

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text from PDF and return chunks with metadata"""
    chunks = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        # Create chunks using the text splitter
        chunker = create_chunker()
        text_chunks = chunker.split_text(full_text)
        
        # Get file metadata
        metadata = get_file_metadata(pdf_path)
        
        # Create chunks with metadata
        for i, chunk in enumerate(text_chunks):
            chunk_id = hashlib.md5(f"{pdf_path}_{i}".encode()).hexdigest()
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_text": chunk,
                "chunk_index": i,
                "embedding": None,  # Will be filled later
                **metadata
            })
            
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
    
    return chunks

def verify_vector_index(collection, VECTOR_DIMENSION=768):
    """Verify or create vector search index"""
    try:
        # Check if index already exists
        existing_indexes = collection.list_search_indexes()
        for index in existing_indexes:
            if index["name"] == "hcl_vector_index":
                logger.info("Vector search index exists")
                return True

        # Create new index
        search_index = SearchIndexModel(
            definition={
                "fields": [
                    {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": VECTOR_DIMENSION,
                    "similarity": "cosine"
                 }]
            },
            name="hcl_vector_index",
            type="vectorSearch"
        )

        # Create the search index
        collection.create_search_index(model=search_index)
        logger.info("Successfully created vector search index")
        return True
        
    except Exception as e:
        logger.error(f"Error with vector search index: {str(e)}")
        return False

def process_markdown(md_path: str) -> List[Dict]:
    """Process markdown file and return chunks with metadata"""
    chunks = []
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert markdown to plain text
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        # Create chunks using the text splitter
        chunker = create_chunker()
        text_chunks = chunker.split_text(text)
        
        # Get file metadata
        metadata = get_file_metadata(md_path)
        
        # Create chunks with metadata
        for i, chunk in enumerate(text_chunks):
            chunk_id = hashlib.md5(f"{md_path}_{i}".encode()).hexdigest()
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_text": chunk,
                "chunk_index": i,
                "embedding": None,  # Will be filled later
                **metadata
            })
            
    except Exception as e:
        logger.error(f"Error processing markdown {md_path}: {str(e)}")
    
    return chunks

def ingest_documents(pdf_dir: str = "downloaded_pdfs", md_dir: str = "scraped_content"):
    """Main function to ingest documents"""
    try:
        # Verify vector search index exists or create it
        if not verify_vector_index(collection):
            logger.error("Failed to verify/create vector search index. Aborting ingestion.")
            return

        # Initialize embedding model
        model = initialize_embeddings_model()
        
        all_chunks = []
        
        # Process PDFs
        if os.path.exists(pdf_dir):
            for filename in os.listdir(pdf_dir):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(pdf_dir, filename)
                    chunks = extract_text_from_pdf(pdf_path)
                    all_chunks.extend(chunks)
                    logger.info(f"Processed PDF: {filename} - {len(chunks)} chunks")
        else:
            logger.warning(f"PDF directory not found: {pdf_dir}")
        
        # Process Markdown files
        if os.path.exists(md_dir):
            for filename in os.listdir(md_dir):
                if filename.endswith('.md'):
                    md_path = os.path.join(md_dir, filename)
                    chunks = process_markdown(md_path)
                    all_chunks.extend(chunks)
                    logger.info(f"Processed Markdown: {filename} - {len(chunks)} chunks")
        else:
            logger.warning(f"Markdown directory not found: {md_dir}")
        
        # Generate embeddings and insert into MongoDB
        for chunk in all_chunks:
            try:
                chunk["embedding"] = get_embeddings(chunk["chunk_text"], model)
                result = collection.insert_one(chunk)
                logger.info(f"Inserted chunk {chunk['chunk_id']}: {result.inserted_id}")
            except Exception as e:
                logger.error(f"Error processing chunk {chunk['chunk_id']}: {str(e)}")
                continue
        
        logger.info(f"Total chunks ingested: {len(all_chunks)}")
        
    except Exception as e:
        logger.error(f"Error in ingest_documents: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        ingest_documents()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
