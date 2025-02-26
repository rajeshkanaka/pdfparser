import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
from datetime import datetime

# Unstructured Serverless API imports
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
)

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Redis for advanced caching (assumes a local Redis instance)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def compute_file_hash(file_path: str) -> str:
    """
    Compute a SHA-256 hash of the file contents for caching purposes.

    Args:
        file_path (str): The path to the file to be hashed.

    Returns:
        str: A hexadecimal string representing the SHA-256 hash of the file.

    Raises:
        FileNotFoundError: If the file at `file_path` does not exist.
        IOError: If there is an error reading the file.
    """
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                data = f.read(65536)  # Read in 64KB chunks for efficiency
                if not data:
                    break
                hasher.update(data)
        return hasher.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

def process_pdf_single(pdf_path: str, max_retries: int = 2) -> Optional[List[Dict]]:
    """
    Process a single PDF using the Unstructured.io Serverless API with retry logic and fallback strategies.

    Args:
        pdf_path (str): The path to the PDF file to process.
        max_retries (int): Maximum number of retry attempts for failed processing.

    Returns:
        Optional[List[Dict]]: Structured data extracted from the PDF, or None if all retries fail.
    """
    file_hash = compute_file_hash(pdf_path)
    cache_key = f"pdf_cache:{file_hash}"

    # Check Redis cache first
    cached_data = redis_client.get(cache_key)
    if cached_data:
        logger.info(f"Cache hit for PDF: {pdf_path}")
        return json.loads(cached_data)

    output_dir = os.path.join(os.getcwd(), "structured-output")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{os.path.basename(pdf_path)}.json"
    output_path = os.path.join(output_dir, output_filename)

    attempt = 0
    while attempt < max_retries:
        try:
            pipeline = Pipeline.from_configs(
                context=ProcessorConfig(
                    request_timeout=300,
                    headers={"Accept": "application/json"},
                ),
                indexer_config=LocalIndexerConfig(
                    input_path=str(Path(pdf_path).parent),
                    recursive=False,
                    filename_filter=lambda x: x == pdf_path,
                ),
                downloader_config=LocalDownloaderConfig(
                    download_dir=output_dir,
                    preserve_structure=True,
                ),
                source_connection_config=LocalConnectionConfig(),
                partitioner_config=PartitionerConfig(
                    partition_by_api=True,
                    api_key=os.getenv("UNSTRUCTURED_API_KEY"),
                    partition_endpoint=os.getenv("UNSTRUCTURED_API_URL"),
                    strategy="hi_res",
                    additional_partition_args={
                        "split_pdf_page": True,
                        "split_pdf_allow_failed": True,
                        "split_pdf_concurrency_level": 15,
                        "chunking_strategy": "by_title",  # Semantic chunking
                        "max_characters": 1000,           # Word limit for chunks
                        "combine_text_under_n_chars": 200 # Combine small chunks
                    },
                ),
            )

            pipeline.run()
            if not os.path.exists(output_path):
                logger.warning(f"Output file not found for {pdf_path}, attempt {attempt + 1}")
                attempt += 1
                continue

            with open(output_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            # Validate output - check for empty or meaningless chunks
            if not chunks or all(len(chunk.get("text", "").strip()) == 0 for chunk in chunks):
                logger.warning(f"Semantically empty output for {pdf_path}, attempting fallback")
                attempt += 1
                continue

            # Cache the result in Redis with 24-hour expiration
            redis_client.setex(cache_key, 86400, json.dumps(chunks))
            logger.info(f"Successfully processed and cached PDF: {pdf_path}")
            return chunks

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}, attempt {attempt + 1}: {str(e)}")
            attempt += 1
            if attempt == max_retries:
                logger.error(f"Max retries reached for {pdf_path}, processing failed")
                return None

    return None

def process_pdfs_batch(pdf_paths: List[str], max_workers: int = 4) -> Dict[str, List[Dict]]:
    """
    Process multiple PDFs in parallel using a thread pool executor.

    Args:
        pdf_paths (List[str]): List of PDF file paths to process.
        max_workers (int): Maximum number of concurrent workers.

    Returns:
        Dict[str, List[Dict]]: Dictionary mapping PDF paths to their processed data.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {executor.submit(process_pdf_single, pdf_path): pdf_path for pdf_path in pdf_paths}
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                result = future.result()
                if result is None:
                    logger.error(f"Failed to process PDF: {pdf_path}")
                    results[pdf_path] = []
                else:
                    results[pdf_path] = result
                    logger.info(f"Successfully processed PDF in batch: {pdf_path}")
            except Exception as e:
                logger.error(f"Error in batch processing for {pdf_path}: {str(e)}")
                results[pdf_path] = []
    return results

if __name__ == "__main__":
    # Example usage (for testing purposes)
    logging.basicConfig(level=logging.INFO)
    pdf_path = "example.pdf"
    try:
        # Single PDF processing
        data = process_pdf_single(pdf_path)
        if data:
            print(f"Processed single PDF data: {data[:1]}")  # Print first item for brevity

        # Batch processing example
        pdf_list = [pdf_path, "example2.pdf"]  # Replace with actual PDFs for testing
        batch_results = process_pdfs_batch(pdf_list)
        for pdf, result in batch_results.items():
            print(f"Batch processed {pdf}: {result[:1] if result else '[]'}")
    except Exception as e:
        print(f"Error: {e}")