
# PDF Processor Pro

A Streamlit application for processing and comparing PDFs using Unstructured.io and OpenAI.

## Features

- Upload and compare two PDFs
- Highlight differences between documents
- Ask questions about PDF content using AI
- Export differences to Excel

## Installation

1. **Clone this repository**:
   ```bash
   git clone <repository-url>
   cd pdfparser
   ```

2. **Set up the environment using `uv`**:
   - Ensure you have `uv` installed. If not, install it with:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - Create a virtual environment and install dependencies:
     ```bash
     uv venv
     source .venv/bin/activate
     uv pip install -e .
     ```
   - Note: These commands are optimized for Zsh (the default shell on macOS) and work seamlessly in iTerm2 on a MacBook M4.

3. **Configure environment variables**:
   - Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Add your API keys for Unstructured.io and OpenAI to the `.env` file.

## Usage

Run the application with:
```bash
streamlit run pdf_processor.py
```

Then:
1. Upload two PDFs in the "Upload" tab
2. View differences in the "Comparison" tab
3. Ask questions in the "Chat" tab

## Apple Silicon (M1/M2/M3/M4) Mac Compatibility

If you encounter errors related to Metal Performance Shaders when running on Apple Silicon Macs (e.g., MacBook M4), use one of these solutions:

### Option 1: Install CPU-only PyTorch
```bash
uv pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### Option 2: Set environment variables before running
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export CUDA_VISIBLE_DEVICES=""
streamlit run pdf_processor.py
```

### Option 3: Add to .env file
Add these lines to your `.env` file:
```
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
CUDA_VISIBLE_DEVICES=
```

## Troubleshooting

### Metal Performance Shaders Error
If you see an error like:
```
failed assertion `Error: MLIR pass manager failed'
```
This is related to GPU acceleration on Apple Silicon Macs. Use the compatibility options above to resolve it.

### Missing ScriptRunContext Warnings
Warnings about "missing ScriptRunContext" are normal when running scripts directly. They can be safely ignored, as noted in the warning messages themselves.

## License


