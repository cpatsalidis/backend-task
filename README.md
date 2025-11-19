# Facial Region Processing

A facial image processing pipeline that detects, segments, and labels facial regions with automatic tilt correction and cropping.

## Installation

### Local Development

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv

2. **Install dependencies:**
   ```bash
   pip install -r requirements-local.txt
   ```

### Docker Setup

No additional installation needed - Docker will handle everything during build.

## Running

### Local Execution

```bash
cd src
python main.py
```

**Input files** (configured in `src/config.py`):
- `Backend_Engineer/original_image.png` - Original facial image
- `Backend_Engineer/segmentation_map.png` - Segmentation map
- `Backend_Engineer/landmarks.txt` - Facial landmarks data

**Output:**
- `Backend_Engineer/result_final.png` - Processed image with labeled regions

### API Service (Docker)

1. **Build and start the service:**
   ```bash
   docker-compose up --build
   ```

2. **The API will be available at:**
   - API: `http://localhost:8000`
   - Documentation: `http://localhost:8000/api/docs`
   - Health Check: `http://localhost:8000/health`

3. **POST request at:**
   'http://localhost:8000/api/v1/frontal/crop/submit'

### API Service (Local - Development)

```bash
# Install dependencies (if not already done)
pip install -r requirements-local.txt

# Run the API (from project root)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
