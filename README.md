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

### API Service (Local - Development)

```bash
# Install dependencies (if not already done)
pip install -r requirements-local.txt

# Run the API (from project root)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

The API uses an **asynchronous, non-blocking job queue system**. Jobs are submitted and processed in the background, allowing instant responses.

### Submit a Job

**Endpoint:** `POST /api/v1/frontal/crop/submit`

Submits a processing job and returns immediately with a job ID and status.

**Request Body:**
```json
{
  "image": "base64_encoded_image_string",
  "landmarks": [
    {"x": 100.5, "y": 150.2},
    {"x": 200.3, "y": 151.8}
  ],
  "segmentation_map": "base64_encoded_segmentation_map"
}
```

**Query Parameters:**
- `delay` (optional): Simulate processing delay in seconds (e.g., `?delay=20` for 20 seconds)

**Response (HTTP 202 Accepted):**
```json
{
  "id": 123,
  "status": "pending"
}
```

### Check Job Status

**Endpoint:** `GET /api/v1/frontal/crop/status/{job_id}`

Returns the current status of a job. When completed, includes the processing results.

**Response (Pending/Processing):**
```json
{
  "id": 123,
  "status": "processing"
}
```

**Response (Completed):**
```json
{
  "id": 123,
  "status": "completed",
  "svg": "base64_encoded_svg_string",
  "mask_contours": {
    "1": [[100.0, 50.0], [110.0, 55.0]],
    "2": [[150.0, 200.0], [160.0, 210.0]]
  }
}
```

**Response (Failed):**
```json
{
  "id": 123,
  "status": "failed",
  "error": "Error message"
}
```

### Job Status Values

- `pending`: Job is queued but not yet started
- `processing`: Job is currently being processed
- `completed`: Job completed successfully, results available
- `failed`: Job failed with an error

### Adding Delay for Testing

To simulate complex processing and demonstrate the async behavior, add the `delay` query parameter:

```bash
POST /api/v1/frontal/crop/submit?delay=20
```

This will add a 20-second delay before processing starts, allowing you to observe the status transitions from `pending` → `processing` → `completed`.
