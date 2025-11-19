# 🎯 Facial Region Processing

> A facial image processing pipeline that detects, segments, and labels facial regions with automatic tilt correction and cropping.

---

## 📦 Installation

### Local Development

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements-local.txt
   ```

### 🐳 Docker Setup

> **No additional installation needed** - Docker will handle everything during build.

---

## 🚀 Running

### 📁 Local Execution

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

---

### 🐳 API Service (Docker)

#### Quick Start

```bash
# Start all services (API, PostgreSQL, Prometheus)
docker-compose up -d

# View logs
docker-compose logs -f facial-processing-api

# Stop all services
docker-compose down

# Stop and remove volumes (clears database)
docker-compose down -v
```

```

#### Service Endpoints

Once running, the API will be available at:

| Service | URL | Description |
|---------|-----|-------------|
| **API** | `http://localhost:8000` | Main API endpoint |
| **Documentation** | `http://localhost:8000/api/docs` | Interactive API docs |
| **Health Check** | `http://localhost:8000/health` | Service health status |
| **Metrics** | `http://localhost:8000/metrics` | Prometheus metrics |

#### 📊 Prometheus Monitoring

- **Prometheus UI**: `http://localhost:9090`
- **Scraping**: Automatically scrapes metrics from the API every 5 seconds
- **Features**: View metrics, create queries, and build dashboards

---

### 💻 API Service (Local - Development)

```bash
# Install dependencies (if not already done)
pip install -r requirements-local.txt

# Run the API (from project root)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

> **💡 Note:** The API can run without PostgreSQL. If the database is not available:
> - The API will start normally with a warning message
> - Caching will be disabled (no cache hits/misses)
> - All other functionality works normally
> - To disable database completely, set `USE_DATABASE=false` environment variable

---

### ⚡ Load Testing Mode

> **🚀 Maximum Performance Mode** - Optimized for load testing with minimal overhead

#### Local Development

```bash
export LOAD_TESTING_MODE=true
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Docker

```bash
# Set environment variable and restart
LOAD_TESTING_MODE=true docker-compose up -d --build
```

#### Docker Compose (Permanent)

Edit `docker-compose.yml` and change:
```yaml
- LOAD_TESTING_MODE=${LOAD_TESTING_MODE:-false}
```
to:
```yaml
- LOAD_TESTING_MODE=true
```

#### What Happens When Enabled

| Feature | Behavior |
|---------|----------|
| **Delay Parameter** | Ignored (always 0) |
| **Logging Level** | Reduced to WARNING (minimal output) |
| **INFO Logs** | Disabled |
| **Performance** | Maximum speed |

---

## 📡 API Usage

> The API uses an **asynchronous, non-blocking job queue system**. Jobs are submitted and processed in the background, allowing instant responses.

---

### 📤 Submit a Job

**Endpoint:** `POST /api/v1/frontal/crop/submit`

Submits a processing job and returns immediately with a job ID and status.

#### Request Body

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

#### Query Parameters

- `delay` (optional): Simulate processing delay in seconds (e.g., `?delay=20` for 20 seconds)

> **Note:** Delay is ignored when `LOAD_TESTING_MODE=true`

#### Response (HTTP 202 Accepted)

```json
{
  "id": 123,
  "status": "pending"
}
```

---

### 📊 Check Job Status

**Endpoint:** `GET /api/v1/frontal/crop/status/{job_id}`

Returns the current status of a job. When completed, includes the processing results.

#### Response Examples

**Pending/Processing:**
```json
{
  "id": 123,
  "status": "processing"
}
```

**Completed:**
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

**Failed:**
```json
{
  "id": 123,
  "status": "failed",
  "error": "Error message"
}
```

---

### 📋 Job Status Values

| Status | Description |
|--------|-------------|
| `pending` | Job is queued but not yet started |
| `processing` | Job is currently being processed |
| `completed` | Job completed successfully, results available |
| `failed` | Job failed with an error |

---

### ⏱️ Adding Delay for Testing

To simulate complex processing and demonstrate the async behavior, add the `delay` query parameter:

```bash
POST /api/v1/frontal/crop/submit?delay=20
```

> This will add a 20-second delay before processing starts, allowing you to observe the status transitions from `pending` → `processing` → `completed`.

---

## 📈 Monitoring & Observability

> The API includes **Prometheus metrics** for observability and monitoring.

---

### 📊 Prometheus Metrics

When running with Docker Compose, Prometheus is automatically started and configured to scrape metrics from the API.

#### Access Points

- **Prometheus UI**: `http://localhost:9090`
- **Metrics Endpoint**: `http://localhost:8000/metrics`

#### Available Metrics

**HTTP Request Metrics:**
- `http_requests_total` - Total number of HTTP requests by method, endpoint, and status code
- `http_request_duration_seconds` - HTTP request duration histogram

**Job Processing Metrics:**
- `jobs_total` - Total number of jobs by status (pending, completed, failed)
- `jobs_in_progress` - Current number of jobs being processed (gauge)
- `job_processing_duration_seconds` - Job processing duration histogram

---

### 🎨 Rich Console Logging

The API uses **Rich** for beautifully formatted console logs with colors, emojis, and enhanced readability.

**Example log output:**
```
✅ Created job #123 with status pending
🚀 Starting crop submit processing
🖼️  Decoding base64 images...
📍 Processing 478 landmarks...
🎨 Processing facial regions...
✅ Job #123 completed successfully in 2.45s
```

---

## 💾 Database & Caching

> The API uses **PostgreSQL** with a **perceptual cache system** to store mask contours and avoid duplicate processing.

---

### 🔍 How the Cache Works

**Step-by-step:**

1. **Request received** → API receives image processing request
2. **Cache check** → System computes two hashes:
   - **Image hash (MD5)**: Exact match for identical images
   - **Perceptual hash (aHash)**: Similarity match for visually similar images
3. **Cache lookup**:
   - First checks for exact match (same image)
   - If not found, checks for perceptual match (similar images within threshold)
4. **Cache hit** → Returns cached SVG and mask contours immediately (no processing)
5. **Cache miss** → Processes image normally, then stores result in database

---

### 📍 Where Data is Stored

**PostgreSQL Database:**
- **Location**: PostgreSQL container (port 5432)
- **Database name**: `facial_processing`
- **Table**: `mask_contour_cache`
- **What's stored**:
  - ✅ SVG overlay (base64 encoded)
  - ✅ Mask contours (JSON format)
  - ✅ Image hashes (for matching)
  - ✅ Access count (cache hit tracking)
  - ✅ Timestamps

> **Note**: The original images are NOT stored - only the processing results (SVG and contours) are cached.

---

### 👀 Viewing Cached Data

**API Endpoints:**

1. **Cache Statistics:**
   ```bash
   GET /api/v1/cache/stats
   ```
   Returns total entries, cache hits, and most accessed entry.

2. **List Cache Entries:**
   ```bash
   GET /api/v1/cache/entries?limit=10&offset=0
   ```
   Returns paginated list of cache entries with metadata.

3. **Get Specific Entry:**
   ```bash
   GET /api/v1/cache/entries/{entry_id}
   ```
   Returns full cache entry including SVG and mask contours.

---

### ⚙️ Cache Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| **Similarity threshold** | Default: 5 (Hamming distance, range 0-64) | Lower = stricter matching (fewer false positives)<br>Higher = more lenient matching (more cache hits) |
| **Storage** | Automatic | Results are automatically stored after processing |
| **Persistence** | Docker volume | Data persists in PostgreSQL volume across container restarts |
