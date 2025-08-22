# TinyNet API

A FastAPI-based backend for TinyNet, a chat-first mind web application.

## Features

- **FastAPI** with async support
- **SQLAlchemy 2.x** with async engine
- **Alembic** for database migrations
- **Pydantic v2** for data validation
- **SQLite** by default (PostgreSQL ready)
- **512-dim Hashing Vectorizer** for text processing
- **Bootstrap Labeller** for training data generation
- **Complete API Contract** with OpenAPI specification

## Quick Start

### 1. Install Dependencies

```bash
make install
# or manually:
pip install -r requirements.txt
```

### 2. Environment Setup

The `.env` file is already configured with defaults:
- Database: SQLite at `./tinynet.db`
- Debug: `false`
- App Name: `TinyNet API`

Override as needed:
```bash
# For PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:password@localhost/tinynet
```

### 3. Run the API

```bash
# Development server
make api

# Production server
make api-prod

# Or manually:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the API

- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/healthz
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

## Development Commands

### Makefile Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make test          # Run tests
make api           # Start development server
make bootstrap     # Generate training data
make vectorize     # Test vectorizer
make db-upgrade    # Run migrations
make clean         # Clean up generated files
```

### Database Migrations

```bash
# Generate initial migration
alembic revision --autogenerate -m "Initial migration"

# Run migrations
alembic upgrade head

# Check migration status
alembic current

# Rollback one migration
alembic downgrade -1
```

### Code Quality

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with debug mode
DEBUG=true uvicorn app.main:app --reload
```

## Project Structure

```
tinynet-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── config.py        # Configuration settings
│   ├── db.py           # Database setup
│   ├── models.py       # SQLAlchemy models
│   ├── schemas.py      # Pydantic schemas
│   ├── api_openapi.yaml # OpenAPI specification
│   ├── routers/        # API route modules
│   │   ├── __init__.py
│   │   ├── classify.py # Classification endpoints
│   │   ├── nodes.py    # Node management
│   │   └── home.py     # Home page endpoints
│   └── ml/             # Machine learning components
│       ├── __init__.py
│       ├── vectorizer.py    # 512-dim hashing vectorizer
│       └── bootstrap.py     # Keyword bootstrap labeller
├── scripts/
│   └── bootstrap_labels.py  # CLI for training data generation
├── data/
│   └── raw/                 # Sample markdown files
├── tests/                   # Test suite
│   ├── test_vectorizer.py   # Vectorizer tests
│   ├── test_bootstrap.py    # Bootstrap labeller tests
│   └── test_api_contract.py # API contract tests
├── alembic/            # Database migrations
├── alembic.ini        # Alembic configuration
├── requirements.txt    # Python dependencies
├── Makefile           # Development commands
├── .env               # Environment variables
└── README.md          # This file
```

## Machine Learning Components

### Hashing Vectorizer (512-dim)

A deterministic text vectorizer that converts short text + metadata into 512-dimensional vectors:

```python
from app.ml.vectorizer import HashingVectorizer512

vec = HashingVectorizer512(n_features=512, seed=13)
x = vec.encode("Ran 2 miles, shin tight", ts=1724025600)
assert x.shape == (512,)
```

**Features:**
- Word unigrams + bigrams
- Character 3-grams + 4-grams
- Metadata: hour bucket, weekday, length bucket
- Stable hashing with SHA1
- L2 normalization

### Bootstrap Labeller

Automatically generates training data from markdown files using keyword rules:

```bash
# Process markdown files
python scripts/bootstrap_labels.py data/raw/*.md --out data/train.jsonl

# Output format:
{
  "text": "Ran 2 miles, shin tight",
  "categories": ["Fitness", "Running"],
  "state": "continue",
  "meta": {"source": "fitness_notes.md", "ts": null}
}
```

**Keyword Rules:**
- **Categories**: Fitness, Running, Strength, Music, Guitar, Learning, Admin
- **States**: start, continue, pause, end, blocked, idea

## API Endpoints

### Classification
- `POST /classify` - Classify text into categories and states
- `POST /correct` - Provide classification corrections
- `POST /train` - Submit training samples

### Nodes
- `GET /nodes/search` - Search for nodes
- `GET /nodes/{id}` - Get node details
- `GET /nodes/{id}/logs` - Get node progress logs

### Home
- `GET /home/review` - Get items needing review

### Health
- `GET /healthz` - Health check
- `GET /docs` - Interactive API documentation
- `GET /openapi.json` - OpenAPI specification

## Database Models

- **User**: Basic user information
- **Node**: Mind map nodes with status tracking
- **Link**: Connections between nodes with strength
- **ProgressLog**: Progress tracking for nodes
- **Todo**: Task management linked to nodes

## Configuration

The API uses Pydantic Settings for configuration management:

- Environment variables with `TINYNET_` prefix
- `.env` file support
- Type-safe configuration with validation

## Database

- **Default**: SQLite with `aiosqlite` driver
- **Production Ready**: PostgreSQL with `asyncpg` driver
- **Migrations**: Alembic with async support
- **Indexes**: Optimized for common queries

## Testing

```bash
# Run all tests
make test

# Run specific test file
python3 -m pytest tests/test_vectorizer.py -v

# Run with coverage
python3 -m pytest tests/ --cov=app --cov-report=html
```

## Development Workflow

1. **Setup**: `make dev-setup`
2. **Start API**: `make api`
3. **Generate Data**: `make bootstrap`
4. **Run Tests**: `make test`
5. **Database Changes**: `make db-migrate`

## Contributing

1. Install dependencies
2. Set up environment variables
3. Run migrations
4. Start development server
5. Make changes and test
6. Create new migrations if needed

## License

MIT License - see LICENSE file for details
