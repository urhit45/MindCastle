# TinyNet API

A FastAPI-based backend for TinyNet, a chat-first mind web application.

## Features

- **FastAPI** with async support
- **SQLAlchemy 2.x** with async engine
- **Alembic** for database migrations
- **Pydantic v2** for data validation
- **SQLite** by default (PostgreSQL ready)

## Quick Start

### 1. Install Dependencies

```bash
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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Access the API

- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/healthz
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Development Commands

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
│   └── routers/        # API route modules
├── alembic/            # Database migrations
│   ├── env.py         # Migration environment
│   ├── script.py.mako # Migration template
│   └── versions/      # Migration files
├── alembic.ini        # Alembic configuration
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
└── README.md          # This file
```

## Database Models

- **User**: Basic user information
- **Node**: Mind map nodes with status tracking
- **Link**: Connections between nodes with strength
- **ProgressLog**: Progress tracking for nodes
- **Todo**: Task management linked to nodes

## API Endpoints

- `GET /` - Welcome message
- `GET /healthz` - Health check
- `GET /docs` - Interactive API documentation

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

## Contributing

1. Install dependencies
2. Set up environment variables
3. Run migrations
4. Start development server
5. Make changes and test
6. Create new migrations if needed

## License

MIT License - see LICENSE file for details
