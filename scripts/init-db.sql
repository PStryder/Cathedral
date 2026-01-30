-- Cathedral Database Initialization Script
-- This runs automatically when the PostgreSQL container is first created

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable other useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant permissions (if needed for specific users)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cathedral;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cathedral;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Cathedral database initialized with pgvector extension';
END $$;
