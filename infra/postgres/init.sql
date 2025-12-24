CREATE TABLE IF NOT EXISTS sample_users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) UNIQUE,
    name VARCHAR(100)
);
