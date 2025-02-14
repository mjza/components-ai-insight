CREATE TABLE stage_posts_cleaned (
    id INTEGER PRIMARY KEY,
    posttypeid INTEGER, -- 1 = Question, 2 = Answer
    title TEXT,         -- Lowercased & cleaned
    body TEXT,          -- Lowercased, cleaned, HTML removed
    tags TEXT          -- Lowercased & tokenized
);

CREATE TABLE stage_libraries_cleaned (
    id INTEGER PRIMARY KEY,
    library_name TEXT UNIQUE, -- Normalized library name
    original_name TEXT, -- Raw library name from Libraries.io
    platform TEXT, -- NPM, Maven, PyPI, etc.
    description TEXT -- Tokenized description for later analysis
);

CREATE TABLE post_library_links (
    post_id INTEGER,
    library_id INTEGER,
    confidence_score FLOAT, -- Matching confidence (0-1)
    matching_method TEXT, -- "fuzzy", "NER", "ML"
    FOREIGN KEY (post_id) REFERENCES stage_posts_cleaned(id),
    FOREIGN KEY (library_id) REFERENCES stage_libraries_cleaned(id)
);
