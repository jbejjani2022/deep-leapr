#!/bin/bash

# Setup script for NLP libraries required by the expert text API

echo "Setting up NLP libraries for expert text API..."

# Download spaCy English model
echo "Downloading spaCy English model (en_core_web_sm)..."
python -m spacy download en_core_web_sm

# Download NLTK data (VADER lexicon for sentiment analysis)
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

# Download TextBlob corpora
echo "Downloading TextBlob corpora..."
python -m textblob.download_corpora

echo "✓ NLP library setup complete!"

