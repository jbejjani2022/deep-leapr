import random
from typing import List, Tuple

from chess_position import ChessPosition


def load_prompt_template(prompt_path: str) -> str:
    """Load the prompt template from a file."""
    with open(prompt_path, "r") as f:
        return f.read()


def format_chess_api_description() -> str:
    """Format the chess API description for the prompt."""
    return """# Chess API Documentation
## Class chess.Board Methods
- board.turn: True if White to move, False if Black
- board.fullmove_number: Current move number
- board.halfmove_clock: Halfmove clock for 50-move rule
- board.is_check(): True if current player is in check
- board.is_checkmate(): True if current player is checkmated
- board.is_stalemate(): True if stalemate
- board.is_insufficient_material(): Returns True if insufficient material
- board.piece_at(square): Returns piece at given square (or None)
- board.piece_map(): Returns dict mapping squares to pieces
- board.legal_moves: Iterator over legal moves
- board.attackers(color, square): Returns set of squares attacking given square
- board.is_attacked_by(color, square): Returns True if square is attacked by color

## Chess Squares and Pieces
- chess.A1, chess.A2, ..., chess.H8: Square constants
- chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING: Piece types
- chess.WHITE, chess.BLACK: Colors
- piece.piece_type: Type of piece (PAWN, KNIGHT, etc.)
- piece.color: Color of piece (WHITE or BLACK)

## Useful Functions
- chess.square_name(square): Convert square index to name (e.g., "e4")
- chess.parse_square(name): Convert square name to index
- chess.square_file(square): Get file (0-7) of square
- chess.square_rank(square): Get rank (0-7) of square
- chess.square_distance(sq1, sq2): Manhattan distance between squares
"""


def format_image_api_description() -> str:
    """Format the image processing API description for the prompt."""
    return """
# Image Processing API Documentation

The features receive an image as a numpy array, so you can use any numpy functions on it. For RGB images, shape is (height, width, 3). For grayscale, shape is (height, width).

## Image Processing Methods
- image.shape: Returns (height, width, channels) for RGB or (height, width) for grayscale
- image.mean(): Average pixel intensity across all channels
- image.std(): Standard deviation of pixel intensities
- image.max(), image.min(): Maximum and minimum pixel values
- np.sum(image): Sum of all pixel values
- np.count_nonzero(image): Count of non-zero pixels

## Handle Both Grayscale and RGB
- Check format: len(image.shape) == 2 for grayscale, len(image.shape) == 3 for RGB
- Unpack safely: h, w = image.shape[:2]  # Works for both formats
- For RGB only: image[:,:,0] (red), image[:,:,1] (green), image[:,:,2] (blue)

## Useful Functions
- np.mean(image): Average intensity
- np.std(image): Standard deviation
- np.gradient(image): Image gradients - for RGB use on single channel: np.gradient(image[:,:,0])
- np.where(condition, x, y): Conditional selection
- np.argmax(image), np.argmin(image): Location of max/min values
- np.percentile(image, q): Percentile values
- np.histogram(image.flatten(), bins): Intensity histogram

## Spatial Analysis
- image[start_row:end_row, start_col:end_col]: Region selection
- Center region: image[h//4:3*h//4, w//4:3*w//4]
- Edge detection: np.gradient(np.mean(image, axis=2)) for RGB
- Color channel differences: image[:,:,0] - image[:,:,1]

## Example Feature Function
def feature(image: np.ndarray) -> float:
    "Average pixel intensity in the center region"
    if len(image.shape) == 3:
        h, w, c = image.shape
        gray = np.mean(image, axis=2)
    else:
        h, w = image.shape
        gray = image
    center_h, center_w = h // 4, w // 4
    center_region = gray[center_h:3*center_h, center_w:3*center_w]
    return float(np.mean(center_region))
"""


def format_text_api_description() -> str:
    """Format the text processing API description for the prompt."""
    return """
# Text Processing API Documentation

The features receive text as a string, so you can use any string methods and text processing functions.

## String Methods
- text.lower(), text.upper(): Case conversion
- text.strip(): Remove whitespace
- text.split(delimiter): Split into list
- text.count(substring): Count occurrences
- text.startswith(prefix), text.endswith(suffix): Check prefixes/suffixes
- text.find(substring): Find position of substring
- text.replace(old, new): Replace text

## Text Analysis
- len(text): Length of text
- text.isdigit(), text.isalpha(), text.isalnum(): Character type checks
- sum(1 for c in text if c.isupper()): Count uppercase letters
- text.split(): Split on whitespace to get words

## Regular Expressions (re module)
- re.findall(pattern, text): Find all matches
- re.search(pattern, text): Find first match
- re.sub(pattern, replacement, text): Replace patterns
- len(re.findall(r'\\w+', text)): Count words
- len(re.findall(r'[.!?]', text)): Count sentences

## Useful Patterns
- Word count: len(text.split())
- Sentence count: text.count('.') + text.count('!') + text.count('?')
- Average word length: sum(len(word) for word in text.split()) / len(text.split())
- Punctuation density: sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)

## Example Feature Function
def feature(text: str) -> float:
    "Average word length in the text"
    words = text.split()
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)
"""


def format_text_api_description_plus() -> str:
    """Format an enriched text processing API description."""
    base = format_text_api_description().strip()
    extras = """
# Additional Text Processing Utilities

In addition to everything listed above, you may also use these modules:

## statistics
- statistics.mean(data), statistics.median(data), statistics.pstdev(data)

## collections
- Counter(iterable): count tokens quickly
- defaultdict(int): build sparse frequency maps

## itertools
- itertools.islice(iterable, n): take subsequences
- itertools.pairwise(words): iterate over adjacent word pairs

## string / unicodedata
- string.punctuation, string.ascii_letters
- unicodedata.category(char): inspect Unicode categories (e.g., punctuation vs. letter)

## numpy (np)
- np.array(list_of_numbers), np.percentile(array, q)
- np.diff(array): differences between consecutive values

## Example Feature Function
def feature(text: str) -> float:
    "Median sentence length measured in words"
    sentences = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sentences if s.split()]
    if not lengths:
        return 0.0
    return float(statistics.median(lengths))
"""
    return f"{base}\n\n{extras.strip()}\n"


def format_text_api_description_expert() -> str:
    """Format an expert-level text processing API with advanced NLP libraries."""
    base = format_text_api_description_plus().strip()
    expert_extras = """
# NLP Libraries

In addition to everything listed above, you have access to NLP libraries for linguistic analysis:

## spaCy - Advanced Linguistic Analysis

The spaCy English model is **pre-loaded and available as `nlp`**. You can use it directly without calling `spacy.load()`.

### Processing Text
```python
# nlp is already loaded - you can just use it directly
doc = nlp(text)  # Process text once, then access all linguistic features
```

### Token-Level Features
Each token in doc has the following attributes:
- token.text: The actual text
- token.pos_: Part-of-speech tag (NOUN, VERB, ADJ, ADV, PRON, DET, ADP, etc.)
- token.tag_: Detailed POS tag (NN, VBZ, JJ, etc.)
- token.dep_: Dependency relation (nsubj, dobj, prep, etc.)
- token.lemma_: Lemmatized form (e.g., "running" -> "run")
- token.is_stop: True if token is a stopword
- token.is_alpha: True if token is alphabetic
- token.is_punct: True if token is punctuation
- token.head: The syntactic head of this token
- token.children: Iterator over syntactic children

### Sentence-Level Features
- doc.sents: Iterator over sentences
- len(list(doc.sents)): Count sentences

### Named Entities
- doc.ents: Tuple of named entities
- ent.text: Entity text
- ent.label_: Entity type (PERSON, ORG, GPE, DATE, MONEY, etc.)

### Common Patterns
```python
# Count specific POS tags
verb_count = sum(1 for token in doc if token.pos_ == 'VERB')
noun_count = sum(1 for token in doc if token.pos_ == 'NOUN')
adj_count = sum(1 for token in doc if token.pos_ == 'ADJ')

# POS ratios
verb_ratio = verb_count / len(doc) if len(doc) > 0 else 0.0

# Lexical diversity with lemmas
unique_lemmas = len(set(token.lemma_.lower() for token in doc if token.is_alpha))
lexical_diversity = unique_lemmas / len(doc) if len(doc) > 0 else 0.0

# Dependency tree depth
def get_tree_depth(token):
    if not list(token.children):
        return 0
    return 1 + max(get_tree_depth(child) for child in token.children)
max_depth = max(get_tree_depth(sent.root) for sent in doc.sents) if list(doc.sents) else 0

# Named entity density
entity_count = len(doc.ents)
entity_density = entity_count / len(doc) if len(doc) > 0 else 0.0
```

## NLTK - Sentiment Analysis

The VADER sentiment analyzer is **pre-loaded and available as `sia`**. You can use it directly for sentiment analysis.

### VADER Sentiment
```python
# sia (SentimentIntensityAnalyzer) is already loaded
scores = sia.polarity_scores(text)
# Returns: {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.8}
```

- scores['compound']: Overall sentiment (-1.0 to 1.0, negative to positive)
- scores['pos']: Positive sentiment proportion (0.0 to 1.0)
- scores['neg']: Negative sentiment proportion (0.0 to 1.0)
- scores['neu']: Neutral sentiment proportion (0.0 to 1.0)

## TextBlob - Simple Sentiment and Subjectivity

TextBlob provides easy sentiment analysis with polarity and subjectivity scores.

```python
from textblob import TextBlob
blob = TextBlob(text)

# Sentiment
polarity = blob.sentiment.polarity  # -1.0 (negative) to 1.0 (positive)
subjectivity = blob.sentiment.subjectivity  # 0.0 (objective) to 1.0 (subjective)
```

## Textstat - Readability Metrics

Textstat provides various readability scores.

```python
import textstat

# Readability scores
flesch_reading_ease = textstat.flesch_reading_ease(text)  # Higher = easier
flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)  # US grade level
smog_index = textstat.smog_index(text)  # Years of education needed
automated_readability_index = textstat.automated_readability_index(text)
```

## Example Feature Function
def feature(text: str) -> float:
    "Type-token ratio using lemmas (lexical diversity)"
    doc = nlp(text)
    if len(doc) == 0:
        return 0.0
    lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
    if not lemmas:
        return 0.0
    unique_lemmas = len(set(lemmas))
    return float(unique_lemmas) / len(lemmas)

## Reminder

1. **Pre-loaded Models**: `nlp` (spaCy) and `sia` (VADER) are already loaded - you can use them directly
   - DON'T: `nlp = spacy.load('en_core_web_sm')`
   - DO: `doc = nlp(text)
"""
    return f"{base}\n\n{expert_extras.strip()}\n"


def format_database(database: List[Tuple[str, float]], max_samples: int) -> str:
    """Format the database of features for the prompt."""
    SCORE_POWER = 2
    # Filter out entries with None scores
    valid_entries = [(code, score) for code, score in database if score is not None]

    if len(valid_entries) > max_samples:
        # Sample with weights proportional to the scores (inverse for error metrics)
        weights = [1.0 / (abs(score) + 1) ** SCORE_POWER for _, score in valid_entries]
        sum_weights = sum(weights)

        if sum_weights > 0:
            weights = [w / sum_weights for w in weights]
        else:
            weights = [1.0 / len(valid_entries) for _ in valid_entries]

        sampled_indices = random.choices(
            range(len(valid_entries)), weights=weights, k=max_samples
        )
        database_sample = [valid_entries[i] for i in sampled_indices]
    else:
        database_sample = valid_entries

    formatted_database = []
    for i, (feature_code, mae) in enumerate(database_sample):
        formatted_database.append(f"Feature {i+1} (MAE: {mae:.2f}):\n{feature_code}")

    return "\n\n".join(formatted_database)


def format_examples(samples: List, batch_size: int, domain: str = "chess") -> str:
    """Format a batch of samples for the prompt."""
    if len(samples) > batch_size:
        samples_batch = random.sample(samples, batch_size)
    else:
        samples_batch = samples

    formatted_examples = []
    for i, sample in enumerate(samples_batch):
        if domain == "chess":
            formatted_examples.append(
                f"Position {i+1} (Evaluation: {sample.evaluation}):\n{sample}"
            )
        elif domain == "image_classification":
            dataset_name = sample.metadata.get("dataset", "Unknown")
            task_type = sample.metadata.get("task_type", "Unknown")
            image_shape = sample.image.shape
            formatted_examples.append(
                f"Image {i+1} (Target: {sample.target}, Dataset: {dataset_name}, Task: {task_type}):\n"
                f"Shape: {image_shape}, Mean intensity: {sample.image.mean():.1f}"
            )

    return "\n\n".join(formatted_examples)
