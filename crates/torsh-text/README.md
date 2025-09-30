# torsh-text

Text processing and NLP utilities for ToRSh, leveraging scirs2-text for efficient text operations.

## Overview

This crate provides comprehensive text processing capabilities:

- **Tokenization**: Various tokenization strategies (word, subword, character)
- **Vocabulary Management**: Efficient vocabulary building and management
- **Text Datasets**: Common NLP datasets and data loaders
- **Preprocessing**: Text normalization, cleaning, and augmentation
- **Embeddings**: Word embeddings and embedding layers
- **Utilities**: Text generation, metrics, and analysis tools

Note: This crate integrates with scirs2-text for optimized text processing operations.

## Usage

### Tokenization

```rust
use torsh_text::prelude::*;

// Basic tokenizer
let tokenizer = BasicTokenizer::new()
    .do_lower_case(true)
    .strip_accents(true);

let text = "Hello, World! This is a test.";
let tokens = tokenizer.tokenize(text)?;

// WordPiece tokenizer
let vocab = load_vocab("bert-base-uncased.txt")?;
let wordpiece = WordPieceTokenizer::new(vocab)
    .unk_token("[UNK]")
    .max_input_chars_per_word(100);

let tokens = wordpiece.tokenize("unaffable")?;
// ["un", "##aff", "##able"]

// Sentence Piece tokenizer
let sp_model = SentencePieceTokenizer::from_file("model.model")?;
let tokens = sp_model.encode("Hello world")?;
let decoded = sp_model.decode(&tokens)?;

// BPE tokenizer
let bpe = BPETokenizer::from_file("vocab.json", "merges.txt")?;
let encoding = bpe.encode("Hello, world!")?;
```

### Vocabulary Management

```rust
use torsh_text::vocab::*;

// Build vocabulary from corpus
let corpus = vec![
    "This is a sentence",
    "This is another sentence",
    "And one more",
];

let vocab = Vocab::build_from_iterator(
    corpus.iter().flat_map(|s| s.split_whitespace()),
    min_freq=1,
    max_size=Some(10000),
    special_tokens=vec!["<pad>", "<unk>", "<sos>", "<eos>"],
)?;

// Convert between tokens and indices
let indices = vocab.tokens_to_indices(&["This", "is", "a", "test"])?;
let tokens = vocab.indices_to_tokens(&indices)?;

// Save and load vocabulary
vocab.save("vocab.txt")?;
let loaded = Vocab::from_file("vocab.txt")?;
```

### Text Datasets

```rust
use torsh_text::datasets::*;

// IMDB sentiment dataset
let imdb = IMDB::new("./data", split="train", download=true)?;
for (text, label) in imdb.iter() {
    println!("Text: {}, Sentiment: {}", text, label);
}

// Custom text classification dataset
let dataset = TextClassificationDataset::from_csv(
    "data.csv",
    text_column="review",
    label_column="sentiment",
    tokenizer=tokenizer,
    max_length=512,
)?;

// Language modeling dataset
let lm_dataset = LanguageModelingDataset::from_file(
    "corpus.txt",
    tokenizer=tokenizer,
    block_size=128,
)?;

// Translation dataset
let translation = TranslationDataset::new(
    source_files=vec!["train.en"],
    target_files=vec!["train.de"],
    source_tokenizer=en_tokenizer,
    target_tokenizer=de_tokenizer,
)?;
```

### Text Preprocessing

```rust
use torsh_text::preprocessing::*;

// Text normalization
let normalizer = TextNormalizer::new()
    .lowercase(true)
    .remove_punctuation(true)
    .normalize_unicode(true)
    .expand_contractions(true);

let normalized = normalizer.normalize("Don't forget the café!")?;
// "do not forget the cafe"

// Text augmentation
let augmenter = TextAugmenter::new()
    .add_synonym_replacement(0.1)
    .add_random_insertion(0.1)
    .add_random_swap(0.1)
    .add_random_deletion(0.1);

let augmented = augmenter.augment("The quick brown fox")?;

// Padding and truncation
let padded = pad_sequence(
    &sequences,
    batch_first=true,
    padding_value=0,
    max_length=Some(100),
)?;
```

### Embeddings

```rust
use torsh_text::embeddings::*;

// Pre-trained embeddings
let glove = GloVe::from_file("glove.6B.300d.txt")?;
let word_vector = glove.get_vector("hello")?;

// Embedding layer
let embedding = Embedding::new(
    num_embeddings=vocab.size(),
    embedding_dim=300,
    padding_idx=Some(0),
)?;

// Initialize with pre-trained
embedding.from_pretrained(&glove, &vocab, freeze=false)?;

// Contextual embeddings
let bert_embeddings = BertEmbeddings::new(
    vocab_size=30522,
    hidden_size=768,
    pad_token_id=0,
    max_position_embeddings=512,
    type_vocab_size=2,
    layer_norm_eps=1e-12,
    dropout=0.1,
)?;
```

### Text Generation

```rust
use torsh_text::generation::*;

// Text generation utilities
let generator = TextGenerator::new(model)
    .temperature(0.8)
    .top_k(50)
    .top_p(0.95)
    .repetition_penalty(1.2);

let generated = generator.generate(
    prompt="Once upon a time",
    max_length=100,
    num_return_sequences=3,
)?;

// Beam search
let beam_output = generator.beam_search(
    input_ids,
    beam_size=5,
    max_length=50,
    length_penalty=0.6,
)?;

// Sampling strategies
let sampled = generator.sample(
    input_ids,
    do_sample=true,
    temperature=0.9,
    top_k=40,
)?;
```

### Text Analysis

```rust
use torsh_text::metrics::*;

// BLEU score
let bleu = calculate_bleu(
    &references,
    &hypotheses,
    max_n=4,
    smooth=true,
)?;

// ROUGE scores
let rouge = calculate_rouge(
    &references,
    &hypotheses,
    rouge_types=vec!["rouge1", "rouge2", "rougeL"],
)?;

// Perplexity
let perplexity = calculate_perplexity(&model, &test_data)?;

// Text statistics
let stats = TextStatistics::from_corpus(&corpus)?;
println!("Vocabulary size: {}", stats.vocab_size);
println!("Average sentence length: {:.2}", stats.avg_sentence_length);
```

### Integration with Models

```rust
use torsh_text::models::*;
use torsh_nn::prelude::*;

// Text classification model
struct TextClassifier {
    embedding: Embedding,
    encoder: LSTM,
    classifier: Linear,
}

impl TextClassifier {
    fn new(vocab_size: usize, num_classes: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, 128, Some(0)),
            encoder: LSTM::new(128, 256, 2, true, true, 0.1, false),
            classifier: Linear::new(512, num_classes, true), // bidirectional
        }
    }
}

// Sequence-to-sequence model
struct Seq2Seq {
    encoder: Encoder,
    decoder: Decoder,
    attention: Attention,
}

// Transformer model
let transformer = TransformerModel::new(
    vocab_size=vocab.size(),
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
)?;
```

### Utilities

```rust
// N-gram extraction
let ngrams = extract_ngrams(&tokens, n=3)?;

// TF-IDF
let tfidf = TfIdf::fit(&documents)?;
let tfidf_matrix = tfidf.transform(&documents)?;

// Text similarity
let sim = cosine_similarity(&text1_embedding, &text2_embedding)?;

// Sentence splitting
let sentences = split_sentences(text)?;

// Language detection
let language = detect_language(text)?;
```

## Integration with SciRS2

This crate leverages scirs2-text for:
- Efficient string operations
- Optimized tokenization algorithms
- Fast vocabulary lookups
- Vectorized text processing

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.