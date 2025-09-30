# Tokenization Guide

## Overview

The torsh-text crate provides a comprehensive tokenization system supporting multiple tokenization algorithms including BPE, WordPiece, SentencePiece, and custom tokenizers.

## Basic Usage

### WhitespaceTokenizer

```rust
use torsh_text::tokenization::WhitespaceTokenizer;

let tokenizer = WhitespaceTokenizer::new();
let tokens = tokenizer.tokenize("Hello world")?;
// Result: ["Hello", "world"]
```

### CharTokenizer

```rust
use torsh_text::tokenization::CharTokenizer;

let tokenizer = CharTokenizer::new();
let tokens = tokenizer.tokenize("Hello")?;
// Result: ["H", "e", "l", "l", "o"]
```

### BPE Tokenizer

```rust
use torsh_text::tokenization::BPETokenizer;

// Training BPE tokenizer
let texts = vec!["hello world", "world hello"];
let mut tokenizer = BPETokenizer::new();
tokenizer.train(&texts, 1000, None)?;

// Using trained tokenizer
let tokens = tokenizer.tokenize("hello world")?;
let token_ids = tokenizer.encode("hello world")?;
```

## Advanced Features

### Subword Regularization

```rust
use torsh_text::tokenization::SubwordRegularizer;

let regularizer = SubwordRegularizer::new(0.1); // 10% dropout
let regularized_tokens = regularizer.regularize(&tokens)?;
```

### Fast Tokenizers

```rust
use torsh_text::tokenization::FastTokenizer;

let mut tokenizer = FastTokenizer::new(base_tokenizer);
tokenizer.enable_caching(10000); // Cache up to 10k entries

// Batch processing
let texts = vec!["text1", "text2", "text3"];
let all_tokens = tokenizer.batch_encode(&texts)?;
```

### Unified Tokenizer API

```rust
use torsh_text::tokenization::{UnifiedTokenizer, TokenizerConfig};

let config = TokenizerConfig {
    tokenizer_type: "bpe".to_string(),
    vocab_size: 30000,
    special_tokens: ["<pad>", "<unk>", "<s>", "</s>"].iter().map(|s| s.to_string()).collect(),
    ..Default::default()
};

let tokenizer = UnifiedTokenizer::from_config(config)?;
```

## Special Token Handling

```rust
use torsh_text::vocab::Vocabulary;

let mut vocab = Vocabulary::new();
vocab.add_special_token("pad", "<pad>");
vocab.add_special_token("unk", "<unk>");
vocab.add_special_token("bos", "<s>");
vocab.add_special_token("eos", "</s>");

// Access special tokens
let pad_id = vocab.get_special_token_id("pad")?;
```

## Best Practices

1. **Choose the right tokenizer**: 
   - Use BPE for general text with rich vocabulary
   - Use WordPiece for transformer models
   - Use character-level for languages with complex morphology

2. **Vocabulary management**:
   - Always include special tokens (`<pad>`, `<unk>`, `<s>`, `</s>`)
   - Consider frequency filtering for large vocabularies
   - Use consistent vocabulary across training and inference

3. **Performance optimization**:
   - Enable caching for frequently tokenized text
   - Use batch processing for multiple texts
   - Consider subword regularization during training only

4. **Error handling**:
   - Always handle unknown tokens gracefully
   - Validate input text encoding
   - Check vocabulary compatibility between tokenizers