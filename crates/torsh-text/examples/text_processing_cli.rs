use scirs2_core::random::Random;
use std::io::{self, Write};
use torsh_text::prelude::*;
use torsh_text::TextAnalyzer;

#[derive(Debug)]
enum Command {
    Tokenize {
        text: String,
        method: String,
    },
    Preprocess {
        text: String,
        options: PreprocessingOptions,
    },
    Analyze {
        text: String,
    },
    Similarity {
        text1: String,
        text2: String,
    },
    Generate {
        prompt: String,
        length: usize,
    },
    Help,
    Quit,
}

#[derive(Debug)]
struct PreprocessingOptions {
    normalize: bool,
    clean_urls: bool,
    clean_emails: bool,
    clean_html: bool,
    lowercase: bool,
}

impl Default for PreprocessingOptions {
    fn default() -> Self {
        Self {
            normalize: true,
            clean_urls: true,
            clean_emails: true,
            clean_html: true,
            lowercase: true,
        }
    }
}

fn parse_command(input: &str) -> Command {
    let parts: Vec<&str> = input.trim().split_whitespace().collect();

    if parts.is_empty() {
        return Command::Help;
    }

    match parts[0].to_lowercase().as_str() {
        "tokenize" | "tok" => {
            if parts.len() < 3 {
                println!("Usage: tokenize <method> <text>");
                println!("Methods: whitespace, char, bpe");
                return Command::Help;
            }
            Command::Tokenize {
                method: parts[1].to_string(),
                text: parts[2..].join(" "),
            }
        }
        "preprocess" | "prep" => {
            if parts.len() < 2 {
                println!("Usage: preprocess <text>");
                return Command::Help;
            }
            Command::Preprocess {
                text: parts[1..].join(" "),
                options: PreprocessingOptions::default(),
            }
        }
        "analyze" | "stats" => {
            if parts.len() < 2 {
                println!("Usage: analyze <text>");
                return Command::Help;
            }
            Command::Analyze {
                text: parts[1..].join(" "),
            }
        }
        "similarity" | "sim" => {
            if parts.len() < 3 {
                println!("Usage: similarity <text1> | <text2>");
                println!("Use | to separate the two texts");
                return Command::Help;
            }
            let full_text = parts[1..].join(" ");
            let texts: Vec<&str> = full_text.split(" | ").collect();
            if texts.len() != 2 {
                println!("Please separate texts with ' | '");
                return Command::Help;
            }
            Command::Similarity {
                text1: texts[0].trim().to_string(),
                text2: texts[1].trim().to_string(),
            }
        }
        "generate" | "gen" => {
            if parts.len() < 3 {
                println!("Usage: generate <length> <prompt>");
                return Command::Help;
            }
            if let Ok(length) = parts[1].parse::<usize>() {
                Command::Generate {
                    length,
                    prompt: parts[2..].join(" "),
                }
            } else {
                println!("Invalid length parameter");
                Command::Help
            }
        }
        "help" | "h" | "?" => Command::Help,
        "quit" | "exit" | "q" => Command::Quit,
        _ => {
            println!("Unknown command: {}", parts[0]);
            Command::Help
        }
    }
}

fn execute_tokenize(text: &str, method: &str) -> Result<()> {
    println!("\n🔤 Tokenizing with method: {}", method);
    println!("📝 Input: {}", text);

    let tokens = match method.to_lowercase().as_str() {
        "whitespace" | "ws" => {
            let tokenizer = WhitespaceTokenizer::new();
            tokenizer.tokenize(text)?
        }
        "char" | "character" => {
            let tokenizer = CharTokenizer::new(None);
            tokenizer.tokenize(text)?
        }
        "bpe" => {
            let tokenizer = BPETokenizer::new();
            tokenizer.tokenize(text)?
        }
        _ => {
            println!("❌ Unknown tokenization method: {}", method);
            println!("Available methods: whitespace, char, bpe");
            return Ok(());
        }
    };

    println!("🎯 Tokens ({} total):", tokens.len());
    for (i, token) in tokens.iter().enumerate() {
        println!("  {}. '{}'", i + 1, token);
    }

    Ok(())
}

fn execute_preprocess(text: &str, options: &PreprocessingOptions) -> Result<()> {
    println!("\n🧹 Preprocessing text...");
    println!("📝 Input: {}", text);

    let mut pipeline = TextPreprocessingPipeline::new();

    if options.normalize {
        pipeline = pipeline.with_normalization(TextNormalizer::default());
    }

    if options.clean_urls || options.clean_emails || options.clean_html {
        let cleaner = TextCleaner::new()
            .remove_urls(options.clean_urls)
            .remove_emails(options.clean_emails)
            .remove_html(options.clean_html);
        pipeline = pipeline.with_cleaning(cleaner);
    }

    if options.lowercase {
        pipeline = pipeline.add_custom_step(Box::new(CustomStep::new(
            |text: &str| text.to_lowercase(),
            "lowercase".to_string(),
        )));
    }

    let processed = pipeline.process_text(text)?;

    println!("✨ Processed: {}", processed);
    println!("📊 Options used:");
    println!("   • Normalize: {}", options.normalize);
    println!("   • Clean URLs: {}", options.clean_urls);
    println!("   • Clean emails: {}", options.clean_emails);
    println!("   • Clean HTML: {}", options.clean_html);
    println!("   • Lowercase: {}", options.lowercase);

    Ok(())
}

fn execute_analyze(text: &str) -> Result<()> {
    println!("\n📊 Analyzing text...");
    println!("📝 Input: {}", text);

    let analyzer = TextAnalyzer::default();
    let stats = analyzer.analyze(text)?;

    println!("📈 Text Statistics:");
    println!("   • Character count: {}", stats.char_count);
    println!("   • Word count: {}", stats.word_count);
    println!("   • Sentence count: {}", stats.sentence_count);
    println!("   • Average word length: {:.2}", stats.avg_word_length);
    println!(
        "   • Average sentence length: {:.2}",
        stats.avg_sentence_length
    );

    println!("   • Type-token ratio: {:.3}", stats.type_token_ratio);

    // N-gram analysis
    let extractor = NgramExtractor::new(2);
    let bigrams = extractor.extract_word_ngrams(text);

    if !bigrams.is_empty() {
        println!("🔗 Most common bigrams:");
        let mut bigram_counts: Vec<_> = bigrams.into_iter().collect();
        bigram_counts.sort_by(|a, b| b.1.cmp(&a.1));

        for (bigram, count) in bigram_counts.iter().take(5) {
            println!("   • '{}': {} times", bigram, count);
        }
    }

    Ok(())
}

fn execute_similarity(text1: &str, text2: &str) -> Result<()> {
    println!("\n🔍 Calculating text similarity...");
    println!("📝 Text 1: {}", text1);
    println!("📝 Text 2: {}", text2);

    let jaccard = TextSimilarity::jaccard_similarity(text1, text2);
    let dice = TextSimilarity::dice_similarity(text1, text2);
    let overlap = TextSimilarity::overlap_similarity(text1, text2);

    println!("📊 Similarity Scores:");
    println!("   • Jaccard similarity: {:.3}", jaccard);
    println!("   • Dice similarity: {:.3}", dice);
    println!("   • Overlap similarity: {:.3}", overlap);

    // Interpretation
    let avg_similarity = (jaccard + dice + overlap) / 3.0;
    let interpretation = match avg_similarity {
        x if x >= 0.8 => "Very High",
        x if x >= 0.6 => "High",
        x if x >= 0.4 => "Moderate",
        x if x >= 0.2 => "Low",
        _ => "Very Low",
    };

    println!(
        "🎯 Overall similarity: {:.3} ({})",
        avg_similarity, interpretation
    );

    Ok(())
}

fn execute_generate(prompt: &str, length: usize) -> Result<()> {
    println!("\n🎲 Generating text...");
    println!("📝 Prompt: {}", prompt);
    println!("📏 Target length: {} tokens", length);

    // Simple text generation (this is a demo - real generation would need a trained model)
    let words = vec![
        "the", "and", "to", "of", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on",
        "are", "as", "with", "his", "they", "at", "be", "this", "have", "from", "or", "one", "had",
        "by", "word", "but", "what", "some", "we", "can", "out", "other", "were", "all", "there",
        "when",
    ];

    let mut generated = prompt.to_string();
    let mut rng = Random::seed(42);

    for _ in 0..length {
        let word = words[rng.gen_range(0..words.len())];
        generated.push(' ');
        generated.push_str(word);
    }

    println!("✨ Generated text:");
    println!("{}", generated);
    println!("\n⚠️  Note: This is a simple demo generator. For real text generation,");
    println!("   you would need a trained language model.");

    Ok(())
}

fn print_help() {
    println!("\n🚀 ToRSh Text Processing CLI");
    println!("===========================");
    println!("Available commands:");
    println!("  tokenize <method> <text>     - Tokenize text (methods: whitespace, char, bpe)");
    println!("  preprocess <text>            - Preprocess text with default options");
    println!("  analyze <text>               - Analyze text statistics and patterns");
    println!("  similarity <text1> | <text2> - Calculate similarity between two texts");
    println!("  generate <length> <prompt>   - Generate text from prompt (demo only)");
    println!("  help                         - Show this help message");
    println!("  quit                         - Exit the CLI");
    println!("\nExamples:");
    println!("  tokenize whitespace \"Hello world!\"");
    println!("  preprocess \"Visit https://example.com for more info!\"");
    println!("  analyze \"The quick brown fox jumps over the lazy dog.\"");
    println!("  similarity \"Hello world\" | \"Hello there\"");
    println!("  generate 5 \"Once upon a time\"");
}

fn print_welcome() {
    println!("🚀 Welcome to ToRSh Text Processing CLI!");
    println!("========================================");
    println!("This interactive tool demonstrates the capabilities of the torsh-text library.");
    println!("Type 'help' to see available commands or 'quit' to exit.");
    println!();
}

fn main() -> Result<()> {
    print_welcome();

    loop {
        print!("torsh-text> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let command = parse_command(&input);

                match command {
                    Command::Tokenize { text, method } => {
                        if let Err(e) = execute_tokenize(&text, &method) {
                            println!("❌ Error: {}", e);
                        }
                    }
                    Command::Preprocess { text, options } => {
                        if let Err(e) = execute_preprocess(&text, &options) {
                            println!("❌ Error: {}", e);
                        }
                    }
                    Command::Analyze { text } => {
                        if let Err(e) = execute_analyze(&text) {
                            println!("❌ Error: {}", e);
                        }
                    }
                    Command::Similarity { text1, text2 } => {
                        if let Err(e) = execute_similarity(&text1, &text2) {
                            println!("❌ Error: {}", e);
                        }
                    }
                    Command::Generate { prompt, length } => {
                        if let Err(e) = execute_generate(&prompt, length) {
                            println!("❌ Error: {}", e);
                        }
                    }
                    Command::Help => print_help(),
                    Command::Quit => {
                        println!("👋 Goodbye! Thanks for using ToRSh Text Processing CLI!");
                        break;
                    }
                }
            }
            Err(e) => {
                println!("❌ Error reading input: {}", e);
            }
        }

        println!(); // Add spacing between commands
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tokenize_command() {
        let cmd = parse_command("tokenize whitespace hello world");
        match cmd {
            Command::Tokenize { method, text } => {
                assert_eq!(method, "whitespace");
                assert_eq!(text, "hello world");
            }
            _ => panic!("Expected Tokenize command"),
        }
    }

    #[test]
    fn test_parse_help_command() {
        let cmd = parse_command("help");
        match cmd {
            Command::Help => {}
            _ => panic!("Expected Help command"),
        }
    }

    #[test]
    fn test_parse_quit_command() {
        let cmd = parse_command("quit");
        match cmd {
            Command::Quit => {}
            _ => panic!("Expected Quit command"),
        }
    }

    #[test]
    fn test_execute_tokenize() -> Result<()> {
        // This would be more comprehensive with actual tokenizers
        execute_tokenize("hello world", "whitespace")?;
        Ok(())
    }

    #[test]
    fn test_execute_analyze() -> Result<()> {
        execute_analyze("The quick brown fox jumps over the lazy dog.")?;
        Ok(())
    }
}
