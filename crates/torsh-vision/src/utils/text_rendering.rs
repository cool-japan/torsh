//! Text Rendering Module for ToRSh Vision
//!
//! This module provides text rendering capabilities for drawing text on images.
//! It includes a simple bitmap font system for rendering text onto image surfaces.
//!
//! # Features
//!
//! - Simple 5x7 pixel bitmap font rendering
//! - Support for basic ASCII characters (A-Z, 0-9, and common symbols)
//! - Text-on-image operations with customizable colors
//! - Font management and character mapping
//! - Efficient pixel-level text rendering
//!
//! # Examples
//!
//! ```rust
//! use image::{RgbImage, Rgb};
//! use torsh_vision::utils::text_rendering::draw_simple_text;
//!
//! let mut image = RgbImage::new(200, 100);
//! let color = Rgb([255, 255, 255]); // White text
//!
//! draw_simple_text(&mut image, "Hello World", 10, 10, color);
//! ```

use image::{Rgb, RgbImage};
use std::collections::HashMap;

/// Draw simple text on an image using a 5x7 pixel bitmap font
///
/// This function renders text directly onto an RGB image using a built-in
/// bitmap font. Each character is 5 pixels wide and 7 pixels tall, with
/// 1 pixel spacing between characters.
///
/// # Arguments
///
/// * `image` - Mutable reference to the RGB image to draw on
/// * `text` - The text string to render
/// * `start_x` - X coordinate for the top-left corner of the text
/// * `start_y` - Y coordinate for the top-left corner of the text
/// * `color` - RGB color for the text pixels
///
/// # Examples
///
/// ```rust
/// use image::{RgbImage, Rgb};
/// use torsh_vision::utils::text_rendering::draw_simple_text;
///
/// let mut image = RgbImage::new(200, 100);
/// let white = Rgb([255, 255, 255]);
///
/// // Draw white text at position (10, 10)
/// draw_simple_text(&mut image, "Score: 95.3", 10, 10, white);
///
/// // Draw red text at position (10, 25)
/// let red = Rgb([255, 0, 0]);
/// draw_simple_text(&mut image, "Error!", 10, 25, red);
/// ```
///
/// # Text Rendering Details
///
/// - Characters are rendered using a 5x7 pixel bitmap font
/// - Unsupported characters are rendered as a question mark
/// - Text is drawn left-to-right with automatic spacing
/// - Clipping is performed at image boundaries
pub fn draw_simple_text(
    image: &mut RgbImage,
    text: &str,
    start_x: u32,
    start_y: u32,
    color: Rgb<u8>,
) {
    // Get the bitmap font mapping
    let font_map = get_simple_font_map();

    let mut x_offset = 0;
    for ch in text.chars() {
        if let Some(bitmap) = font_map.get(&ch) {
            draw_character(image, bitmap, start_x + x_offset, start_y, color);
            x_offset += 6; // 5 pixels wide + 1 pixel spacing
        } else {
            // Unknown character, draw a question mark
            draw_character(
                image,
                &UNKNOWN_CHAR_BITMAP,
                start_x + x_offset,
                start_y,
                color,
            );
            x_offset += 6;
        }
    }
}

/// Draw a single character bitmap onto an image
///
/// This function renders a single character from its 5x7 bitmap representation
/// onto the specified position in the image.
///
/// # Arguments
///
/// * `image` - Mutable reference to the RGB image
/// * `bitmap` - Array of 7 bytes representing the character bitmap (one byte per row)
/// * `start_x` - X coordinate for the top-left corner of the character
/// * `start_y` - Y coordinate for the top-left corner of the character
/// * `color` - RGB color for the character pixels
///
/// # Bitmap Format
///
/// Each character is represented as an array of 7 bytes (one per row).
/// Each byte encodes 5 pixels, with bit 4 being the leftmost pixel.
/// A bit value of 1 means the pixel should be drawn, 0 means transparent.
///
/// # Examples
///
/// ```rust
/// use image::{RgbImage, Rgb};
/// use torsh_vision::utils::text_rendering::draw_character;
///
/// let mut image = RgbImage::new(100, 100);
/// let color = Rgb([255, 0, 0]);
///
/// // Draw the letter 'A' manually
/// let letter_a = [
///     0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b00000,
/// ];
/// draw_character(&mut image, &letter_a, 10, 10, color);
/// ```
pub fn draw_character(
    image: &mut RgbImage,
    bitmap: &[u8; 7], // 5x7 bitmap (7 rows)
    start_x: u32,
    start_y: u32,
    color: Rgb<u8>,
) {
    for (y, &row) in bitmap.iter().enumerate() {
        for x in 0..5 {
            if (row >> (4 - x)) & 1 == 1 {
                let pixel_x = start_x + x as u32;
                let pixel_y = start_y + y as u32;

                // Ensure we don't draw outside image boundaries
                if pixel_x < image.width() && pixel_y < image.height() {
                    image.put_pixel(pixel_x, pixel_y, color);
                }
            }
        }
    }
}

/// Get the bitmap font mapping for supported characters
///
/// This function returns a HashMap containing bitmap representations
/// for all supported characters. Each character maps to a 7-element
/// array representing a 5x7 pixel bitmap.
///
/// # Supported Characters
///
/// - Uppercase letters: A-Z
/// - Numbers: 0-9
/// - Special characters: space, period, comma, colon, hyphen
///
/// # Returns
///
/// A HashMap mapping characters to their 5x7 bitmap representations.
/// Each bitmap is encoded as 7 bytes (one per row).
///
/// # Character Set Details
///
/// The font includes:
/// - All 26 uppercase English letters (A-Z)
/// - All 10 digits (0-9)
/// - Common punctuation: space ( ), period (.), comma (,), colon (:), hyphen (-)
/// - A fallback question mark for unsupported characters
///
/// # Examples
///
/// ```rust
/// use torsh_vision::utils::text_rendering::get_simple_font_map;
///
/// let font_map = get_simple_font_map();
///
/// // Get the bitmap for letter 'A'
/// if let Some(bitmap) = font_map.get(&'A') {
///     println!("Letter A bitmap: {:?}", bitmap);
/// }
///
/// // Check if a character is supported
/// let is_supported = font_map.contains_key(&'@');
/// println!("@ symbol supported: {}", is_supported);
/// ```
pub fn get_simple_font_map() -> HashMap<char, [u8; 7]> {
    let mut map = HashMap::new();

    // Uppercase letters A-Z
    // Each character is represented as 7 bytes, each byte represents a row
    // Bit pattern: 0b[bit4][bit3][bit2][bit1][bit0] where bit4 is leftmost pixel
    map.insert(
        'A',
        [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b00000,
        ],
    );
    map.insert(
        'B',
        [
            0b11110, 0b10001, 0b11110, 0b11110, 0b10001, 0b11110, 0b00000,
        ],
    );
    map.insert(
        'C',
        [
            0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111, 0b00000,
        ],
    );
    map.insert(
        'D',
        [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110, 0b00000,
        ],
    );
    map.insert(
        'E',
        [
            0b11111, 0b10000, 0b11110, 0b11110, 0b10000, 0b11111, 0b00000,
        ],
    );
    map.insert(
        'F',
        [
            0b11111, 0b10000, 0b11110, 0b11110, 0b10000, 0b10000, 0b00000,
        ],
    );
    map.insert(
        'G',
        [
            0b01111, 0b10000, 0b10011, 0b10001, 0b10001, 0b01111, 0b00000,
        ],
    );
    map.insert(
        'H',
        [
            0b10001, 0b10001, 0b11111, 0b11111, 0b10001, 0b10001, 0b00000,
        ],
    );
    map.insert(
        'I',
        [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111, 0b00000,
        ],
    );
    map.insert(
        'J',
        [
            0b11111, 0b00001, 0b00001, 0b00001, 0b10001, 0b01110, 0b00000,
        ],
    );
    map.insert(
        'K',
        [
            0b10001, 0b10010, 0b11100, 0b11100, 0b10010, 0b10001, 0b00000,
        ],
    );
    map.insert(
        'L',
        [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111, 0b00000,
        ],
    );
    map.insert(
        'M',
        [
            0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b00000,
        ],
    );
    map.insert(
        'N',
        [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b00000,
        ],
    );
    map.insert(
        'O',
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110, 0b00000,
        ],
    );
    map.insert(
        'P',
        [
            0b11110, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000, 0b00000,
        ],
    );
    map.insert(
        'Q',
        [
            0b01110, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101, 0b00000,
        ],
    );
    map.insert(
        'R',
        [
            0b11110, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001, 0b00000,
        ],
    );
    map.insert(
        'S',
        [
            0b01111, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110, 0b00000,
        ],
    );
    map.insert(
        'T',
        [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000,
        ],
    );
    map.insert(
        'U',
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110, 0b00000,
        ],
    );
    map.insert(
        'V',
        [
            0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100, 0b00000,
        ],
    );
    map.insert(
        'W',
        [
            0b10001, 0b10001, 0b10001, 0b10101, 0b11011, 0b10001, 0b00000,
        ],
    );
    map.insert(
        'X',
        [
            0b10001, 0b01010, 0b00100, 0b00100, 0b01010, 0b10001, 0b00000,
        ],
    );
    map.insert(
        'Y',
        [
            0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000,
        ],
    );
    map.insert(
        'Z',
        [
            0b11111, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111, 0b00000,
        ],
    );

    // Numbers 0-9
    map.insert(
        '0',
        [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b01110, 0b00000,
        ],
    );
    map.insert(
        '1',
        [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110, 0b00000,
        ],
    );
    map.insert(
        '2',
        [
            0b01110, 0b10001, 0b00010, 0b00100, 0b01000, 0b11111, 0b00000,
        ],
    );
    map.insert(
        '3',
        [
            0b01110, 0b10001, 0b00110, 0b00110, 0b10001, 0b01110, 0b00000,
        ],
    );
    map.insert(
        '4',
        [
            0b00010, 0b00110, 0b01010, 0b11111, 0b00010, 0b00010, 0b00000,
        ],
    );
    map.insert(
        '5',
        [
            0b11111, 0b10000, 0b11110, 0b00001, 0b10001, 0b01110, 0b00000,
        ],
    );
    map.insert(
        '6',
        [
            0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110, 0b00000,
        ],
    );
    map.insert(
        '7',
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000,
        ],
    );
    map.insert(
        '8',
        [
            0b01110, 0b10001, 0b01110, 0b01110, 0b10001, 0b01110, 0b00000,
        ],
    );
    map.insert(
        '9',
        [
            0b01110, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110, 0b00000,
        ],
    );

    // Special characters and punctuation
    map.insert(
        ' ',
        [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
    );
    map.insert(
        '.',
        [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00000,
        ],
    );
    map.insert(
        ',',
        [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000, 0b00000,
        ],
    );
    map.insert(
        ':',
        [
            0b00000, 0b00100, 0b00000, 0b00000, 0b00100, 0b00000, 0b00000,
        ],
    );
    map.insert(
        '-',
        [
            0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
    );

    map
}

/// Bitmap representation for unknown/unsupported characters
///
/// This constant provides a fallback bitmap (question mark) for any
/// character that is not supported by the font system.
///
/// The bitmap represents a question mark in 5x7 pixel format:
/// ```text
///  ###
/// #   #
///    #
///   #
///
///   #
/// ```
pub const UNKNOWN_CHAR_BITMAP: [u8; 7] = [
    0b01110, 0b10001, 0b00010, 0b00100, 0b00000, 0b00100, 0b00000,
];

/// Calculate the rendered width of a text string in pixels
///
/// This function calculates how many pixels wide the rendered text
/// will be, including character spacing.
///
/// # Arguments
///
/// * `text` - The text string to measure
///
/// # Returns
///
/// The total width in pixels that the text will occupy when rendered.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::utils::text_rendering::calculate_text_width;
///
/// let width = calculate_text_width("Hello");
/// println!("Text width: {} pixels", width); // 29 pixels (5 chars * 5 pixels + 4 spaces)
/// ```
pub fn calculate_text_width(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }

    // Each character is 5 pixels wide, plus 1 pixel spacing between characters
    // Last character doesn't need trailing space
    (text.chars().count() as u32 * 6).saturating_sub(1)
}

/// Calculate the rendered height of text in pixels
///
/// # Returns
///
/// The height in pixels for the bitmap font (always 7 pixels).
///
/// # Examples
///
/// ```rust
/// use torsh_vision::utils::text_rendering::calculate_text_height;
///
/// let height = calculate_text_height();
/// println!("Text height: {} pixels", height); // 7 pixels
/// ```
pub fn calculate_text_height() -> u32 {
    7 // Fixed height for 5x7 bitmap font
}

/// Check if a character is supported by the font system
///
/// # Arguments
///
/// * `ch` - The character to check
///
/// # Returns
///
/// `true` if the character can be rendered, `false` otherwise.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::utils::text_rendering::is_character_supported;
///
/// assert!(is_character_supported('A'));
/// assert!(is_character_supported('5'));
/// assert!(!is_character_supported('@'));
/// ```
pub fn is_character_supported(ch: char) -> bool {
    let font_map = get_simple_font_map();
    font_map.contains_key(&ch)
}

/// Draw text with a background rectangle
///
/// This function draws text with an optional background rectangle,
/// useful for ensuring text readability over complex backgrounds.
///
/// # Arguments
///
/// * `image` - Mutable reference to the RGB image
/// * `text` - The text string to render
/// * `start_x` - X coordinate for the top-left corner of the text
/// * `start_y` - Y coordinate for the top-left corner of the text
/// * `text_color` - RGB color for the text pixels
/// * `bg_color` - Optional RGB color for the background rectangle
/// * `padding` - Padding around the text in pixels
///
/// # Examples
///
/// ```rust
/// use image::{RgbImage, Rgb};
/// use torsh_vision::utils::text_rendering::draw_text_with_background;
///
/// let mut image = RgbImage::new(200, 100);
/// let white = Rgb([255, 255, 255]);
/// let black = Rgb([0, 0, 0]);
///
/// // Draw white text on black background with 2 pixel padding
/// draw_text_with_background(&mut image, "Score: 95.3", 10, 10, white, Some(black), 2);
/// ```
pub fn draw_text_with_background(
    image: &mut RgbImage,
    text: &str,
    start_x: u32,
    start_y: u32,
    text_color: Rgb<u8>,
    bg_color: Option<Rgb<u8>>,
    padding: u32,
) {
    if let Some(bg) = bg_color {
        let text_width = calculate_text_width(text);
        let text_height = calculate_text_height();

        let bg_x = start_x.saturating_sub(padding);
        let bg_y = start_y.saturating_sub(padding);
        let bg_width = text_width + 2 * padding;
        let bg_height = text_height + 2 * padding;

        // Draw background rectangle
        for y in bg_y..=(bg_y + bg_height).min(image.height() - 1) {
            for x in bg_x..=(bg_x + bg_width).min(image.width() - 1) {
                image.put_pixel(x, y, bg);
            }
        }
    }

    // Draw the text on top
    draw_simple_text(image, text, start_x, start_y, text_color);
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_character_support() {
        assert!(is_character_supported('A'));
        assert!(is_character_supported('Z'));
        assert!(is_character_supported('0'));
        assert!(is_character_supported('9'));
        assert!(is_character_supported(' '));
        assert!(is_character_supported('.'));
        assert!(!is_character_supported('@'));
        assert!(!is_character_supported('a')); // lowercase not supported
    }

    #[test]
    fn test_text_width_calculation() {
        assert_eq!(calculate_text_width(""), 0);
        assert_eq!(calculate_text_width("A"), 5);
        assert_eq!(calculate_text_width("AB"), 11); // 5 + 1 + 5
        assert_eq!(calculate_text_width("ABC"), 17); // 5 + 1 + 5 + 1 + 5
    }

    #[test]
    fn test_text_height() {
        assert_eq!(calculate_text_height(), 7);
    }

    #[test]
    fn test_draw_simple_text() {
        let mut image = RgbImage::new(50, 20);
        let white = Rgb([255, 255, 255]);

        // Should not panic
        draw_simple_text(&mut image, "ABC", 0, 0, white);
        draw_simple_text(&mut image, "123", 0, 10, white);
    }

    #[test]
    fn test_draw_character() {
        let mut image = RgbImage::new(10, 10);
        let red = Rgb([255, 0, 0]);

        let test_bitmap = [
            0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111,
        ];

        // Should not panic
        draw_character(&mut image, &test_bitmap, 0, 0, red);
    }

    #[test]
    fn test_unknown_character() {
        let mut image = RgbImage::new(20, 10);
        let blue = Rgb([0, 0, 255]);

        // Should render unknown character as question mark
        draw_simple_text(&mut image, "@#$", 0, 0, blue);
    }

    #[test]
    fn test_text_with_background() {
        let mut image = RgbImage::new(100, 30);
        let white = Rgb([255, 255, 255]);
        let black = Rgb([0, 0, 0]);

        // Should not panic
        draw_text_with_background(&mut image, "TEST", 10, 10, white, Some(black), 2);
        draw_text_with_background(&mut image, "NO BG", 10, 20, white, None, 0);
    }
}
