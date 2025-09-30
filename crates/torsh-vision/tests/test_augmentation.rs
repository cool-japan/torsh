//! Integration tests for data augmentation framework

use torsh_core::device::{CpuDevice, DeviceType};
use torsh_tensor::creation::*;
use torsh_vision::transforms::*;

#[test]
fn test_basic_transforms() {
    let input = ones(&[3, 64, 64]).unwrap();

    // Test resize
    let resized = Resize::new((32, 32)).forward(&input).unwrap();
    assert_eq!(resized.shape().dims(), &[3, 32, 32]);

    // Test center crop
    let cropped = CenterCrop::new((32, 32)).forward(&input).unwrap();
    assert_eq!(cropped.shape().dims(), &[3, 32, 32]);

    // Test horizontal flip
    let flipped = RandomHorizontalFlip::new(1.0).forward(&input).unwrap();
    assert_eq!(flipped.shape().dims(), &[3, 64, 64]);

    // Test normalization
    let normalized = Normalize::new(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5])
        .forward(&input)
        .unwrap();
    assert_eq!(normalized.shape().dims(), &[3, 64, 64]);
}

#[test]
fn test_compose_transforms() {
    let cpu_device = CpuDevice::new();
    let input = ones(&[3, 128, 128]).unwrap();

    let pipeline = Compose::new(vec![
        Box::new(Resize::new((64, 64))),
        Box::new(CenterCrop::new((48, 48))),
        Box::new(RandomHorizontalFlip::new(0.0)), // Deterministic for testing
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        )),
    ]);

    let output = pipeline.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[3, 48, 48]);
}

#[test]
fn test_random_transforms() {
    let cpu_device = CpuDevice::new();
    let input = ones(&[3, 100, 100]).unwrap();

    // Test random crop
    let cropped = RandomCrop::new((64, 64)).forward(&input).unwrap();
    assert_eq!(cropped.shape().dims(), &[3, 64, 64]);

    // Test random resized crop
    let resized_cropped = RandomResizedCrop::new((32, 32)).forward(&input).unwrap();
    assert_eq!(resized_cropped.shape().dims(), &[3, 32, 32]);

    // Test random erasing
    let erased = RandomErasing::new(1.0).forward(&input).unwrap();
    assert_eq!(erased.shape().dims(), &[3, 100, 100]);
}

#[test]
fn test_color_transforms() {
    let cpu_device = CpuDevice::new();
    let input = ones(&[3, 32, 32]).unwrap();

    // Test color jitter
    let jittered = ColorJitter::new()
        .brightness(0.2)
        .contrast(0.2)
        .forward(&input)
        .unwrap();
    assert_eq!(jittered.shape().dims(), &[3, 32, 32]);
}

#[test]
fn test_transform_error_handling() {
    // Test with invalid tensor shapes
    let cpu_device = CpuDevice::new();
    let invalid_input = ones(&[32, 32]).unwrap(); // 2D instead of 3D

    let result = Resize::new((16, 16)).forward(&invalid_input);
    assert!(result.is_err());

    let result = CenterCrop::new((16, 16)).forward(&invalid_input);
    assert!(result.is_err());
}

#[test]
fn test_training_vs_inference_pipelines() {
    let cpu_device = CpuDevice::new();
    let input = ones(&[3, 256, 256]).unwrap();

    // Training pipeline (with randomness)
    let training_pipeline = Compose::new(vec![
        Box::new(RandomResizedCrop::new((224, 224))),
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(ColorJitter::new().brightness(0.1)),
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        )),
    ]);

    // Test pipeline (deterministic)
    let test_pipeline = Compose::new(vec![
        Box::new(Resize::new((256, 256))),
        Box::new(CenterCrop::new((224, 224))),
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        )),
    ]);

    let training_output = training_pipeline.forward(&input).unwrap();
    let test_output = test_pipeline.forward(&input).unwrap();

    assert_eq!(training_output.shape().dims(), &[3, 224, 224]);
    assert_eq!(test_output.shape().dims(), &[3, 224, 224]);
}
