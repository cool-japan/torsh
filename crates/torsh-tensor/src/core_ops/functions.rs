//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use torsh_core::device::DeviceType;
    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.device(), DeviceType::Cpu);
    }
    #[test]
    fn test_zeros_and_ones() {
        let zeros =
            Tensor::<f32>::zeros(&[3, 3], DeviceType::Cpu).expect("zeros creation should succeed");
        assert_eq!(zeros.numel(), 9);
        assert_eq!(zeros.get(&[0, 0]).expect("get should succeed"), 0.0);
        let ones =
            Tensor::<f32>::ones(&[2, 3], DeviceType::Cpu).expect("ones creation should succeed");
        assert_eq!(ones.numel(), 6);
        assert_eq!(ones.get(&[1, 2]).expect("get should succeed"), 1.0);
    }
    #[test]
    fn test_tensor_indexing() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        assert_eq!(tensor.get(&[0, 0]).expect("get should succeed"), 1.0);
        assert_eq!(tensor.get(&[0, 2]).expect("get should succeed"), 3.0);
        assert_eq!(tensor.get(&[1, 1]).expect("get should succeed"), 5.0);
    }
    #[test]
    fn test_tensor_properties() {
        let data = vec![1.0f32; 100];
        let tensor = Tensor::from_data(data, vec![10, 10], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        assert!(!tensor.is_view());
        assert!(!tensor.is_memory_mapped());
        assert!(!tensor.requires_grad());
        let with_grad = tensor.requires_grad_(true);
        assert!(with_grad.requires_grad());
    }
    #[test]
    fn test_ones_like_zeros_like() {
        let original =
            Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
                .expect("tensor creation should succeed");
        let zeros_like = original.zeros_like().expect("zeros_like should succeed");
        assert_eq!(zeros_like.shape().dims(), &[2, 2]);
        assert_eq!(zeros_like.get(&[0, 0]).expect("get should succeed"), 0.0);
        let ones_like = original.ones_like().expect("ones_like should succeed");
        assert_eq!(ones_like.shape().dims(), &[2, 2]);
        assert_eq!(ones_like.get(&[1, 1]).expect("get should succeed"), 1.0);
    }
}
