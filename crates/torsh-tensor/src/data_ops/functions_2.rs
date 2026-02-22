//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use torsh_core::device::DeviceType;
    #[test]
    fn test_fill_operations() {
        let mut tensor =
            Tensor::<f32>::zeros(&[2, 3], DeviceType::Cpu).expect("failed to create zeros tensor");
        tensor.fill_(5.0).expect("fill_ failed");
        assert_eq!(tensor.get_item(&[0, 0]).expect("get_item failed"), 5.0);
        assert_eq!(tensor.get_item(&[1, 2]).expect("get_item failed"), 5.0);
        tensor.zero_().expect("zero_ failed");
        assert_eq!(tensor.get_item(&[0, 0]).expect("get_item failed"), 0.0);
        tensor.ones_().expect("ones_ failed");
        assert_eq!(tensor.get_item(&[1, 1]).expect("get_item failed"), 1.0);
    }
    #[test]
    fn test_item_access() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        assert_eq!(tensor.get_item(&[0, 0]).expect("get_item failed"), 1.0);
        assert_eq!(tensor.get_item(&[1, 2]).expect("get_item failed"), 6.0);
        tensor.set_item(&[0, 1], 10.0).expect("set_item failed");
        assert_eq!(tensor.get_item(&[0, 1]).expect("get_item failed"), 10.0);
        assert_eq!(tensor.get_item_flat(0).expect("get_item_flat failed"), 1.0);
        tensor.set_item_flat(0, 15.0).expect("set_item_flat failed");
        assert_eq!(tensor.get_item_flat(0).expect("get_item_flat failed"), 15.0);
    }
    #[test]
    fn test_gather_1d() {
        let data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let tensor = Tensor::from_data(data, vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let indices = Tensor::from_data(vec![0i64, 2, 4], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let gathered = tensor.gather(0, &indices).expect("gather failed");
        assert_eq!(
            gathered.to_vec().expect("to_vec failed"),
            vec![10.0, 30.0, 50.0]
        );
    }
    #[test]
    fn test_scatter_1d() {
        let tensor =
            Tensor::<f32>::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![1i64, 3], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let src = Tensor::from_data(vec![100.0f32, 200.0], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let scattered = tensor.scatter(0, &indices, &src).expect("scatter failed");
        let result = scattered.to_vec().expect("to_vec failed");
        assert_eq!(result[1], 100.0);
        assert_eq!(result[3], 200.0);
        assert_eq!(result[0], 0.0);
    }
    #[test]
    fn test_repeat() {
        let data = vec![1.0f32, 2.0];
        let tensor = Tensor::from_data(data, vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let repeated = tensor.repeat(&[3]).expect("repeat failed");
        assert_eq!(repeated.shape().dims(), &[6]);
        assert_eq!(
            repeated.to_vec().expect("to_vec failed"),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }
    #[test]
    fn test_copy_() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let mut tensor1 = Tensor::from_data(data1, vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let data2 = vec![4.0f32, 5.0, 6.0];
        let tensor2 = Tensor::from_data(data2, vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        tensor1.copy_(&tensor2).expect("copy_ failed");
        assert_eq!(
            tensor1.to_vec().expect("to_vec failed"),
            vec![4.0, 5.0, 6.0]
        );
    }
    #[test]
    fn test_from_scalar() {
        let tensor = Tensor::<f32>::from_scalar(42.0, &[2, 3], DeviceType::Cpu)
            .expect("failed to create tensor from scalar");
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(tensor.get_item(&[i, j]).expect("get_item failed"), 42.0);
            }
        }
    }
    #[test]
    fn test_multi_to_flat_index() {
        let tensor = Tensor::<f32>::zeros(&[2, 3, 4], DeviceType::Cpu)
            .expect("failed to create zeros tensor");
        assert_eq!(
            tensor
                .multi_to_flat_index(&[0, 0, 0])
                .expect("multi_to_flat_index failed"),
            0
        );
        assert_eq!(
            tensor
                .multi_to_flat_index(&[1, 2, 3])
                .expect("multi_to_flat_index failed"),
            23
        );
        assert_eq!(
            tensor
                .multi_to_flat_index(&[1, 0, 0])
                .expect("multi_to_flat_index failed"),
            12
        );
    }
    #[test]
    fn test_error_handling() {
        let tensor =
            Tensor::<f32>::zeros(&[2, 3], DeviceType::Cpu).expect("failed to create zeros tensor");
        assert!(tensor.get_item(&[2, 0]).is_err());
        assert!(tensor.get_item(&[0, 3]).is_err());
        assert!(tensor.get_item(&[0]).is_err());
        assert!(tensor.get_item(&[0, 1, 2]).is_err());
    }
    #[test]
    fn test_index_add_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let index = Tensor::from_data(vec![0i64, 2, 4], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let source = Tensor::from_data(vec![10.0f32, 20.0, 30.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_add(0, &index, &source)
            .expect("index_add failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![11.0, 2.0, 23.0, 4.0, 35.0]
        );
    }
    #[test]
    fn test_index_add_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let index = Tensor::from_data(vec![0i64, 2], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let source =
            Tensor::from_data(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2], DeviceType::Cpu)
                .expect("failed to create tensor from data");
        let result = tensor
            .index_add(1, &index, &source)
            .expect("index_add failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![11.0, 2.0, 23.0, 34.0, 5.0, 46.0]
        );
    }
    #[test]
    fn test_index_add_negative_dim() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let index = Tensor::from_data(vec![0i64], vec![1], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let source = Tensor::from_data(vec![5.0f32, 6.0], vec![2, 1], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_add(-1, &index, &source)
            .expect("index_add failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![6.0, 2.0, 9.0, 4.0]
        );
    }
    #[test]
    fn test_index_copy_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let index = Tensor::from_data(vec![1i64, 3], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let source = Tensor::from_data(vec![100.0f32, 200.0], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_copy(0, &index, &source)
            .expect("index_copy failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 100.0, 3.0, 200.0, 5.0]
        );
    }
    #[test]
    fn test_index_copy_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let index = Tensor::from_data(vec![0i64, 2], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let source = Tensor::from_data(
            vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let result = tensor
            .index_copy(0, &index, &source)
            .expect("index_copy failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0]
        );
    }
    #[test]
    fn test_index_copy_negative_dim() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let index = Tensor::from_data(vec![1i64], vec![1], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let source = Tensor::from_data(vec![9.0f32, 8.0], vec![2, 1], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_copy(-1, &index, &source)
            .expect("index_copy failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 9.0, 3.0, 8.0]
        );
    }
    #[test]
    fn test_index_fill_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let index = Tensor::from_data(vec![1i64, 3], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_fill(0, &index, 99.0)
            .expect("index_fill failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 99.0, 3.0, 99.0, 5.0]
        );
    }
    #[test]
    fn test_index_fill_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let index = Tensor::from_data(vec![0i64, 2], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_fill(1, &index, -1.0)
            .expect("index_fill failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![-1.0, 2.0, -1.0, -1.0, 5.0, -1.0]
        );
    }
    #[test]
    fn test_index_fill_multiple_indices() {
        let tensor = Tensor::from_data(vec![0.0f32; 10], vec![10], DeviceType::Cpu)
            .expect("operation failed");
        let index = Tensor::from_data(vec![0i64, 2, 4, 6, 8], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_fill(0, &index, 1.0)
            .expect("index_fill failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
    }
    #[test]
    fn test_index_fill_negative_dim() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let index = Tensor::from_data(vec![0i64], vec![1], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_fill(-2, &index, 7.0)
            .expect("index_fill failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![7.0, 7.0, 3.0, 4.0]
        );
    }
    #[test]
    fn test_scatter_add_1d() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![0i64, 1, 2, 0, 1], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .scatter_add(0, &indices, &src)
            .expect("scatter_add failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![5.0, 7.0, 3.0, 0.0, 0.0]
        );
    }
    #[test]
    fn test_scatter_add_2d() {
        let tensor = Tensor::ones(&[3, 3], DeviceType::Cpu).expect("failed to create ones tensor");
        let indices = Tensor::from_data(
            vec![0i64, 2, 1, 1, 0, 2, 2, 1, 0],
            vec![3, 3],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let src = Tensor::from_data(
            vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            vec![3, 3],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let result = tensor
            .scatter_add(1, &indices, &src)
            .expect("scatter_add failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![11.0, 31.0, 21.0, 51.0, 41.0, 61.0, 91.0, 81.0, 71.0]
        );
    }
    #[test]
    fn test_scatter_add_negative_index() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![-1i64, -2], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let src = Tensor::from_data(vec![10.0f32, 20.0], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .scatter_add(0, &indices, &src)
            .expect("scatter_add failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![0.0, 0.0, 0.0, 20.0, 10.0]
        );
    }
    #[test]
    fn test_put_basic() {
        let tensor =
            Tensor::zeros(&[3, 3], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![0i64, 4, 8], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let values = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor.put_(&indices, &values).expect("put_ failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        );
    }
    #[test]
    fn test_put_1d() {
        let tensor = Tensor::zeros(&[10], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![1i64, 3, 5, 7, 9], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let values = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor.put_(&indices, &values).expect("put_ failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0]
        );
    }
    #[test]
    fn test_put_negative_indices() {
        let tensor = Tensor::ones(&[5], DeviceType::Cpu).expect("failed to create ones tensor");
        let indices = Tensor::from_data(vec![-1i64, -3], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let values = Tensor::from_data(vec![99.0f32, 88.0], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor.put_(&indices, &values).expect("put_ failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 1.0, 88.0, 1.0, 99.0]
        );
    }
    #[test]
    fn test_put_overwrite() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![0i64, 1, 0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let values = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor.put_(&indices, &values).expect("put_ failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![3.0, 2.0, 0.0, 0.0, 0.0]
        );
    }
    #[test]
    fn test_masked_scatter_basic() {
        let tensor =
            Tensor::zeros(&[3, 3], DeviceType::Cpu).expect("failed to create zeros tensor");
        let mask = Tensor::from_data(
            vec![true, false, false, false, true, false, false, false, true],
            vec![3, 3],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let source = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .masked_scatter(&mask, &source)
            .expect("masked_scatter failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        );
    }
    #[test]
    fn test_masked_scatter_1d() {
        let tensor = Tensor::ones(&[10], DeviceType::Cpu).expect("failed to create ones tensor");
        let mask = Tensor::from_data(
            vec![
                false, true, true, false, false, true, false, true, false, true,
            ],
            vec![10],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let source = Tensor::from_data(
            vec![10.0f32, 20.0, 30.0, 40.0, 50.0],
            vec![5],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let result = tensor
            .masked_scatter(&mask, &source)
            .expect("masked_scatter failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 10.0, 20.0, 1.0, 1.0, 30.0, 1.0, 40.0, 1.0, 50.0]
        );
    }
    #[test]
    fn test_masked_scatter_excess_source() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let mask = Tensor::from_data(
            vec![true, false, true, false, false],
            vec![5],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let source = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .masked_scatter(&mask, &source)
            .expect("masked_scatter failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 0.0, 2.0, 0.0, 0.0]
        );
    }
    #[test]
    fn test_masked_scatter_insufficient_source() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let mask = Tensor::from_data(
            vec![true, false, true, false, true],
            vec![5],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let source = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor.masked_scatter(&mask, &source);
        assert!(result.is_err());
    }
    #[test]
    fn test_index_put_2d() {
        let tensor =
            Tensor::zeros(&[3, 3], DeviceType::Cpu).expect("failed to create zeros tensor");
        let row_idx = Tensor::from_data(vec![0i64, 1, 2], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let col_idx = Tensor::from_data(vec![1i64, 2, 0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let values = Tensor::from_data(vec![10.0f32, 20.0, 30.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_put(&[row_idx, col_idx], &values)
            .expect("index_put failed");
        let result_data = result.to_vec().expect("to_vec failed");
        assert_eq!(result_data[1], 10.0);
        assert_eq!(result_data[5], 20.0);
        assert_eq!(result_data[6], 30.0);
    }
    #[test]
    fn test_index_put_1d() {
        let tensor = Tensor::zeros(&[10], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![1i64, 3, 5, 7], vec![4], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let values = Tensor::from_data(vec![10.0f32, 20.0, 30.0, 40.0], vec![4], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_put(&[indices], &values)
            .expect("index_put failed");
        let expected = vec![0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0, 0.0, 0.0];
        assert_eq!(result.to_vec().expect("to_vec failed"), expected);
    }
    #[test]
    fn test_index_put_broadcast() {
        let tensor = Tensor::ones(&[5], DeviceType::Cpu).expect("failed to create ones tensor");
        let indices = Tensor::from_data(vec![1i64, 2, 3], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let value = Tensor::from_data(vec![99.0f32], vec![1], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_put(&[indices], &value)
            .expect("index_put failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 99.0, 99.0, 99.0, 1.0]
        );
    }
    #[test]
    fn test_index_put_negative_indices() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![-1i64, -2], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let values = Tensor::from_data(vec![10.0f32, 20.0], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .index_put(&[indices], &values)
            .expect("index_put failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![0.0, 0.0, 0.0, 20.0, 10.0]
        );
    }
    #[test]
    fn test_scatter_reduce_sum() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![0i64, 1, 2, 0, 1], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .scatter_reduce(0, &indices, &src, "sum")
            .expect("scatter_reduce failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![5.0, 7.0, 3.0, 0.0, 0.0]
        );
    }
    #[test]
    fn test_scatter_reduce_prod() {
        let tensor = Tensor::ones(&[5], DeviceType::Cpu).expect("failed to create ones tensor");
        let indices = Tensor::from_data(vec![0i64, 1, 2, 0, 1], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let src = Tensor::from_data(vec![2.0f32, 3.0, 4.0, 5.0, 6.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .scatter_reduce(0, &indices, &src, "prod")
            .expect("scatter_reduce failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![10.0, 18.0, 4.0, 1.0, 1.0]
        );
    }
    #[test]
    fn test_scatter_reduce_amax() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(vec![0i64, 1, 2, 0, 1], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .scatter_reduce(0, &indices, &src, "amax")
            .expect("scatter_reduce failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![4.0, 5.0, 3.0, 0.0, 0.0]
        );
    }
    #[test]
    fn test_scatter_reduce_amin() {
        let tensor = Tensor::from_data(vec![10.0f32; 5], vec![5], DeviceType::Cpu)
            .expect("operation failed");
        let indices = Tensor::from_data(vec![0i64, 1, 2, 0, 1], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .scatter_reduce(0, &indices, &src, "amin")
            .expect("scatter_reduce failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 2.0, 3.0, 10.0, 10.0]
        );
    }
    #[test]
    fn test_scatter_reduce_2d() {
        let tensor =
            Tensor::zeros(&[3, 3], DeviceType::Cpu).expect("failed to create zeros tensor");
        let indices = Tensor::from_data(
            vec![0i64, 0, 1, 1, 1, 2, 2, 2, 2],
            vec![3, 3],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let src = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let result = tensor
            .scatter_reduce(1, &indices, &src, "sum")
            .expect("scatter_reduce failed");
        let result_data = result.to_vec().expect("to_vec failed");
        assert_eq!(result_data[0], 3.0);
        assert_eq!(result_data[1], 3.0);
        assert_eq!(result_data[4], 9.0);
        assert_eq!(result_data[5], 6.0);
        assert_eq!(result_data[8], 24.0);
    }
    #[test]
    fn test_diagonal_scatter_main() {
        let tensor =
            Tensor::zeros(&[3, 3], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .diagonal_scatter(&src, 0, 0, 1)
            .expect("diagonal_scatter failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        );
    }
    #[test]
    fn test_diagonal_scatter_above() {
        let tensor =
            Tensor::zeros(&[3, 4], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src = Tensor::from_data(vec![10.0f32, 20.0, 30.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .diagonal_scatter(&src, 1, 0, 1)
            .expect("diagonal_scatter failed");
        let result_data = result.to_vec().expect("to_vec failed");
        assert_eq!(result_data[1], 10.0);
        assert_eq!(result_data[6], 20.0);
        assert_eq!(result_data[11], 30.0);
    }
    #[test]
    fn test_diagonal_scatter_below() {
        let tensor =
            Tensor::zeros(&[4, 3], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src = Tensor::from_data(vec![5.0f32, 6.0, 7.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .diagonal_scatter(&src, -1, 0, 1)
            .expect("diagonal_scatter failed");
        let result_data = result.to_vec().expect("to_vec failed");
        assert_eq!(result_data[3], 5.0);
        assert_eq!(result_data[7], 6.0);
        assert_eq!(result_data[11], 7.0);
    }
    #[test]
    fn test_diagonal_scatter_2x2() {
        let tensor = Tensor::ones(&[2, 2], DeviceType::Cpu).expect("failed to create ones tensor");
        let src = Tensor::from_data(vec![99.0f32, 88.0], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .diagonal_scatter(&src, 0, 0, 1)
            .expect("diagonal_scatter failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![99.0, 1.0, 1.0, 88.0]
        );
    }
    #[test]
    fn test_select_scatter_2d() {
        let tensor =
            Tensor::zeros(&[3, 4], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .select_scatter(&src, 0, 1)
            .expect("select_scatter failed");
        let result_data = result.to_vec().expect("to_vec failed");
        assert_eq!(result_data[4], 1.0);
        assert_eq!(result_data[5], 2.0);
        assert_eq!(result_data[6], 3.0);
        assert_eq!(result_data[7], 4.0);
    }
    #[test]
    fn test_select_scatter_3d() {
        let tensor = Tensor::<f32>::zeros(&[2, 3, 4], DeviceType::Cpu)
            .expect("failed to create zeros tensor");
        let src =
            Tensor::<f32>::ones(&[2, 4], DeviceType::Cpu).expect("failed to create ones tensor");
        let result = tensor
            .select_scatter(&src, 1, 1)
            .expect("select_scatter failed");
        let result_data = result.to_vec().expect("to_vec failed");
        for i in 4..8 {
            assert_eq!(result_data[i], 1.0);
        }
        for i in 16..20 {
            assert_eq!(result_data[i], 1.0);
        }
    }
    #[test]
    fn test_select_scatter_negative_dim() {
        let tensor = Tensor::<f32>::zeros(&[3, 4, 5], DeviceType::Cpu)
            .expect("failed to create zeros tensor");
        let src =
            Tensor::<f32>::ones(&[3, 4], DeviceType::Cpu).expect("failed to create ones tensor");
        let result = tensor
            .select_scatter(&src, -1, -1)
            .expect("select_scatter failed");
        let result_data = result.to_vec().expect("to_vec failed");
        for i in 0..3 {
            for j in 0..4 {
                let idx = i * 20 + j * 5 + 4;
                assert_eq!(result_data[idx], 1.0);
            }
        }
    }
    #[test]
    fn test_select_scatter_1d() {
        let tensor = Tensor::<f32>::zeros(&[2, 1, 3], DeviceType::Cpu)
            .expect("failed to create zeros tensor");
        let src =
            Tensor::<f32>::ones(&[2, 3], DeviceType::Cpu).expect("failed to create ones tensor");
        let result = tensor
            .select_scatter(&src, 1, 0)
            .expect("select_scatter failed");
        let result_data = result.to_vec().expect("to_vec failed");
        for &val in result_data.iter() {
            assert_eq!(val, 1.0);
        }
    }
    #[test]
    fn test_slice_scatter_basic() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src = Tensor::from_data(vec![10.0f32, 20.0], vec![2], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .slice_scatter(&src, 0, Some(1), Some(3), 1)
            .expect("slice_scatter failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![0.0, 10.0, 20.0, 0.0, 0.0]
        );
    }
    #[test]
    fn test_slice_scatter_2d_rows() {
        let tensor =
            Tensor::<f32>::zeros(&[5, 3], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src =
            Tensor::<f32>::ones(&[2, 3], DeviceType::Cpu).expect("failed to create ones tensor");
        let result = tensor
            .slice_scatter(&src, 0, Some(1), Some(3), 1)
            .expect("slice_scatter failed");
        let result_data = result.to_vec().expect("to_vec failed");
        for i in 3..9 {
            assert_eq!(result_data[i], 1.0);
        }
        assert_eq!(result_data[0], 0.0);
        assert_eq!(result_data[9], 0.0);
    }
    #[test]
    fn test_slice_scatter_2d_cols() {
        let tensor =
            Tensor::zeros(&[3, 5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
            DeviceType::Cpu,
        )
        .expect("failed to create tensor from data");
        let result = tensor
            .slice_scatter(&src, 1, Some(1), Some(3), 1)
            .expect("slice_scatter failed");
        let result_data = result.to_vec().expect("to_vec failed");
        assert_eq!(result_data[1], 1.0);
        assert_eq!(result_data[2], 2.0);
        assert_eq!(result_data[6], 3.0);
        assert_eq!(result_data[7], 4.0);
        assert_eq!(result_data[11], 5.0);
        assert_eq!(result_data[12], 6.0);
    }
    #[test]
    fn test_slice_scatter_step() {
        let tensor = Tensor::zeros(&[10], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .slice_scatter(&src, 0, Some(0), Some(6), 2)
            .expect("slice_scatter failed");
        let expected = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(result.to_vec().expect("to_vec failed"), expected);
    }
    #[test]
    fn test_slice_scatter_negative_indices() {
        let tensor = Tensor::zeros(&[10], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src = Tensor::from_data(vec![7.0f32, 8.0, 9.0], vec![3], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .slice_scatter(&src, 0, Some(-5), Some(-2), 1)
            .expect("slice_scatter failed");
        let mut expected = vec![0.0; 10];
        expected[5] = 7.0;
        expected[6] = 8.0;
        expected[7] = 9.0;
        assert_eq!(result.to_vec().expect("to_vec failed"), expected);
    }
    #[test]
    fn test_slice_scatter_none_bounds() {
        let tensor = Tensor::zeros(&[5], DeviceType::Cpu).expect("failed to create zeros tensor");
        let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .slice_scatter(&src, 0, None, None, 1)
            .expect("slice_scatter failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }
    #[test]
    fn test_slice_scatter_empty_slice() {
        let tensor =
            Tensor::<f32>::ones(&[5], DeviceType::Cpu).expect("failed to create ones tensor");
        let src = Tensor::from_data(vec![], vec![0], DeviceType::Cpu)
            .expect("failed to create tensor from data");
        let result = tensor
            .slice_scatter(&src, 0, Some(3), Some(1), 1)
            .expect("slice_scatter failed");
        assert_eq!(
            result.to_vec().expect("to_vec failed"),
            vec![1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }
}
