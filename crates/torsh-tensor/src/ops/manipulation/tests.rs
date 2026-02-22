#[cfg(test)]
mod tests {
    use crate::{Tensor, TensorElement};
    use torsh_core::device::DeviceType;

    #[test]
    fn test_cat_1d() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).expect("tensor creation failed");
        let b = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).expect("tensor creation failed");

        let result = Tensor::cat(&[a, b], 0).expect("cat failed");

        assert_eq!(result.shape().dims(), &[4]);
        let data = result.data().expect("data retrieval failed");
        assert_eq!(data.len(), 4);
    }

    #[test]
    fn test_stack_1d() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).expect("tensor creation failed");
        let b = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).expect("tensor creation failed");

        let result = Tensor::stack(&[a, b], 0).expect("stack failed");

        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_stack_shape_mismatch() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).expect("tensor creation failed");
        let b = Tensor::from_data(vec![3.0f32, 4.0, 5.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        assert!(Tensor::stack(&[a, b], 0).is_err());
    }

    #[test]
    fn test_flip_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).expect("tensor creation failed");

        let flipped = tensor.flip(&[0]).expect("flip failed");
        let result = flipped.data().expect("data retrieval failed");

        assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_roll_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).expect("tensor creation failed");

        let rolled = tensor.roll(&[2], &[0]).expect("roll failed");
        let result = rolled.data().expect("data retrieval failed");

        assert_eq!(result, vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_chunk() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![6],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let chunks = tensor.chunk(3, 0).expect("chunk failed");

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape().dims(), &[2]);
        assert_eq!(chunks[1].shape().dims(), &[2]);
        assert_eq!(chunks[2].shape().dims(), &[2]);
    }

    #[test]
    fn test_split() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
            vec![5],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let splits = tensor.split(2, 0).expect("split failed");

        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].shape().dims(), &[2]);
        assert_eq!(splits[1].shape().dims(), &[2]);
        assert_eq!(splits[2].shape().dims(), &[1]); // Last one is smaller
    }

    #[test]
    fn test_fliplr() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let flipped = tensor.fliplr().expect("fliplr failed");
        let result = flipped.data().expect("data retrieval failed");

        // [[1, 2, 3], [4, 5, 6]] -> [[3, 2, 1], [6, 5, 4]]
        assert_eq!(result, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_flipud() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let flipped = tensor.flipud().expect("flipud failed");
        let result = flipped.data().expect("data retrieval failed");

        // [[1, 2, 3], [4, 5, 6]] -> [[4, 5, 6], [1, 2, 3]]
        assert_eq!(result, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rot90_once() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let rotated = tensor.rot90(1, &[0, 1]).expect("rot90 failed");
        let result = rotated.data().expect("data retrieval failed");
        let shape = rotated.shape().dims();

        assert_eq!(shape, &[2, 2]);
        // [[1, 2], [3, 4]] -> [[2, 4], [1, 3]] (90 degrees counter-clockwise)
        assert_eq!(result, vec![2.0, 4.0, 1.0, 3.0]);
    }

    #[test]
    fn test_rot90_twice() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let rotated = tensor.rot90(2, &[0, 1]).expect("rot90 failed");
        let result = rotated.data().expect("data retrieval failed");

        // 180 degrees: [[1, 2], [3, 4]] -> [[4, 3], [2, 1]]
        assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_rot90_negative() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let rotated = tensor.rot90(-1, &[0, 1]).expect("rot90 failed");
        let result = rotated.data().expect("data retrieval failed");

        // -90 degrees (clockwise): [[1, 2], [3, 4]] -> [[3, 1], [4, 2]]
        assert_eq!(result, vec![3.0, 1.0, 4.0, 2.0]);
    }

    #[test]
    fn test_tile_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).expect("tensor creation failed");

        let tiled = tensor.tile(&[3]).expect("tile failed");
        let result = tiled.data().expect("data retrieval failed");
        let shape = tiled.shape().dims();

        assert_eq!(shape, &[6]); // 2 * 3
        assert_eq!(result, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tile_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let tiled = tensor.tile(&[2, 3]).expect("tile failed");
        let result = tiled.data().expect("data retrieval failed");
        let shape = tiled.shape().dims();

        assert_eq!(shape, &[4, 6]); // [2*2, 2*3]
        // [[1, 2], [3, 4]] tiled with [2, 3] should repeat pattern
        assert_eq!(result.len(), 24);
    }

    #[test]
    fn test_tile_expand_dims() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).expect("tensor creation failed");

        let tiled = tensor.tile(&[3, 2]).expect("tile failed");
        let shape = tiled.shape().dims();

        assert_eq!(shape, &[3, 4]); // Expands to 2D and tiles
    }

    #[test]
    fn test_cat_2d() {
        let a = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");
        let b = Tensor::from_data(
            vec![5.0f32, 6.0, 7.0, 8.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let result = Tensor::cat(&[a, b], 0).expect("cat failed");

        assert_eq!(result.shape().dims(), &[4, 2]);
        let data = result.data().expect("data retrieval failed");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_stack_2d() {
        let a = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");
        let b = Tensor::from_data(
            vec![5.0f32, 6.0, 7.0, 8.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let result = Tensor::stack(&[a, b], 0).expect("stack failed");

        assert_eq!(result.shape().dims(), &[2, 2, 2]);
    }

    #[test]
    fn test_repeat_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        let result = tensor.repeat(&[4]).expect("repeat failed");

        assert_eq!(result.shape().dims(), &[12]); // 3 * 4
        let data = result.data().expect("data retrieval failed");
        assert_eq!(
            data,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_repeat_2d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation failed");

        let result = tensor.repeat(&[2, 3]).expect("repeat failed");

        assert_eq!(result.shape().dims(), &[4, 6]); // [2*2, 2*3]
        assert_eq!(result.data().expect("data retrieval failed").len(), 24);
    }

    #[test]
    fn test_repeat_expand_dims() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).expect("tensor creation failed");

        let result = tensor.repeat(&[3, 2]).expect("repeat failed");

        assert_eq!(result.shape().dims(), &[3, 4]); // Expands to [1, 2] then repeats to [3, 4]
    }

    #[test]
    fn test_repeat_interleave_1d_no_dim() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        let result = tensor.repeat_interleave(2, None).expect("repeat_interleave failed");

        assert_eq!(result.shape().dims(), &[6]); // 3 * 2
        let data = result.data().expect("data retrieval failed");
        assert_eq!(data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_repeat_interleave_1d_with_dim() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        let result = tensor.repeat_interleave(3, Some(0)).expect("repeat_interleave failed");

        assert_eq!(result.shape().dims(), &[9]); // 3 * 3
        let data = result.data().expect("data retrieval failed");
        assert_eq!(
            data,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn test_repeat_interleave_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let result = tensor.repeat_interleave(2, Some(0)).expect("repeat_interleave failed");

        assert_eq!(result.shape().dims(), &[4, 3]); // Repeat along dim 0
        let data = result.data().expect("data retrieval failed");
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]
        assert_eq!(
            data,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_repeat_interleave_2d_dim1() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        let result = tensor.repeat_interleave(2, Some(1)).expect("repeat_interleave failed");

        assert_eq!(result.shape().dims(), &[2, 6]); // Repeat along dim 1
        let data = result.data().expect("data retrieval failed");
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]]
        assert_eq!(
            data,
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
        );
    }

    #[test]
    fn test_unflatten_1d_to_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![6],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let result = tensor.unflatten(0, &[2, 3]).expect("unflatten failed");

        assert_eq!(result.shape().dims(), &[2, 3]);
        let data = result.data().expect("data retrieval failed");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unflatten_2d_to_3d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let result = tensor.unflatten(1, &[2, 2]).expect("unflatten failed");

        assert_eq!(result.shape().dims(), &[2, 2, 2]);
        assert_eq!(result.data().expect("data retrieval failed").len(), 8);
    }

    #[test]
    fn test_unflatten_size_mismatch() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).expect("tensor creation failed");

        let result = tensor.unflatten(0, &[2, 3]); // 2*3=6 != 4

        assert!(result.is_err());
    }

    #[test]
    fn test_take_along_dim_1d() {
        let tensor = Tensor::from_data(vec![10.0f32, 20.0, 30.0, 40.0], vec![4], DeviceType::Cpu).expect("tensor creation failed");
        let indices = Tensor::from_data(vec![3i64, 1, 2], vec![3], DeviceType::Cpu).expect("indices creation failed");

        let result = tensor.take_along_dim(&indices, None).expect("take_along_dim failed");
        let data = result.data().expect("data retrieval failed");

        assert_eq!(data, vec![40.0, 20.0, 30.0]);
    }

    #[test]
    fn test_take_along_dim_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).expect("tensor creation failed");
        let indices = Tensor::from_data(
            vec![2i64, 0, 1, 2],
            vec![2, 2],
            DeviceType::Cpu,
        ).expect("indices creation failed");

        let result = tensor.take_along_dim(&indices, Some(1)).expect("take_along_dim failed");
        let data = result.data().expect("data retrieval failed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        // Row 0: [1, 2, 3] with indices [2, 0] -> [3, 1]
        // Row 1: [4, 5, 6] with indices [1, 2] -> [5, 6]
        assert_eq!(data, vec![3.0, 1.0, 5.0, 6.0]);
    }

    #[test]
    fn test_take_along_dim_out_of_range() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let indices = Tensor::from_data(vec![5i64], vec![1], DeviceType::Cpu).expect("indices creation failed");

        let result = tensor.take_along_dim(&indices, None);

        assert!(result.is_err()); // Index 5 out of range for size 3
    }

    #[test]
    fn test_take_along_dim_argmax_use_case() {
        // Common use case: gather max values using argmax indices
        let tensor = Tensor::from_data(
            vec![1.0f32, 5.0, 3.0, 2.0, 7.0, 4.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Get argmax along dim 1
        let argmax_result = tensor.argmax(Some(1), true).expect("argmax failed");

        // Use take_along_dim to gather the max values
        let max_values = tensor.take_along_dim(&argmax_result, Some(1)).expect("take_along_dim failed");
        let data = max_values.data().expect("data retrieval failed");

        assert_eq!(max_values.shape().dims(), &[2, 1]);
        // Row 0: max is 5.0, Row 1: max is 7.0
        assert_eq!(data, vec![5.0, 7.0]);
    }

    #[test]
    fn test_movedim_single() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Move dim 0 to position 2: [2,3,4] -> [3,4,2]
        let result = tensor.movedim(&[0], &[2]).expect("movedim failed");

        assert_eq!(result.shape().dims(), &[3, 4, 2]);
    }

    #[test]
    fn test_movedim_multiple() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Move dims [0, 1] to positions [2, 0]: [2,3,4] -> [3,4,2]
        let result = tensor.movedim(&[0, 1], &[2, 0]).expect("movedim failed");

        assert_eq!(result.shape().dims(), &[3, 4, 2]);
    }

    #[test]
    fn test_movedim_negative_indices() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Move last dim to first position: [2,3,4] -> [4,2,3]
        let result = tensor.movedim(&[-1], &[0]).expect("movedim failed");

        assert_eq!(result.shape().dims(), &[4, 2, 3]);
    }

    #[test]
    fn test_movedim_length_mismatch() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let result = tensor.movedim(&[0, 1], &[2]);

        assert!(result.is_err()); // source and destination must have same length
    }

    #[test]
    fn test_movedim_duplicate_source() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let result = tensor.movedim(&[0, 0], &[1, 2]);

        assert!(result.is_err()); // Repeated dim in source
    }

    #[test]
    fn test_movedim_duplicate_destination() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let result = tensor.movedim(&[0, 1], &[2, 2]);

        assert!(result.is_err()); // Repeated dim in destination
    }

    #[test]
    fn test_moveaxis_alias() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let result1 = tensor.movedim(&[0], &[2]).expect("movedim failed");
        let result2 = tensor.moveaxis(&[0], &[2]).expect("moveaxis failed");

        assert_eq!(result1.shape().dims(), result2.shape().dims());
    }

    #[test]
    fn test_swapaxes_simple() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Swap dims 0 and 1: [2,3] -> [3,2]
        let result = tensor.swapaxes(0, 1).expect("swapaxes failed");

        assert_eq!(result.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_swapaxes_3d() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Swap dims 0 and 2: [2,3,4] -> [4,3,2]
        let result = tensor.swapaxes(0, 2).expect("swapaxes failed");

        assert_eq!(result.shape().dims(), &[4, 3, 2]);
    }

    #[test]
    fn test_swapaxes_negative_indices() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Swap last two dims: [2,3,4] -> [2,4,3]
        let result = tensor.swapaxes(-1, -2).expect("swapaxes failed");

        assert_eq!(result.shape().dims(), &[2, 4, 3]);
    }

    #[test]
    fn test_swapaxes_same_dim() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Swapping same dimension should return identical shape
        let result = tensor.swapaxes(1, 1).expect("swapaxes failed");

        assert_eq!(result.shape().dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_swapaxes_out_of_range() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let result = tensor.swapaxes(0, 5);

        assert!(result.is_err()); // Dimension 5 out of range for 3-D tensor
    }

    #[test]
    fn test_swapdims_alias() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let result1 = tensor.swapaxes(0, 2).expect("swapaxes failed");
        let result2 = tensor.swapdims(0, 2).expect("swapdims failed");

        assert_eq!(result1.shape().dims(), result2.shape().dims());
    }

    #[test]
    fn test_movedim_integration_with_data() {
        // Test that data is actually rearranged correctly
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // [[1, 2, 3], [4, 5, 6]] with shape [2, 3]
        // Move dim 1 to position 0: should become [3, 2]
        let result = tensor.movedim(&[1], &[0]).expect("movedim failed");

        assert_eq!(result.shape().dims(), &[3, 2]);
        // After transpose: [[1, 4], [2, 5], [3, 6]]
        let data = result.data().expect("data retrieval failed");
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_swapaxes_integration_with_data() {
        // Test that data is actually rearranged correctly
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // [[1, 2, 3], [4, 5, 6]] with shape [2, 3]
        let result = tensor.swapaxes(0, 1).expect("swapaxes failed");

        assert_eq!(result.shape().dims(), &[3, 2]);
        // After transpose: [[1, 4], [2, 5], [3, 6]]
        let data = result.data().expect("data retrieval failed");
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_broadcast_to_same_shape() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let result = tensor.broadcast_to(&[2, 2]).expect("broadcast_to failed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(result.data().expect("data retrieval failed"), tensor.data().expect("data retrieval failed"));
    }

    #[test]
    fn test_broadcast_to_expand_dim() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Broadcast [2] to [3, 2]
        let result = tensor.broadcast_to(&[3, 2]).expect("broadcast_to failed");

        assert_eq!(result.shape().dims(), &[3, 2]);
        let data = result.data().expect("data retrieval failed");
        // Should repeat [1, 2] three times
        assert_eq!(data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_broadcast_to_expand_singleton() {
        let tensor = Tensor::from_data(
            vec![5.0f32],
            vec![1],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Broadcast [1] to [4]
        let result = tensor.broadcast_to(&[4]).expect("broadcast_to failed");

        assert_eq!(result.shape().dims(), &[4]);
        let data = result.data().expect("data retrieval failed");
        assert_eq!(data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_broadcast_to_2d_singleton() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3, 1],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Broadcast [3, 1] to [3, 4]
        let result = tensor.broadcast_to(&[3, 4]).expect("broadcast_to failed");

        assert_eq!(result.shape().dims(), &[3, 4]);
        let data = result.data().expect("data retrieval failed");
        // Each row should be repeated 4 times
        assert_eq!(
            data,
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn test_broadcast_to_add_leading_dims() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Broadcast [2] to [2, 3, 2]
        let result = tensor.broadcast_to(&[2, 3, 2]).expect("broadcast_to failed");

        assert_eq!(result.shape().dims(), &[2, 3, 2]);
        assert_eq!(result.data().expect("data retrieval failed").len(), 12); // 2 * 3 * 2
    }

    #[test]
    fn test_broadcast_to_incompatible() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Cannot broadcast [3] to [2]
        let result = tensor.broadcast_to(&[2]);

        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_to_fewer_dims() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Cannot broadcast [2, 2] to [2] (fewer dimensions)
        let result = tensor.broadcast_to(&[2]);

        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_to_complex_pattern() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![1, 3],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        // Broadcast [1, 3] to [2, 3]
        let result = tensor.broadcast_to(&[2, 3]).expect("broadcast_to failed");

        assert_eq!(result.shape().dims(), &[2, 3]);
        let data = result.data().expect("data retrieval failed");
        // [1, 2, 3] repeated twice
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_expand_as_basic() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let target = Tensor::from_data(
            vec![0.0f32; 6],
            vec![3, 2],
            DeviceType::Cpu,
        ).expect("target creation failed");

        let result = tensor.expand_as(&target).expect("expand_as failed");

        assert_eq!(result.shape().dims(), target.shape().dims());
        assert_eq!(result.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_expand_as_same_shape() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let target = Tensor::from_data(
            vec![0.0f32; 4],
            vec![2, 2],
            DeviceType::Cpu,
        ).expect("target creation failed");

        let result = tensor.expand_as(&target).expect("expand_as failed");

        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(result.data().expect("data retrieval failed"), tensor.data().expect("data retrieval failed"));
    }

    #[test]
    fn test_expand_as_with_singleton() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3, 1],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let target = Tensor::from_data(
            vec![0.0f32; 12],
            vec![3, 4],
            DeviceType::Cpu,
        ).expect("target creation failed");

        let result = tensor.expand_as(&target).expect("expand_as failed");

        assert_eq!(result.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_expand_as_incompatible() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3],
            DeviceType::Cpu,
        ).expect("tensor creation failed");

        let target = Tensor::from_data(
            vec![0.0f32; 2],
            vec![2],
            DeviceType::Cpu,
        ).expect("target creation failed");

        let result = tensor.expand_as(&target);

        assert!(result.is_err()); // Cannot broadcast [3] to [2]
    }
}
