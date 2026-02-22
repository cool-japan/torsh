//! Tests for indexing operations

use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::*;

#[test]
fn test_select() -> Result<()> {
    let t = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DeviceType::Cpu,
    )?;

    // Select first row
    let selected = t.select(0, 0)?;
    assert_eq!(selected.shape().dims(), &[3]);
    assert_eq!(selected.data().unwrap(), vec![1.0, 2.0, 3.0]);

    // Select second column
    let selected = t.select(1, 1)?;
    assert_eq!(selected.shape().dims(), &[2]);
    assert_eq!(selected.data().unwrap(), vec![2.0, 5.0]);

    Ok(())
}

#[test]
fn test_slice() -> Result<()> {
    let t = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
        DeviceType::Cpu,
    )?;

    // Slice rows from 0 to 2
    let sliced = t.slice(0, 0, 2)?;
    assert_eq!(sliced.shape().dims(), &[2, 3]);
    assert_eq!(sliced.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Slice columns from 1 to 3
    let sliced = t.slice(1, 1, 3)?;
    assert_eq!(sliced.shape().dims(), &[3, 2]);
    assert_eq!(sliced.to_vec().unwrap(), vec![2.0, 3.0, 5.0, 6.0, 8.0, 9.0]);

    Ok(())
}

#[test]
fn test_narrow() -> Result<()> {
    let t = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
        DeviceType::Cpu,
    )?;

    // Narrow along dim 0, start at 1, length 2
    let narrowed = t.narrow(0, 1, 2)?;
    assert_eq!(narrowed.shape().dims(), &[2, 3]);
    assert_eq!(narrowed.data().unwrap(), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // Narrow along dim 1, start at 0, length 2
    let narrowed = t.narrow(1, 0, 2)?;
    assert_eq!(narrowed.shape().dims(), &[3, 2]);
    assert_eq!(narrowed.data().unwrap(), vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0]);

    Ok(())
}

#[test]
fn test_masked_select() -> Result<()> {
    let t = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DeviceType::Cpu,
    )?;

    let mask = Tensor::from_data(
        vec![true, false, true, false, true, false],
        vec![2, 3],
        DeviceType::Cpu,
    )?;

    let selected = t.masked_select(&mask)?;
    assert_eq!(selected.shape().dims(), &[3]);
    assert_eq!(selected.data().unwrap(), vec![1.0, 3.0, 5.0]);

    Ok(())
}

#[test]
fn test_take() -> Result<()> {
    let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], DeviceType::Cpu)?;

    let indices = Tensor::from_data(vec![0i64, 2, 4], vec![3], DeviceType::Cpu)?;

    let taken = t.take(&indices)?;
    assert_eq!(taken.shape().dims(), &[3]);
    assert_eq!(taken.data().unwrap(), vec![1.0, 3.0, 5.0]);

    // Test negative indices
    let indices = Tensor::from_data(vec![-1i64, -3, -5], vec![3], DeviceType::Cpu)?;

    let taken = t.take(&indices)?;
    assert_eq!(taken.data().unwrap(), vec![6.0, 4.0, 2.0]);

    Ok(())
}

#[test]
fn test_put() -> Result<()> {
    let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], DeviceType::Cpu)?;

    let indices = Tensor::from_data(vec![0i64, 2, 4], vec![3], DeviceType::Cpu)?;

    let values = Tensor::from_data(vec![10.0, 30.0, 50.0], vec![3], DeviceType::Cpu)?;

    let result = t.put(&indices, &values)?;
    assert_eq!(
        result.data().unwrap(),
        vec![10.0, 2.0, 30.0, 4.0, 50.0, 6.0]
    );

    Ok(())
}

#[test]
fn test_where_tensor() -> Result<()> {
    let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu)?;

    let y = Tensor::from_data(vec![10.0, 20.0, 30.0, 40.0], vec![4], DeviceType::Cpu)?;

    let condition = Tensor::from_data(vec![true, false, true, false], vec![4], DeviceType::Cpu)?;

    let result = x.where_tensor(&condition, &y)?;
    assert_eq!(result.data().unwrap(), vec![1.0, 20.0, 3.0, 40.0]);

    Ok(())
}
