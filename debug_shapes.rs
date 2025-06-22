use torsh_tensor::{tensor_2d, creation::*};

fn main() {
    let a = tensor_2d![[1.0, 2.0]];
    let b = tensor_2d![[3.0], [4.0]];
    
    println!("Tensor a shape: {:?}", a.shape().dims());
    println!("Tensor b shape: {:?}", b.shape().dims());
    
    println!("Are they broadcast compatible? {}", a.shape().broadcast_compatible(&b.shape()));
    
    match a.add(&b) {
        Ok(result) => {
            println!("Addition succeeded! Result shape: {:?}", result.shape().dims());
        },
        Err(e) => {
            println!("Addition failed with error: {:?}", e);
        }
    }
}