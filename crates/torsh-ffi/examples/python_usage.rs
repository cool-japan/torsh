use torsh_ffi::python::*;

fn main() {
    println!("ToRSh Python FFI Example");
    println!("This example demonstrates how to use ToRSh from Python");
    println!();

    // In a real scenario, this would be called from Python
    println!("Example Python usage:");
    println!("```python");
    println!("import torsh");
    println!();
    println!("# Create tensors");
    println!("x = torsh.tensor([[1.0, 2.0], [3.0, 4.0]])");
    println!("y = torsh.tensor([[5.0, 6.0], [7.0, 8.0]])");
    println!();
    println!("# Basic operations");
    println!("z = x + y");
    println!("print(z)  # tensor([[6.0, 8.0], [10.0, 12.0]])");
    println!();
    println!("# Matrix multiplication");
    println!("result = x.matmul(y)");
    println!("print(result)");
    println!();
    println!("# Neural network");
    println!("model = torsh.Linear(2, 1)");
    println!("output = model.forward(x)");
    println!();
    println!("# Optimizer");
    println!("optimizer = torsh.SGD(model.parameters(), lr=0.01)");
    println!("optimizer.step()");
    println!("```");
}
