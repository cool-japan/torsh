use std::env;
use std::path::PathBuf;
use bindgen;
use cc;

fn main() {
    // Check if CUDA is available
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include = format!("{}/include", cuda_path);
    let cuda_lib64 = format!("{}/lib64", cuda_path);
    let cuda_lib = format!("{}/lib", cuda_path);

    println!("cargo:rustc-link-search=native={}", cuda_lib64);
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-lib=cudnn");

    // Generate bindings for custom CUDA kernels
    let bindings = bindgen::Builder::default()
        .header("src/kernels/cuda_kernels.h")
        .clang_arg(format!("-I{}", cuda_include))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("cuda_bindings.rs"))
        .expect("Couldn't write bindings!");

    // Compile CUDA kernels
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_50,code=sm_50")
        .flag("-gencode")
        .flag("arch=compute_60,code=sm_60")
        .flag("-gencode")
        .flag("arch=compute_70,code=sm_70")
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75")
        .flag("-gencode")
        .flag("arch=compute_80,code=sm_80")
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86")
        .include(&cuda_include)
        .file("src/kernels/tensor_ops.cu")
        .file("src/kernels/neural_ops.cu")
        .file("src/kernels/reduction_ops.cu")
        .compile("cuda_kernels");

    println!("cargo:rerun-if-changed=src/kernels/");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}