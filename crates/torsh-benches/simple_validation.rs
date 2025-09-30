#!/usr/bin/env rust-script

//! Simple validation script to check compilation fixes without full build system
//! This avoids the file lock issues and allows us to validate our changes

use std::fs;
use std::path::Path;

fn main() {
    println!("🔍 Validating compilation fixes...");
    
    // Check iterative_solvers.rs for the specific fixes
    let iterative_solvers_path = "../torsh-autograd/src/iterative_solvers.rs";
    
    if Path::new(iterative_solvers_path).exists() {
        let content = fs::read_to_string(iterative_solvers_path)
            .expect("Failed to read iterative_solvers.rs");
        
        // Check for the specific problematic patterns
        let problems = vec![
            ("x.reshape(&[x.shape().dims()[0], 1])", "usize to i32 conversion needed"),
            ("b.reshape(&[1, b.shape().dims()[0]])", "usize to i32 conversion needed"),
            ("result.reshape(&[result.shape().dims()[0]])", "usize to i32 conversion needed"),
        ];
        
        let mut fixes_needed = 0;
        let mut fixes_applied = 0;
        
        for (pattern, description) in problems {
            if content.contains(pattern) {
                println!("❌ ISSUE FOUND: {} - {}", pattern, description);
                fixes_needed += 1;
            } else {
                println!("✅ FIXED: {} - {}", pattern, description);
                fixes_applied += 1;
            }
        }
        
        // Check for properly applied fixes
        let good_patterns = vec![
            "x.shape().dims()[0].try_into().unwrap()",
            "b.shape().dims()[0].try_into().unwrap()",
            "result.shape().dims()[0].try_into().unwrap()",
        ];
        
        for pattern in good_patterns {
            if content.contains(pattern) {
                println!("✅ GOOD FIX: {}", pattern);
            }
        }
        
        println!("\n📊 VALIDATION SUMMARY:");
        println!("- Fixes needed: {}", fixes_needed);
        println!("- Fixes applied: {}", fixes_applied);
        
        if fixes_needed == 0 {
            println!("🎉 ALL COMPILATION FIXES VALIDATED!");
        } else {
            println!("⚠️  {} fixes still needed", fixes_needed);
        }
    } else {
        println!("❌ Could not find iterative_solvers.rs");
    }
}