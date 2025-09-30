#!/usr/bin/env python3
"""
Comprehensive analysis of torsh-benches codebase for potential issues
that can be fixed without running the build system.
"""

import os
import re
import glob
from pathlib import Path

def analyze_rust_file(file_path):
    """Analyze a Rust file for common issues."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.splitlines()
    
    # Check for common issues
    for i, line in enumerate(lines, 1):
        # Check for unused variables (simple heuristic)
        if re.search(r'let\s+(?!_)\w+\s*=.*;\s*$', line.strip()):
            var_name = re.search(r'let\s+(\w+)\s*=', line.strip())
            if var_name:
                var_name = var_name.group(1)
                # Check if variable is used later in the file
                if var_name not in content[content.find(line):]:
                    issues.append(f"Line {i}: Potentially unused variable '{var_name}'")
        
        # Check for rand API usage (old version)
        if 'rand::' in line and ('gen_range' in line or 'thread_rng' in line):
            issues.append(f"Line {i}: Potential rand API version issue")
        
        # Check for format string issues
        if 'println!' in line or 'format!' in line or 'write!' in line:
            # Count format placeholders
            placeholders = len(re.findall(r'\{\}', line))
            if placeholders > 0:
                issues.append(f"Line {i}: Format string with {placeholders} placeholders - verify arguments")
        
        # Check for .unwrap() usage
        if '.unwrap()' in line:
            issues.append(f"Line {i}: .unwrap() usage - consider error handling")
    
    return issues

def analyze_directory(directory):
    """Analyze all Rust files in a directory."""
    rust_files = glob.glob(os.path.join(directory, '**/*.rs'), recursive=True)
    
    all_issues = {}
    
    for file_path in rust_files:
        rel_path = os.path.relpath(file_path, directory)
        issues = analyze_rust_file(file_path)
        if issues:
            all_issues[rel_path] = issues
    
    return all_issues

def main():
    print("🔍 Comprehensive Analysis of torsh-benches codebase...")
    print("=" * 60)
    
    # Analyze src directory
    src_dir = "src"
    if os.path.exists(src_dir):
        issues = analyze_directory(src_dir)
        
        total_issues = sum(len(file_issues) for file_issues in issues.values())
        
        print(f"📊 Found {total_issues} potential issues across {len(issues)} files")
        print()
        
        for file_path, file_issues in issues.items():
            print(f"📄 {file_path}:")
            for issue in file_issues[:10]:  # Show first 10 issues per file
                print(f"  ⚠️  {issue}")
            if len(file_issues) > 10:
                print(f"  ... and {len(file_issues) - 10} more issues")
            print()
    
    # Check for specific known issues
    print("🔍 Checking for specific known issues...")
    
    # Check for rand API issues
    rand_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith('.rs'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'rand::' in content or 'thread_rng' in content:
                        rand_files.append(file_path)
    
    if rand_files:
        print(f"📦 Found {len(rand_files)} files using rand API:")
        for file_path in rand_files:
            print(f"  - {os.path.relpath(file_path, '.')}")
    
    # Check Cargo.toml for version issues
    cargo_toml = "Cargo.toml"
    if os.path.exists(cargo_toml):
        with open(cargo_toml, 'r') as f:
            content = f.read()
            if 'rand' in content:
                print("📦 Found rand dependency in Cargo.toml")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()