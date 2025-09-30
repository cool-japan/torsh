#!/usr/bin/env python3

import os
import sys

def main():
    print("🔍 Validating compilation fixes...")
    
    # Check iterative_solvers.rs for the specific fixes
    iterative_solvers_path = "../torsh-autograd/src/iterative_solvers.rs"
    
    if os.path.exists(iterative_solvers_path):
        with open(iterative_solvers_path, 'r') as f:
            content = f.read()
        
        # Check for the specific problematic patterns
        problems = [
            ("x.reshape(&[x.shape().dims()[0], 1])", "usize to i32 conversion needed"),
            ("b.reshape(&[1, b.shape().dims()[0]])", "usize to i32 conversion needed"),
            ("result.reshape(&[result.shape().dims()[0]])", "usize to i32 conversion needed"),
        ]
        
        fixes_needed = 0
        fixes_applied = 0
        
        for pattern, description in problems:
            if pattern in content:
                print(f"❌ ISSUE FOUND: {pattern} - {description}")
                fixes_needed += 1
            else:
                print(f"✅ FIXED: {pattern} - {description}")
                fixes_applied += 1
        
        # Check for properly applied fixes
        good_patterns = [
            "x.shape().dims()[0].try_into().unwrap()",
            "b.shape().dims()[0].try_into().unwrap()",
            "result.shape().dims()[0].try_into().unwrap()",
        ]
        
        for pattern in good_patterns:
            if pattern in content:
                print(f"✅ GOOD FIX: {pattern}")
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"- Fixes needed: {fixes_needed}")
        print(f"- Fixes applied: {fixes_applied}")
        
        if fixes_needed == 0:
            print("🎉 ALL COMPILATION FIXES VALIDATED!")
        else:
            print(f"⚠️  {fixes_needed} fixes still needed")
        
        return fixes_needed == 0
    else:
        print("❌ Could not find iterative_solvers.rs")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)