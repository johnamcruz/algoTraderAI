#!/usr/bin/env python3
"""
Quick Model Output Checker

Run this to instantly check if your model has the auxiliary confidence head.
Usage: python check_model.py /path/to/model.onnx
"""

import sys
import os

def check_model(model_path):
    """Quick check of model outputs"""
    
    print("\n" + "=" * 70)
    print("🔍 QUICK MODEL CHECK")
    print("=" * 70)
    
    # Check file exists
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: File not found: {model_path}")
        return False
    
    print(f"\n📁 Model: {os.path.basename(model_path)}")
    print(f"   Path: {model_path}")
    print(f"   Size: {os.path.getsize(model_path):,} bytes")
    
    # Try to load
    try:
        import onnxruntime as ort
        model = ort.InferenceSession(model_path)
        print(f"✅ Model loaded successfully\n")
    except ImportError:
        print(f"\n❌ ERROR: onnxruntime not installed")
        print(f"   Install: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        return False
    
    # Check inputs
    print("📊 MODEL INPUTS:")
    for inp in model.get_inputs():
        print(f"   {inp.name}: {inp.shape} ({inp.type})")
    
    # Check outputs  
    print("\n📊 MODEL OUTPUTS:")
    num_outputs = len(model.get_outputs())
    
    for i, out in enumerate(model.get_outputs()):
        print(f"   Output {i+1}: {out.name}")
        print(f"      Shape: {out.shape}")
        print(f"      Type: {out.type}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("🎯 VERDICT")
    print("=" * 70)
    
    if num_outputs == 1:
        print("\n❌ PROBLEM FOUND!")
        print("   Model has only 1 output (no auxiliary confidence head)")
        print("\n💡 This model will NOT work with the validated strategy!")
        print("   You need the model from your training run that has 4 outputs:")
        print("   - Output 1: Logits (class predictions)")
        print("   - Output 2: Volatility (auxiliary)")
        print("   - Output 3: Direction (auxiliary)")
        print("   - Output 4: Confidence (auxiliary) ← MISSING!")
        print("\n📋 Action: Find the correct model file from training:")
        print("   Look for: checkpoint_best_stable.pth (E37)")
        print("   Or: multi_tf_transformer_v6_3_2_best.pth")
        return False
        
    elif num_outputs >= 4:
        print("\n✅ MODEL LOOKS GOOD!")
        print(f"   Model has {num_outputs} outputs (includes auxiliary heads)")
        print("\n📊 Expected outputs present:")
        print("   ✅ Main classification head")
        print("   ✅ Volatility prediction")
        print("   ✅ Direction prediction")
        print("   ✅ Confidence head")
        print("\n💡 If still getting only HOLD predictions, check:")
        print("   1. Strategy code is using auxiliary confidence (Output 4)")
        print("   2. Scaler matches this model")
        print("   3. Feature calculation is correct (23 features)")
        return True
        
    else:
        print(f"\n⚠️ WARNING: Unexpected number of outputs ({num_outputs})")
        print("   Expected: 1 (bad) or 4+ (good)")
        print("   Got: {num_outputs}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python check_model.py /path/to/model.onnx")
        print("\nExample:")
        print("  python check_model.py /content/drive/.../model.onnx")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = check_model(model_path)
    
    print("\n" + "=" * 70)
    sys.exit(0 if success else 1)
