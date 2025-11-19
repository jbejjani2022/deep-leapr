#!/usr/bin/env python3
"""
Test script to verify expert API libraries are available in multiprocessing contexts.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import sys

def test_imports_in_child_process():
    """Test that all expert libraries can be imported in a child process."""
    try:
        # Import the module that should have expert libraries loaded
        from domain.text_regression import (
            TEXTSTAT_AVAILABLE, 
            SPACY_AVAILABLE, 
            NLTK_SENTIMENT_AVAILABLE,
            TEXTBLOB_AVAILABLE,
            textstat,
            spacy,
            SentimentIntensityAnalyzer,
            TextBlob
        )
        
        results = {
            "textstat": TEXTSTAT_AVAILABLE and textstat is not None,
            "spacy": SPACY_AVAILABLE and spacy is not None,
            "nltk": NLTK_SENTIMENT_AVAILABLE and SentimentIntensityAnalyzer is not None,
            "textblob": TEXTBLOB_AVAILABLE and TextBlob is not None,
        }
        
        # Try actually using them
        test_text = "This is a simple test sentence."
        
        if results["textstat"]:
            try:
                score = textstat.flesch_reading_ease(test_text)
                results["textstat_works"] = True
            except Exception as e:
                results["textstat_error"] = str(e)
                results["textstat_works"] = False
        
        return results
    except Exception as e:
        return {"error": str(e), "traceback": __import__('traceback').format_exc()}


def test_feature_execution():
    """Test that a feature using expert libraries can execute in a child process."""
    from domain.text_regression import TextRegression
    from feature_engine import Feature
    
    domain = TextRegression(api_level="expert")
    
    # Test feature using textstat
    feature_code = '''def feature(text: str) -> float:
    "Flesch reading ease score"
    return float(textstat.flesch_reading_ease(text))
'''
    
    feature = Feature(feature_code, domain)
    test_text = "This is a simple test sentence with easy words."
    
    try:
        result = feature.execute(test_text)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": __import__('traceback').format_exc()}


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Expert API Libraries")
    print("=" * 60)
    
    # Test 1: Direct import in main process
    print("\n1. Testing imports in main process...")
    from domain.text_regression import (
        TEXTSTAT_AVAILABLE, 
        SPACY_AVAILABLE, 
        NLTK_SENTIMENT_AVAILABLE,
        TEXTBLOB_AVAILABLE
    )
    
    print(f"   textstat: {'✓' if TEXTSTAT_AVAILABLE else '✗'}")
    print(f"   spacy: {'✓' if SPACY_AVAILABLE else '✗'}")
    print(f"   nltk sentiment: {'✓' if NLTK_SENTIMENT_AVAILABLE else '✗'}")
    print(f"   textblob: {'✓' if TEXTBLOB_AVAILABLE else '✗'}")
    
    # Test 2: Import in child process
    print("\n2. Testing imports in child process (spawn context)...")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        future = ex.submit(test_imports_in_child_process)
        result = future.result(timeout=30)
        
        if "error" in result:
            print(f"   ✗ Error in child process:")
            print(f"   {result['error']}")
            if "traceback" in result:
                print(result["traceback"])
        else:
            for lib, available in result.items():
                if not lib.endswith("_works") and not lib.endswith("_error"):
                    print(f"   {lib}: {'✓' if available else '✗'}")
            
            if "textstat_works" in result:
                print(f"   textstat.flesch_reading_ease(): {'✓' if result['textstat_works'] else '✗'}")
                if "textstat_error" in result:
                    print(f"     Error: {result['textstat_error']}")
    
    # Test 3: Feature execution
    print("\n3. Testing feature execution in main process...")
    result = test_feature_execution()
    if result["success"]:
        print(f"   ✓ Feature executed successfully: {result['result']}")
    else:
        print(f"   ✗ Feature execution failed:")
        print(f"   {result['error']}")
        if "traceback" in result:
            print(result["traceback"])
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

