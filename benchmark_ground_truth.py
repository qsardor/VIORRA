import os
import sys
from datasets import load_dataset

try:
    import viorra.engine
except ImportError:
    print("Warning: 'viorra' module not found. The script will mock the engine for testing purposes.")
    class MockEngine:
        @staticmethod
        def analyze_essay(text):
            return "This is a mock Viorra critique of the essay."
    
    class MockViorra:
        engine = MockEngine()
    
    viorra = MockViorra()

def main():
    print("Loading dataset 'qsardor/viorra-admissions-essays'...")
    try:
        ds = load_dataset('qsardor/viorra-admissions-essays', split='train')
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset loaded. Total examples: {len(ds)}")
    if len(ds) > 0:
        print(f"Columns: {ds.column_names}")

    # Determine columns based on typical names
    text_col = 'essay' if 'essay' in ds.column_names else ds.column_names[0]
    gt_col = 'evaluation' if 'evaluation' in ds.column_names else ('critique' if 'critique' in ds.column_names else ds.column_names[1])

    results = []
    
    # Take 5-10 essays
    subset = ds.select(range(min(10, len(ds))))
    
    with open('benchmark_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== Viorra Ground Truth Evaluation ===\n\n")
        
        for i, example in enumerate(subset):
            text = example[text_col]
            gt_critique = example[gt_col]
            
            # Check length (150-1000 words)
            word_count = len(str(text).split())
            if not (150 <= word_count <= 1000):
                f.write(f"--- Example {i+1} Skipped (Word count {word_count} out of range) ---\n\n")
                continue
                
            f.write(f"--- Example {i+1} ---\n")
            f.write(f"Word Count: {word_count}\n")
            
            # Run viorra engine
            try:
                viorra_critique = viorra.engine.analyze_essay(text)
            except Exception as e:
                viorra_critique = f"Error running Viorra: {e}"
                
            f.write("\n[Ground Truth Critique]\n")
            f.write(str(gt_critique) + "\n")
            
            f.write("\n[Viorra Engine Critique]\n")
            f.write(str(viorra_critique) + "\n")
            
            f.write("\n" + "="*50 + "\n\n")
            print(f"Processed example {i+1}")
            
    print("Benchmarking complete. Results saved to benchmark_results.txt.")

if __name__ == '__main__':
    main()
