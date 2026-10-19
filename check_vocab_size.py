import sys
import os
import pickle

# Add current directory to path
sys.path.append(os.getcwd())

# Fix the Module Mapping for Pickle
import trafficMimic.dataset.vocab as vocab_module
sys.modules['dataset.vocab'] = vocab_module
sys.modules['dataset'] = vocab_module

files = ["timevocab_56.pkl", "sizevocab_1606.pkl"]

print("-" * 40)
for f_name in files:
    if os.path.exists(f_name):
        with open(f_name, "rb") as f:
            try:
                v = pickle.load(f)
                
                # Strategy 1: Standard length
                try:
                    size = len(v)
                except:
                    # Strategy 2: Check 'data' attribute (from your dir() list)
                    try:
                        size = len(v.data)
                    except:
                        # Strategy 3: Check internal dict via __dict__
                        # We look for any dictionary that isn't empty
                        dicts = [val for val in v.__dict__.values() if isinstance(val, dict)]
                        size = len(dicts[0]) if dicts else "Unknown"

                print(f"FILE: {f_name}")
                print(f">> ACTUAL SIZE TO USE: {size}")
                print("-" * 40)
            except Exception as e:
                print(f"Error reading {f_name}: {e}")
    else:
        print(f"File {f_name} not found.")