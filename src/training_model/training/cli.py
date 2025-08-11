"""
Command-line interface for the training module.
"""
import sys
import os

# Add the scripts directory to the path and import the main function
scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts')
sys.path.insert(0, scripts_dir)

from train import main

if __name__ == "__main__":
    main()