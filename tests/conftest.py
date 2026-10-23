import sys
from pathlib import Path

# make tests/synthetic.py importable as `synthetic`
sys.path.insert(0, str(Path(__file__).parent))
