#!/usr/bin/env python
"""
Main entry point for Synthetic French Exam Generator.
Execute from root directory: python main.py
"""

import sys
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
env_file = Path(__file__).parent / ".env"
load_dotenv(env_file)

# Add root directory to Python path so imports work correctly
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

# Now import the generator
from service.generator import SyntheticTrainingGenerator

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Synthetic French Exam Generator for Training"
    )
    parser.add_argument(
        "--num-exams",
        type=int,
        default=None,
        help="Number of exams to generate (default: config value)",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Run in infinite loop mode",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Override trainer API URL",
    )
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.api_url:
        from config import config
        config.TRAINER_API_URL = args.api_url
    
    # Create and run generator
    generator = SyntheticTrainingGenerator()
    generator.run(num_exams=args.num_exams, infinite=args.infinite)

if __name__ == "__main__":
    main()
