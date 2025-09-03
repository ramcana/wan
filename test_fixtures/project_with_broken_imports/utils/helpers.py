"""Helper utilities for the sample project"""

def clean_string(text: str) -> str:
    """Clean and normalize a string"""
    return text.strip().lower()

def process_data(data):
    """Process some data"""
    return [clean_string(item) for item in data]