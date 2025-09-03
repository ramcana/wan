"""
Integration test for the code quality checking system.
"""

import tempfile
from pathlib import Path

import sys
from pathlib import Path

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_quality.quality_checker import QualityChecker
from code_quality.models import QualityConfig


def test_integration():
    """Test the complete code quality checking workflow."""
    
    # Create a temporary test file
    test_content = '''
"""Test module for code quality checking."""

import os
import sys
from typing import List, Dict, Optional


def calculate_average(numbers: List[float]) -> float:
    """
    Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numbers to average
        
    Returns:
        The average value
        
    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    
    return sum(numbers) / len(numbers)


class DataProcessor:
    """Process and analyze data with various methods."""
    
    def __init__(self, data: List[Dict[str, any]]) -> None:
        """
        Initialize the processor with data.
        
        Args:
            data: List of data dictionaries to process
        """
        self.data = data
        self.processed_count = 0
    
    def process_item(self, item: Dict[str, any]) -> Optional[Dict[str, any]]:
        """
        Process a single data item.
        
        Args:
            item: Data item to process
            
        Returns:
            Processed item or None if invalid
        """
        if not item or 'id' not in item:
            return None
        
        processed = {
            'id': item['id'],
            'processed': True,
            'timestamp': item.get('timestamp', 'unknown')
        }
        
        self.processed_count += 1
        return processed
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            'total_items': len(self.data),
            'processed_items': self.processed_count,
            'remaining_items': len(self.data) - self.processed_count
        }


def main():
    """Main function to demonstrate usage."""
    # Sample data
    sample_data = [
        {'id': 1, 'name': 'Item 1', 'timestamp': '2023-01-01'},
        {'id': 2, 'name': 'Item 2', 'timestamp': '2023-01-02'},
        {'id': 3, 'name': 'Item 3'}
    ]
    
    # Process data
    processor = DataProcessor(sample_data)
    
    for item in sample_data:
        result = processor.process_item(item)
        if result:
            print(f"Processed item {result['id']}")
    
    # Show statistics
    stats = processor.get_statistics()
    print(f"Statistics: {stats}")
    
    # Calculate some averages
    numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
    avg = calculate_average(numbers)
    print(f"Average: {avg}")


if __name__ == "__main__":
    main()
'''
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_content)
        temp_file = Path(f.name)
    
    try:
        # Test with default configuration
        print("Testing with default configuration...")
        checker = QualityChecker()
        report = checker.check_quality(temp_file)
        
        print(f"Files analyzed: {report.files_analyzed}")
        print(f"Total issues: {report.total_issues}")
        print(f"Quality score: {report.quality_score:.1f}/100")
        print(f"Documentation coverage: {report.metrics.documentation_coverage:.1f}%")
        print(f"Type hint coverage: {report.metrics.type_hint_coverage:.1f}%")
        
        # Test specific checks
        print("\nTesting specific checks...")
        doc_report = checker.check_quality(temp_file, checks=['documentation'])
        print(f"Documentation issues: {len(doc_report.issues)}")
        
        type_report = checker.check_quality(temp_file, checks=['type_hints'])
        print(f"Type hint issues: {len(type_report.issues)}")
        
        complexity_report = checker.check_quality(temp_file, checks=['complexity'])
        print(f"Complexity issues: {len(complexity_report.issues)}")
        
        # Test report generation
        print("\nTesting report generation...")
        json_report = checker.generate_report(report, 'json')
        print(f"JSON report length: {len(json_report)} characters")
        
        text_report = checker.generate_report(report, 'text')
        print(f"Text report length: {len(text_report)} characters")
        
        # Test with custom configuration
        print("\nTesting with custom configuration...")
        config = QualityConfig(
            max_line_length=120,
            require_function_docstrings=True,
            max_cyclomatic_complexity=15
        )
        custom_checker = QualityChecker(config)
        custom_report = custom_checker.check_quality(temp_file)
        print(f"Custom config quality score: {custom_report.quality_score:.1f}/100")
        
        print("\n✅ Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
traceback.print_exc()
        return False
        
    finally:
        # Clean up
        temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    test_integration()