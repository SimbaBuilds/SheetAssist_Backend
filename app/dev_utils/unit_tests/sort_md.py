from pydantic import BaseModel


class FileMetadata(BaseModel):
    """Metadata about an uploaded file from frontend"""
    name: str
    type: str  # MIME type
    extension: str
    size: int
    index: int


# Test data
files_metadata = [
    FileMetadata(name="file1.xlsx", type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", extension="xlsx", size=100, index=0),
    FileMetadata(name="file2.csv", type="text/csv", extension="csv", size=200, index=1),
    FileMetadata(name="file3.txt", type="text/plain", extension="txt", size=300, index=2),
    FileMetadata(name="file4.pdf", type="application/pdf", extension="pdf", size=400, index=3)
]


def test_file_metadata_sorting():
    # Create test cases with different orderings
    test_cases = [
        # Test case 1: Original order with mixed types
        files_metadata,
        
        # Test case 2: Reverse order
        files_metadata[::-1],
        
        # Test case 3: CSV first, then non-priority files
        [
            FileMetadata(name="csv_file.csv", type="text/csv", extension="csv", size=200, index=0),
            FileMetadata(name="text_file.txt", type="text/plain", extension="txt", size=300, index=1),
            FileMetadata(name="pdf_file.pdf", type="application/pdf", extension="pdf", size=400, index=2),
        ]
    ]
    
    for i, test_data in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print("Input order:")
        for file in test_data:
            print(f"  {file.name} (Index: {file.index})")
            
        priority_types = {
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/csv'
        }
        
        sorted_data = sorted(
            test_data,
            key=lambda x: (x.type not in priority_types, x.index)
        )
        
        print("\nSorted order:")
        for file in sorted_data:
            print(f"  {file.name} (Index: {file.index})")
        
        # Verify that priority files come first
        for idx, file in enumerate(sorted_data):
            if file.type in priority_types:
                assert idx < len([f for f in sorted_data if f.type not in priority_types]), \
                    f"Priority file {file.name} should come before non-priority files"


if __name__ == "__main__":
    print("Running file metadata sorting tests...")
    test_file_metadata_sorting()
    print("\nAll tests passed successfully!")