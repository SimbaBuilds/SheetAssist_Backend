import json
from typing import Any
import pandas as pd


def sanitize_error_message(e: Exception) -> str:
    """Sanitize error messages to remove binary data"""
    return str(e).encode('ascii', 'ignore').decode('ascii')


def get_data_snapshot(content: Any, data_type: str) -> str:
    """Generate appropriate snapshot based on data type"""
    # Handle tuple type
    if isinstance(content, tuple):
        # Process each item in tuple and join with newlines
        snapshots = []
        for item in content:
            if isinstance(item, pd.DataFrame):
                snapshots.append(item.head(10).to_string())
            elif isinstance(item, dict):
                snapshot_dict = dict(list(item.items())[:5])
                snapshots.append(json.dumps(snapshot_dict, indent=2))
            elif isinstance(item, list):
                snapshots.append(json.dumps(item[:5], indent=2))
            elif hasattr(item, 'file') and hasattr(item, 'filename'):
                # Handle image file objects
                item.file.seek(0)
                size = len(item.file.read())
                item.file.seek(0)
                snapshots.append(f"Image file: {item.filename}, Size: {size} bytes")
            else:
                snapshots.append(str(item)[:500])
        return "\n---\n".join(snapshots)

    # Handle non-tuple types
    if data_type == "DataFrame":
        return content.head(10).to_string()
    elif data_type == "json":
        # For JSON, return first few key-value pairs or array elements
        if isinstance(content, dict):
            snapshot_dict = dict(list(content.items())[:5])
            return json.dumps(snapshot_dict, indent=2)
        elif isinstance(content, list):
            return json.dumps(content[:5], indent=2)
        return str(content)[:500]
    elif data_type == "text":
        # Return first 500 characters for text
        return content[:500] + ("..." if len(content) > 500 else "")
    elif data_type == "image":
        content.file.seek(0)
        size = len(content.file.read())
        content.file.seek(0)
        return f"Image file: {content.filename}, Size: {size} bytes"
    return str(content)[:500]
