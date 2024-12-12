import json
from typing import Any
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass
import logging

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
                df_info = f"DataFrame Info:\n"
                df_info += f"Shape: {item.shape}\n"
                df_info += f"Columns: {list(item.columns)}\n"
                df_info += f"Data Types:\n{item.dtypes}\n"
                df_info += f"\nFirst 5 rows:\n{item.head(5).to_string()}"
                snapshots.append(df_info)
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
        return content.head(5).to_string()
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

@dataclass
class DatasetDiff:
    added_rows: pd.DataFrame
    modified_rows: pd.DataFrame
    deleted_rows: pd.DataFrame
    context_rows: pd.DataFrame
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]

def compute_dataset_diff(old_df: pd.DataFrame, new_df: pd.DataFrame, 
                        context_radius: int = 2) -> DatasetDiff:
    """
    Compute comprehensive diff between old and new datasets with context.
    Handles dataframes with different shapes by comparing only common columns.
    """
    logging.info("Computing dataset diff between dataframes")
    logging.info(f"Old dataframe shape: {old_df.shape}, New dataframe shape: {new_df.shape}")

    # Ensure we only compare common columns
    common_columns = list(set(old_df.columns) & set(new_df.columns))
    logging.info(f"Found {len(common_columns)} common columns")

    if not common_columns:
        logging.warning("No common columns found between dataframes")
        # If no common columns, treat all rows as added/deleted
        return DatasetDiff(
            added_rows=new_df if len(new_df) > 0 else pd.DataFrame(),
            modified_rows=pd.DataFrame(),
            deleted_rows=old_df if len(old_df) > 0 else pd.DataFrame(),
            context_rows=pd.DataFrame(),
            statistics={
                'total_rows_old': len(old_df),
                'total_rows_new': len(new_df),
                'modified_rows_count': 0,
                'added_rows_count': len(new_df),
                'deleted_rows_count': len(old_df),
                'column_changes': {
                    'added_columns': list(set(new_df.columns) - set(old_df.columns)),
                    'deleted_columns': list(set(old_df.columns) - set(new_df.columns))
                }
            },
            metadata={
                'change_patterns': {
                    'is_schema_change': True,
                    'common_columns': [],
                    'added_columns': list(set(new_df.columns) - set(old_df.columns)),
                    'deleted_columns': list(set(old_df.columns) - set(new_df.columns))
                }
            }
        )

    # Find modified and new rows using index comparison on common columns
    common_indices = old_df.index.intersection(new_df.index)
    logging.info(f"Found {len(common_indices)} common indices")

    if len(common_indices) > 0:
        modified_mask = (old_df.loc[common_indices, common_columns] != 
                        new_df.loc[common_indices, common_columns]).any(axis=1)
        modified_indices = common_indices[modified_mask]
        logging.info(f"Found {len(modified_indices)} modified rows")
    else:
        modified_indices = pd.Index([])
        logging.info("No modified rows found")
    
    # Identify added and deleted rows
    added_indices = new_df.index.difference(old_df.index)
    deleted_indices = old_df.index.difference(new_df.index)
    logging.info(f"Found {len(added_indices)} added rows and {len(deleted_indices)} deleted rows")
    
    # Get context rows (surrounding rows for changes)
    all_affected_indices = set(modified_indices) | set(added_indices) | set(deleted_indices)
    context_indices = set()
    for idx in all_affected_indices:
        start = max(0, idx - context_radius)
        end = min(max(old_df.index.max(), new_df.index.max()), idx + context_radius + 1)
        # Only add indices that exist in new_df
        valid_indices = [i for i in range(start, end) if i in new_df.index]
        context_indices.update(valid_indices)
    logging.info(f"Generated {len(context_indices)} context rows")
    
    # Convert context_indices to a list and ensure all indices exist in new_df
    valid_context_indices = list(context_indices & set(new_df.index))
    
    # Compute relevant statistics
    statistics = {
        'total_rows_old': len(old_df),
        'total_rows_new': len(new_df),
        'modified_rows_count': len(modified_indices),
        'added_rows_count': len(added_indices),
        'deleted_rows_count': len(deleted_indices),
        'column_changes': {
            'added_columns': list(set(new_df.columns) - set(old_df.columns)),
            'deleted_columns': list(set(old_df.columns) - set(new_df.columns))
        },
        'column_statistics': {
            col: {
                'old_mean': old_df[col].mean() if col in old_df.columns and pd.api.types.is_numeric_dtype(old_df[col]) else None,
                'new_mean': new_df[col].mean() if col in new_df.columns and pd.api.types.is_numeric_dtype(new_df[col]) else None,
                'modified_columns': [col for col in common_columns if len(modified_indices) > 0 and 
                                   (old_df.loc[modified_indices, col] != new_df.loc[modified_indices, col]).any()]
            }
            for col in set(old_df.columns) | set(new_df.columns)
        }
    }
    logging.info("Computed dataset statistics")
    
    # Prepare metadata about changes
    metadata = {
        'change_locations': {
            'start_row': min(all_affected_indices) if all_affected_indices else None,
            'end_row': max(all_affected_indices) if all_affected_indices else None,
        },
        'change_patterns': {
            'is_append_only': len(added_indices) > 0 and len(modified_indices) == 0 and len(deleted_indices) == 0,
            'is_modify_only': len(added_indices) == 0 and len(modified_indices) > 0 and len(deleted_indices) == 0,
            'is_schema_change': len(set(new_df.columns) ^ set(old_df.columns)) > 0,
            'common_columns': common_columns,
            'affected_columns': list(set(
                col for idx in modified_indices
                for col in common_columns
                if old_df.loc[idx, col] != new_df.loc[idx, col]
            )) if len(modified_indices) > 0 else []
        }
    }
    logging.info("Generated change metadata")
    
    return DatasetDiff(
        added_rows=new_df.loc[added_indices] if len(added_indices) > 0 else pd.DataFrame(),
        modified_rows=new_df.loc[modified_indices] if len(modified_indices) > 0 else pd.DataFrame(),
        deleted_rows=old_df.loc[deleted_indices] if len(deleted_indices) > 0 else pd.DataFrame(),
        context_rows=new_df.loc[valid_context_indices] if valid_context_indices else pd.DataFrame(),
        statistics=statistics,
        metadata=metadata
    )

def prepare_analyzer_context(old_df: pd.DataFrame, new_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare optimized context for the analyzer LLM, focusing on essential differences.
    """
    logging.info("Preparing analyzer context")
    diff = compute_dataset_diff(old_df, new_df)
    logging.info("Computed dataset diff")

    # Get random sample of output dataframe with row numbers
    sample_rows = new_df.sample(n=min(10, len(new_df))).copy()
    sample_rows.index.name = 'row_number'
    sample_output = sample_rows.reset_index().to_dict(orient='records')

    # Only include non-empty changes
    changes = {}
    if not diff.added_rows.empty:
        changes['added_rows'] = diff.added_rows.head(3).to_dict(orient='records')
    
    if not diff.modified_rows.empty:
        changes['modified_rows'] = {
            'before': old_df.loc[diff.modified_rows.index[:3]].to_dict(orient='records'),
            'after': diff.modified_rows.head(3).to_dict(orient='records')
        }
    
    if not diff.deleted_rows.empty:
        changes['deleted_rows'] = diff.deleted_rows.head(3).to_dict(orient='records')

    context = {
        'output_sample': sample_output,
        'changes': {
            **changes,
            'schema_changes': {
                'added_columns': diff.statistics['column_changes']['added_columns'],
                'deleted_columns': diff.statistics['column_changes']['deleted_columns']
            }
        },
        'summary': {
            'total_changes': {
                'added rows': len(diff.added_rows),
                'modified rows': len(diff.modified_rows),
                'deleted rows': len(diff.deleted_rows)
            },
            'shapes': {
                'old': {'rows': len(old_df), 'columns': len(old_df.columns)},
                'new': {'rows': len(new_df), 'columns': len(new_df.columns)}
            },
            'is_schema_change': diff.metadata['change_patterns']['is_schema_change'],
            'is_append_only': diff.metadata['change_patterns']['is_append_only'],
            'is_modify_only': diff.metadata['change_patterns']['is_modify_only']
        }
    }
    
    return context