import json
from typing import Any
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass
import logging
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv(override=True)
MAX_SNAPSHOT_LENGTH = int(os.getenv("MAX_SNAPSHOT_LENGTH"))

def _process_dataframe(df: pd.DataFrame) -> str:
    """Helper function to process DataFrame and generate info string"""
    df = df.replace({pd.NaT: None})
    df_info = f"DataFrame Info:\n"
    df_info += f"Shape: {df.shape}\n"
    df_info += f"Columns: {list(df.columns)}\n"
    df_info += f"\nFirst 5 rows:\n{df.head(5).to_string()}"
    return df_info

def _process_collection(data: Any, limit: int = 5) -> str:
    """Helper function to process dictionaries and lists"""
    if isinstance(data, dict):
        snapshot_dict = {k: (None if isinstance(v, (pd.Series, pd.DataFrame)) else (None if pd.isna(v) else v)) 
                        for k, v in list(data.items())[:limit]}
        return json.dumps(snapshot_dict, indent=2)
    elif isinstance(data, list):
        clean_list = [None if isinstance(x, (pd.Series, pd.DataFrame)) else (None if pd.isna(x) else x) 
                     for x in data[:limit]]
        return json.dumps(clean_list, indent=2)
    return str(None if pd.isna(data) else data)[:MAX_SNAPSHOT_LENGTH]

def get_data_snapshot(content: Any, data_type: str, is_image_like_pdf: bool = False) -> str:
    """Generate appropriate snapshot based on data type"""
    if is_image_like_pdf:
        return content

    if isinstance(content, tuple):
        snapshots = []
        for item in content:
            if isinstance(item, pd.DataFrame):
                snapshots.append(_process_dataframe(item))
            elif isinstance(item, pd.Series):
                snapshots.append(str(item))
            elif isinstance(item, (dict, list)):
                snapshots.append(_process_collection(item))
            elif hasattr(item, 'file') and hasattr(item, 'filename'):
                item.file.seek(0)
                size = len(item.file.read())
                item.file.seek(0)
                snapshots.append(f"Image file: {item.filename}, Size: {size} bytes")
            else:
                snapshots.append(_process_collection(item))
        return "\n---\n".join(snapshots)

    if data_type == "DataFrame":
        if isinstance(content, pd.DataFrame):
            return _process_dataframe(content)
        elif isinstance(content, pd.Series):
            return str(content)
        return str(content)
    elif data_type == "json":
        return _process_collection(content)
    elif data_type == "text":
        text = str(None if pd.isna(content) else content)
        return text[:MAX_SNAPSHOT_LENGTH] + ("..." if len(text) > MAX_SNAPSHOT_LENGTH else "")
    elif data_type == "image":
        content.file.seek(0)
        size = len(content.file.read())
        content.file.seek(0)
        return f"Image file: {content.filename}, Size: {size} bytes"
    
    return str(None if pd.isna(content) else content)[:MAX_SNAPSHOT_LENGTH]

def process_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Process a DataFrame to make it JSON serializable by handling datetime, object types, and NA values."""
    processed_df = df.copy()
    
    # Handle datetime columns
    for col in processed_df.select_dtypes(include=['datetime64[ns]']).columns:
        processed_df[col] = processed_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Handle object columns with numpy types
    for col in processed_df.columns:
        col_dtype = processed_df[col].dtype
        if col_dtype == object or str(col_dtype) == 'object':
            processed_df[col] = processed_df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    
    # Replace NaN, NaT, etc with None
    processed_df = processed_df.replace({pd.NaT: None, pd.NA: None, np.nan: None})
    
    return processed_df

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
    logging.info(f"Starting dataset diff computation. Old shape: {old_df.shape}, New shape: {new_df.shape}")
    
    # Process both dataframes to ensure JSON serializable and consistent handling
    old_df = process_dataframe_for_json(old_df.copy())
    new_df = process_dataframe_for_json(new_df.copy())
    
    # Handle NaN indices by filling them with a valid integer
    if old_df.index.isna().any() or new_df.index.isna().any():
        logging.info("Handling NaN indices by resetting index")
        if old_df.index.isna().any():
            old_df = old_df.reset_index(drop=True)
        if new_df.index.isna().any():
            new_df = new_df.reset_index(drop=True)
    
    # Now safely convert to int
    old_df.index = old_df.index.astype(int)
    new_df.index = new_df.index.astype(int)
    
    # Find common columns
    common_columns = list(set(old_df.columns) & set(new_df.columns))
    logging.info(f"Found {len(common_columns)} common columns")

    if not common_columns:
        logging.warning("No common columns found between dataframes")
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
    logging.info(f"Found {len(common_indices)} common indices between dataframes")

    if len(common_indices) > 0:
        old_subset = old_df.loc[common_indices, common_columns]
        new_subset = new_df.loc[common_indices, common_columns]
        
        # Compare dataframes element by element, handling NA values properly
        if len(common_indices) == 1:
            logging.info("Processing single row comparison")
            # Compare all values in the row and create a boolean Series
            comparison = (old_subset.fillna(pd.NA) == new_subset.fillna(pd.NA))
            # Check if any column is different
            has_differences = not comparison.all().all()  # .all() for both row and columns
            modified_indices = common_indices if has_differences else pd.Index([])
            logging.info(f"Single row comparison result: has_differences={has_differences}")
        else:
            logging.info("Processing multiple rows comparison")
            are_equal = (old_subset.fillna(pd.NA) == new_subset.fillna(pd.NA)).all(axis=1)
            modified_indices = common_indices[~are_equal]
    else:
        modified_indices = pd.Index([])
    
    # Identify added and deleted rows
    added_indices = new_df.index.difference(old_df.index)
    deleted_indices = old_df.index.difference(new_df.index)
    
    logging.info(f"Found {len(modified_indices)} modified rows, {len(added_indices)} added rows, "
                f"and {len(deleted_indices)} deleted rows")

    # Get context rows
    all_affected_indices = set(modified_indices) | set(added_indices) | set(deleted_indices)
    context_indices = set()
    for idx in all_affected_indices:
        # Convert index to int and handle NaN
        try:
            idx_int = int(idx)
            start = max(0, idx_int - context_radius)
            end = min(
                int(max(old_df.index.max() or 0, new_df.index.max() or 0)), 
                idx_int + context_radius + 1
            )
            valid_indices = [i for i in range(start, end) if i in new_df.index]
            context_indices.update(valid_indices)
        except (ValueError, TypeError):
            continue  # Skip indices that can't be converted to int
    
    valid_context_indices = list(context_indices & set(new_df.index))
    logging.info(f"Generated {len(valid_context_indices)} context rows")

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
                                   not (old_df.loc[modified_indices, col].isna() & new_df.loc[modified_indices, col].isna()).all() and
                                   (old_df.loc[modified_indices, col] != new_df.loc[modified_indices, col]).any()]
            }
            for col in set(old_df.columns) | set(new_df.columns)
        }
    }
    
    # Prepare metadata about changes
    metadata = {
        'change_locations': {
            'start_row': min(all_affected_indices) if all_affected_indices else None,
            'end_row': max(all_affected_indices) if all_affected_indices else None,
        },
        'change_patterns': {
            'is_schema_change': bool(len(set(new_df.columns) ^ set(old_df.columns)) > 0),
            'common_columns': common_columns,
            'affected_columns': list(set(
                col for idx in modified_indices
                for col in common_columns
                if old_df.loc[idx, col] != new_df.loc[idx, col]
            )) if len(modified_indices) > 0 else []
        }
    }
    
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
    
    def convert_timestamps(df):
        """Convert timestamps to string format"""
        df = df.copy()
        # Handle NaN values before conversion
        df = df.fillna(pd.NA)  # Convert NaN to pandas NA type
        for col in df.select_dtypes(include=['datetime64[ns]']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        return df

    # Create copies and replace NaT values with None before any processing
    old_df = convert_timestamps(old_df.copy().replace({pd.NaT: None}))
    new_df = convert_timestamps(new_df.copy().replace({pd.NaT: None}))
    
    
    def convert_dict_timestamps(d):
        """Convert any timestamp values in a dictionary to strings"""
        for k, v in d.items():
            if isinstance(v, pd.Timestamp):
                d[k] = v.strftime('%Y-%m-%d %H:%M:%S')
        return d
    
    # Create copies and replace NaT values with None before any processing
    old_df = convert_timestamps(old_df.copy().replace({pd.NaT: None}))
    new_df = convert_timestamps(new_df.copy().replace({pd.NaT: None}))
    
    diff = compute_dataset_diff(old_df, new_df)
    logging.info("Computed dataset diff")

    # Replace NaT values with None in both dataframes
    old_df = old_df.replace({pd.NaT: None})
    new_df = new_df.replace({pd.NaT: None})

    # Get random sample of output dataframe with row numbers
    sample_rows = new_df.sample(n=min(10, len(new_df))).copy()
    sample_rows.index.name = 'row_number'
    sample_output = [convert_dict_timestamps(row) for row in sample_rows.reset_index().to_dict(orient='records')]
    
    # Only include non-empty changes
    changes = {}
    if not diff.added_rows.empty:
        changes['added_rows'] = [convert_dict_timestamps(row) for row in diff.added_rows.head(3).to_dict(orient='records')]
    
    if not diff.modified_rows.empty:
        changes['modified_rows'] = {
            'before': [convert_dict_timestamps(row) for row in old_df.loc[diff.modified_rows.index[:3]].to_dict(orient='records')],
            'after': [convert_dict_timestamps(row) for row in diff.modified_rows.head(3).to_dict(orient='records')]
        }
    
    if not diff.deleted_rows.empty:
        changes['deleted_rows'] = [convert_dict_timestamps(row) for row in diff.deleted_rows.head(3).to_dict(orient='records')]
    
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
            'is_schema_change': diff.metadata['change_patterns']['is_schema_change']
        }
    }
    
    return context