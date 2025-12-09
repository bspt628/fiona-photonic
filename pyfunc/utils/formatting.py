"""
Formatting utilities for FIONA photonic models.

Provides pretty-printing functions for matrices and arrays
to improve readability of debug output.
"""

import numpy as np


def format_matrix(name, data, max_rows=8, max_cols=8):
    """
    Format a matrix or nested list for readable output.

    Args:
        name: Variable name to display
        data: Matrix data (list, nested list, or numpy array)
        max_rows: Maximum rows to display before truncating
        max_cols: Maximum columns to display before truncating

    Returns:
        Formatted string representation
    """
    lines = [f'[Python] {name}:']

    # Convert to numpy array if possible
    try:
        arr = np.array(data)
        if arr.dtype == object:
            # Nested list with varying lengths - fall back to simple format
            lines.append(f'  Shape: (nested list, {len(data)} elements)')
            for i, row in enumerate(data[:max_rows]):
                if isinstance(row, (list, tuple)):
                    row_str = ', '.join(str(x) for x in row[:max_cols])
                    if len(row) > max_cols:
                        row_str += ', ...'
                    lines.append(f'  [{i}]: [{row_str}]')
                else:
                    lines.append(f'  [{i}]: {row}')
            if len(data) > max_rows:
                lines.append(f'  ... ({len(data) - max_rows} more rows)')
        else:
            # Regular numpy array
            lines.append(f'  Shape: {arr.shape}, dtype: {arr.dtype}')

            if arr.ndim == 1:
                # 1D array
                if len(arr) <= max_cols:
                    lines.append(f'  {arr}')
                else:
                    lines.append(f'  [{arr[:max_cols//2]}  ...  {arr[-max_cols//2:]}]')

            elif arr.ndim == 2:
                rows, cols = arr.shape
                # Check if this is raw byte-encoded 1D data (last dim is bytes, typically 2-4)
                if cols <= 4 and rows > cols:
                    # Likely 1D vector with bytes: [N][bytes] -> show as 1D
                    lines[-1] = f'  Shape: ({rows},) [raw: {arr.shape}], dtype: {arr.dtype}'
                    lines.append(f'  (1D vector, {rows} elements, {cols} bytes each)')
                    display_rows = min(rows, max_rows)
                    # Show first byte of each element as preview
                    preview = [int(arr[i, 0]) for i in range(display_rows)]
                    lines.append(f'  First bytes: {preview}{"..." if rows > max_rows else ""}')
                else:
                    # Regular 2D array - format as matrix
                    display_rows = min(rows, max_rows)
                    display_cols = min(cols, max_cols)
                    col_width = max(8, max(len(f'{x:.4g}') for x in arr[:display_rows, :display_cols].flatten()))

                    for i in range(display_rows):
                        row_vals = [f'{arr[i, j]:>{col_width}.4g}' for j in range(display_cols)]
                        row_str = ' '.join(row_vals)
                        if cols > max_cols:
                            row_str += '  ...'
                        lines.append(f'  [{row_str}]')

                    if rows > max_rows:
                        lines.append(f'  ... ({rows - max_rows} more rows)')

            elif arr.ndim == 3:
                d1, d2, d3 = arr.shape
                # Check if this is raw byte-encoded 2D data (last dim is bytes, typically 2-4)
                if d3 <= 4:
                    # Likely 2D matrix with bytes: [M][N][bytes] -> show as 2D
                    lines[-1] = f'  Shape: ({d1}, {d2}) [raw: {arr.shape}], dtype: {arr.dtype}'
                    lines.append(f'  (2D matrix, {d1}x{d2} elements, {d3} bytes each)')
                    display_rows = min(d1, max_rows)
                    display_cols = min(d2, max_cols)

                    # Show first byte of each element as preview matrix
                    for i in range(display_rows):
                        row_vals = [f'{int(arr[i, j, 0]):>6}' for j in range(display_cols)]
                        row_str = ' '.join(row_vals)
                        if d2 > max_cols:
                            row_str += '  ...'
                        lines.append(f'  [{row_str}]')

                    if d1 > max_rows:
                        lines.append(f'  ... ({d1 - max_rows} more rows)')
                else:
                    # Generic 3D array
                    with np.printoptions(threshold=50, edgeitems=3, linewidth=100):
                        lines.append(f'  {arr}')
            else:
                # Higher dimensional - use numpy's default
                with np.printoptions(threshold=50, edgeitems=3, linewidth=100):
                    lines.append(f'  {arr}')
    except (ValueError, TypeError):
        # Fallback for non-convertible data
        lines.append(f'  {str(data)[:200]}{"..." if len(str(data)) > 200 else ""}')

    return '\n'.join(lines)
