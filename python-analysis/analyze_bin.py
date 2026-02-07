#!/usr/bin/env python3
"""
ArduPilot .bin Log Analyzer
Analyzes a DataFlash .bin file and prints all available message types and their plottable fields.

Usage:
    python analyze_bin.py <path_to_bin_file>
    
    Or drag and drop a .bin file onto this script.
"""

import sys
import os

try:
    from pymavlink import mavutil
except ImportError:
    print("Error: pymavlink is not installed.")
    print("Install it with: pip install pymavlink")
    sys.exit(1)


def analyze_bin_file(filepath):
    """Analyze a .bin file and print all available message types and fields."""
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(filepath)}")
    print(f"{'='*60}\n")
    
    # Open the log file
    try:
        mlog = mavutil.mavlink_connection(filepath)
    except Exception as e:
        print(f"Error opening file: {e}")
        return
    
    # Dictionary to store message types and their fields
    message_types = {}
    message_counts = {}
    
    print("Scanning log file for message types...")
    print("(This may take a moment for large files)\n")
    
    # Read through the entire log to collect all message types
    while True:
        try:
            msg = mlog.recv_match()
            if msg is None:
                break
            
            msg_type = msg.get_type()
            
            # Skip certain internal message types
            if msg_type in ['FMT', 'FMTU', 'MULT', 'MODE', 'UNIT', 'PARM']:
                if msg_type == 'PARM':
                    # Still count PARM but don't add to plottable
                    message_counts[msg_type] = message_counts.get(msg_type, 0) + 1
                continue
            
            # Count messages
            message_counts[msg_type] = message_counts.get(msg_type, 0) + 1
            
            # Get field names if we haven't seen this type before
            if msg_type not in message_types:
                # Get the message as a dictionary to extract field names
                msg_dict = msg.to_dict()
                # Remove 'mavpackettype' as it's metadata, not a plottable field
                fields = [k for k in msg_dict.keys() if k != 'mavpackettype']
                message_types[msg_type] = fields
                
        except Exception as e:
            # Skip problematic messages
            continue
    
    # Print summary
    total_types = len(message_types)
    total_messages = sum(message_counts.values())
    
    print(f"Found {total_types} plottable message types")
    print(f"Total messages in log: {total_messages:,}\n")
    
    print(f"{'='*60}")
    print("PLOTTABLE PARAMETERS BY MESSAGE TYPE")
    print(f"{'='*60}\n")
    
    # Sort message types alphabetically
    for msg_type in sorted(message_types.keys()):
        fields = message_types[msg_type]
        count = message_counts.get(msg_type, 0)
        
        print(f"📊 {msg_type} ({count:,} samples)")
        print(f"   Fields: {', '.join(fields)}")
        print()
    
    # Print a quick reference list
    print(f"\n{'='*60}")
    print("QUICK REFERENCE - All Plottable Fields")
    print(f"{'='*60}\n")
    
    all_fields = []
    for msg_type in sorted(message_types.keys()):
        for field in message_types[msg_type]:
            all_fields.append(f"{msg_type}.{field}")
    
    # Print in columns
    for field in sorted(all_fields):
        print(f"  {field}")
    
    print(f"\n{'='*60}")
    print(f"Total plottable fields: {len(all_fields)}")
    print(f"{'='*60}\n")
    
    return message_types


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        
        # Interactive mode - ask for file path
        print("\nNo file specified. Enter the path to your .bin file:")
        filepath = input("> ").strip().strip('"').strip("'")
        
        if not filepath:
            print("No file path provided. Exiting.")
            sys.exit(1)
    else:
        filepath = sys.argv[1]
    
    # Handle drag-and-drop (Windows adds quotes sometimes)
    filepath = filepath.strip().strip('"').strip("'")
    
    analyze_bin_file(filepath)
    
    # Keep console open if run by double-clicking
    if len(sys.argv) < 2:
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
