import pandas as pd
import json
from pyarrow import parquet

# Sample JSON data (you can replace this with your actual data)
json_data = [
    {"en": "Hello", "vi": "Xin chào"},
    {"en": "Thank you", "vi": "Cảm ơn bạn"},
    {"en": "How are you?", "vi": "Bạn khỏe không?"},
    {"en": "Good morning", "vi": "Chào buổi sáng"},
    {"en": "Goodbye", "vi": "Tạm biệt"}
]

# Function to create a parquet file from JSON data
def create_parquet_from_json(json_data, output_file="/tmp/language_pairs.parquet"):
    # Convert JSON to pandas DataFrame
    df = pd.DataFrame(json_data)
    
    # Write DataFrame to parquet file
    df.to_parquet(output_file, engine='pyarrow')
    
    print(f"Successfully created parquet file: {output_file}")
    return output_file

# Function to read a parquet file
def read_parquet_file(file_path="language_pairs.parquet"):
    # Read the parquet file
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    print(f"Successfully read parquet file: {file_path}")
    print("\nParquet file contents:")
    print(df)
    
    # Return the DataFrame for further processing if needed
    return df

# Example usage with file path
if __name__ == "__main__":
    # If your JSON data is in a file, you can load it like this:
    # with open('your_json_file.json', 'r', encoding='utf-8') as f:
    #     json_data = json.load(f)
    
    # Create the parquet file
    parquet_file = create_parquet_from_json(json_data)
    
    # Read the parquet file that was just created
    df = read_parquet_file(parquet_file)
    
    # Example: Access specific columns or rows
    print("\nEnglish phrases:")
    print(df['en'].tolist())
    
    print("\nVietnamese phrases:")
    print(df['vi'].tolist())
    
    # Example: Export to other formats if needed
    # df.to_csv("language_pairs.csv", index=False)
    # print("Exported to CSV as well")