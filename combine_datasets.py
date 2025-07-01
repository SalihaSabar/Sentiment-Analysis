import polars as pl
import json
from pathlib import Path
from tqdm import tqdm
import gc

def load_imdb(file):
    print("\nProcessing IMDB dataset...")
    try:
        # Read CSV in chunks using Polars
        df = pl.read_csv(file)
        df = df.with_columns([
            pl.col('sentiment').map_elements(lambda x: 1 if x == 'Positive' else 0, return_dtype=pl.Int64).alias('label'),
            pl.col('review').alias('text')
        ])
        return df.select(['text', 'label'])
    except Exception as e:
        print(f"Error processing IMDB data: {e}")
        return pl.DataFrame()

def load_yelp(file):
    print("\nProcessing Yelp dataset...")
    texts = []
    labels = []
    chunk_size = 10000
    current_chunk = []
    
    try:
        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(file, 'r', encoding='utf-8'))
        
        with open(file, 'r', encoding='utf-8') as f:
            with tqdm(total=total_lines, desc="Reading Yelp reviews") as pbar:
                for line in f:
                    review = json.loads(line)
                    stars = review.get('stars', 0)
                    # Convert rating to binary sentiment
                    label = 1 if stars >= 4 else 0 if stars <= 2 else None
                    
                    if label is not None:
                        current_chunk.append({
                            'text': review.get('text', ''),
                            'label': label
                        })
                    
                    if len(current_chunk) >= chunk_size:
                        # Convert chunk to Polars DataFrame
                        chunk_df = pl.DataFrame(current_chunk)
                        texts.append(chunk_df)
                        current_chunk = []
                        # Force garbage collection
                        gc.collect()
                    
                    pbar.update(1)
                
                # Process remaining reviews
                if current_chunk:
                    chunk_df = pl.DataFrame(current_chunk)
                    texts.append(chunk_df)
        
        # Combine all chunks
        if texts:
            return pl.concat(texts)
        return pl.DataFrame()
    
    except Exception as e:
        print(f"Error processing Yelp data: {e}")
        return pl.DataFrame()

def load_twitter(file):
    print("\nProcessing Twitter dataset...")
    try:
        # Read CSV without headers using Polars
        df = pl.read_csv(file, has_header=False, encoding='latin-1')
        # Rename columns and map sentiment values
        df = df.with_columns([
            pl.col('column_1').map_elements(lambda x: 1 if x == 4 else 0, return_dtype=pl.Int64).alias('label'),
            pl.col('column_5').alias('text')
        ])
        # Filter out neutral sentiment (2) and select required columns
        return df.filter(pl.col('column_1') != 2).select(['text', 'label'])
    except Exception as e:
        print(f"Error processing Twitter data: {e}")
        return pl.DataFrame()

def load_reddit(file):
    print("\nProcessing Reddit dataset...")
    try:
        df = pl.read_csv(file)
        df = df.with_columns([
            pl.col('category').map_elements(lambda x: 0 if x == -1 else 1, return_dtype=pl.Int64).alias('label'),
            pl.col('clean_comment').alias('text')
        ])
        # Filter out neutral sentiment and select required columns
        return df.filter(pl.col('category') != 0).select(['text', 'label'])
    except Exception as e:
        print(f"Error processing Reddit data: {e}")
        return pl.DataFrame()

def load_file(args):
    source, file = args
    try:
        if source == 'imdb':
            return load_imdb(file)
        elif source == 'yelp':
            return load_yelp(file)
        elif source == 'twitter':
            return load_twitter(file)
        elif source == 'reddit':
            return load_reddit(file)
    except Exception as e:
        print(f"Error processing {source} data: {e}")
    return pl.DataFrame()

def process_datasets():
    output_dir = Path('processed_datasets')
    output_dir.mkdir(exist_ok=True)
    
    files = [
        ('imdb', 'raw_datasets/IMDB.csv'),
        ('yelp', 'raw_datasets/Yelp.json'),
        ('twitter', 'raw_datasets/Twitter.csv'),
        ('reddit', 'raw_datasets/Reddit.csv')
    ]
    
    dfs = []
    for file_info in tqdm(files, desc="Processing datasets"):
        df = load_file(file_info)
        if not df.is_empty():
            dfs.append(df)
            print(f"{file_info[0].upper()} dataset: {len(df):,} rows")
        # Force garbage collection after each file
        gc.collect()
    
    print("\nCombining datasets...")
    combined_df = pl.concat(dfs)
    
    # Clean up memory
    del dfs
    gc.collect()
    
    # Remove any rows with missing values
    combined_df = combined_df.drop_nulls()
    
    # Save the combined dataset
    output_file = output_dir / 'combined_dataset.csv'
    print(f"\nSaving combined dataset to {output_file}...")
    combined_df.write_csv(output_file)
    
    # Print statistics
    total_rows = len(combined_df)
    # Get sentiment counts using value_counts() instead
    sentiment_counts = combined_df.get_column('label').value_counts()
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {total_rows:,}")
    print("\nSentiment distribution:")
    for row in sentiment_counts.iter_rows():
        label, count = row
        sentiment = "Negative" if label == 0 else "Positive"
        print(f"{sentiment} ({label}): {count:,}")

if __name__ == '__main__':
    process_datasets()
