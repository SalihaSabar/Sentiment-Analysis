import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from wordcloud import WordCloud
import gc

# Set style for better visualizations
plt.style.use('ggplot')
sns.set_palette("husl")

# Create directory for plots
os.makedirs("eda_plots", exist_ok=True)

# Function to process data in chunks
def process_in_chunks(df, chunk_size=100000):
    word_freq = Counter()
    total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    
    for i in range(0, len(df), chunk_size):
        chunk = df[i:i + chunk_size]
        # Process word frequencies for this chunk
        text = ' '.join(chunk['preprocessed_text'].to_list())
        words = text.split()
        word_freq.update(words)
        
        # Free up memory
        del text, words, chunk
        gc.collect()
        
        # Progress indicator
        current_chunk = i // chunk_size + 1
        print(f"Processing chunk {current_chunk}/{total_chunks}")
    
    return word_freq

# Load and process the dataset with basic statistics
print("Loading dataset and computing basic statistics...")
df = pl.read_csv('processed_datasets/combined_dataset_preprocessed.csv')

# Show basic info
print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns)
print("\nFirst 5 rows:")
print(df.head())

# Text length analysis
print("\nComputing text length statistics...")
df = df.with_columns([
    pl.col('text').str.split(' ').list.len().alias('original_word_count'),
    pl.col('text').str.len_chars().alias('original_char_count'),
    pl.col('preprocessed_text').str.split(' ').list.len().alias('processed_word_count'),
    pl.col('preprocessed_text').str.len_chars().alias('processed_char_count')
])

# Print text statistics
print("\nOriginal text statistics:")
print(df.select(['original_word_count', 'original_char_count']).describe())
print("\nPreprocessed text statistics:")
print(df.select(['processed_word_count', 'processed_char_count']).describe())

# Sample a subset for visualizations
print("\nCreating visualizations with sampled data...")
sample_size = 100000
df_sample = df.sample(n=sample_size, seed=42)

# Word count distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Original text word count
sns.histplot(data=df_sample.to_pandas(), x='original_word_count', bins=30, kde=True, ax=ax1)
ax1.set_title('Distribution of Word Counts (Original Text)\nBased on 100k sample')
ax1.set_xlabel('Number of Words')
ax1.set_ylabel('Frequency')

# Preprocessed text word count
sns.histplot(data=df_sample.to_pandas(), x='processed_word_count', bins=30, kde=True, ax=ax2)
ax2.set_title('Distribution of Word Counts (Preprocessed Text)\nBased on 100k sample')
ax2.set_xlabel('Number of Words')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("eda_plots/word_count_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# Character count distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Original text character count
sns.histplot(data=df_sample.to_pandas(), x='original_char_count', bins=30, kde=True, ax=ax1)
ax1.set_title('Distribution of Character Counts (Original Text)\nBased on 100k sample')
ax1.set_xlabel('Number of Characters')
ax1.set_ylabel('Frequency')

# Preprocessed text character count
sns.histplot(data=df_sample.to_pandas(), x='processed_char_count', bins=30, kde=True, ax=ax2)
ax2.set_title('Distribution of Character Counts (Preprocessed Text)\nBased on 100k sample')
ax2.set_xlabel('Number of Characters')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("eda_plots/char_count_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# Process word frequencies in chunks
print("\nProcessing word frequencies in chunks...")
word_freq = process_in_chunks(df)

# Get top words
print("\nGenerating word frequency plots...")
top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
words = list(top_words.keys())
counts = list(top_words.values())

plt.figure(figsize=(12, 6))
bars = sns.barplot(x=words, y=counts)
plt.title('Top 20 Most Common Words (Preprocessed Text)')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

# Add value labels on top of bars
for i, bar in enumerate(bars.patches):
    plt.text(i, bar.get_height(), f'{int(bar.get_height()):,}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig("eda_plots/top_20_words_preprocessed.png", dpi=300, bbox_inches='tight')
plt.close()

# Generate word cloud from top 1000 words
print("\nGenerating word cloud...")
top_1000_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:1000])
wordcloud = WordCloud(width=1600, height=800,
                     background_color='white',
                     min_font_size=10).generate_from_frequencies(top_1000_words)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Top 1000 Words in Preprocessed Text')
plt.tight_layout(pad=0)
plt.savefig("eda_plots/wordcloud_preprocessed.png", dpi=300, bbox_inches='tight')
plt.close()

# Print sample texts
print("\nSample preprocessed texts:")
sample_texts = df_sample['preprocessed_text'].sample(5, seed=42)
for i, text in enumerate(sample_texts, 1):
    print(f"\n{i}. {text[:200]}...")

# Calculate and print text reduction statistics
print("\nCalculating text reduction statistics...")
reduction_stats = df.with_columns([
    ((pl.col('original_char_count') - pl.col('processed_char_count')) / 
     pl.col('original_char_count') * 100).alias('char_reduction_percent'),
    ((pl.col('original_word_count') - pl.col('processed_word_count')) / 
     pl.col('original_word_count') * 100).alias('word_reduction_percent')
])

print("\nText reduction statistics:")
print("\nCharacter reduction percentage:")
print(reduction_stats['char_reduction_percent'].describe())
print("\nWord reduction percentage:")
print(reduction_stats['word_reduction_percent'].describe())

# Clean up
del df, df_sample, reduction_stats
gc.collect()

print("\nEDA completed successfully!")

