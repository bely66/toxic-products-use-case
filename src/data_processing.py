import random
import pandas as pd
from matplotlib import pyplot as plt
import os
from wordcloud import WordCloud
from collections import Counter
import arabic_reshaper
from bidi.algorithm import get_display
from src.preprocessing import preprocess_data


def process_data(df_path, tag):
    # Load the CSV file in chunks
    chunk_size = 1000  # Adjust the chunk size as per your requirements
    total_rows = sum(1 for _ in open(df_path)) - 1  # Get the total number of rows in the file (excluding the header)
    skip_rows = sorted(random.sample(range(1, total_rows + 1), total_rows - 100))  # Generate a list of rows to skip

    chunks = pd.read_csv(df_path, skiprows=skip_rows, chunksize=chunk_size)

    df = pd.concat(chunks)
    df = df[['final_product_name', 'final_product_description', 'toxicity']]
    df['tag'] = tag
    df = df.dropna(subset=['final_product_name'], how='all')
    df = df.reset_index(drop=True)
    df = df.fillna('')
    df['soup_of_text'] = [f"{df['final_product_name'][i]} {df['final_product_description'][i]}" for i in range(len(df))]
    df['soup_of_text_clean'] = df['soup_of_text'].apply(lambda q: preprocess_data(q))
    df['toxicity'] = df.toxicity.apply(lambda x: int(x))

    texts = df['soup_of_text_clean']
    labels = df['toxicity']

    return texts, labels

# def preprocess_and_analyze(csv_path, subset='train'):
#     # Load the CSV file
#     df = pd.read_csv(csv_path)

#     df = df[['final_product_name', 'final_product_description', 'toxicity']]
#     df['tag'] = tag
#     df = df.dropna(subset=['final_product_name'], how='all')
#     df = df.reset_index(drop=True)
#     df = df.fillna('')
#     df['soup_of_text'] = [f"{df['final_product_name'][i]} {df['final_product_description'][i]}" for i in range(len(df))]
#     df['soup_of_text_clean'] = df['soup_of_text'].apply(lambda q: preprocess_data(q))
#     df['toxicity'] = df.toxicity.apply(lambda x: int(x))


#     save_dir = f'stats/{subset}'
#     # check if the directory exists
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # Initial percentages of toxic vs non-toxic products
#     initial_toxic_percentage = df['toxicity'].mean() * 100
#     initial_non_toxic_percentage = (1 - df['toxicity'].mean()) * 100
    
#     # Dropping NaN values
#     df_dropna = df.dropna()
#     toxic_percentage_dropna = df_dropna['toxicity'].mean() * 100
#     non_toxic_percentage_dropna = (1 - df_dropna['toxicity'].mean()) * 100
    
#     # Dropping duplicates
#     df_drop_duplicates = df.drop_duplicates()
#     toxic_percentage_drop_duplicates = df_drop_duplicates['toxicity'].mean() * 100
#     non_toxic_percentage_drop_duplicates = (1 - df_drop_duplicates['toxicity'].mean()) * 100
    
#     # Percentage of duplicates and NaN values
#     duplicates_percentage = ((len(df) - len(df.drop_duplicates())) / len(df)) * 100
#     nan_values_percentage = ((len(df) - len(df_dropna)) / len(df)) * 100
    
#     # Graphing the statistics
#     categories = ['Initial', 'After Dropping NaN', 'After Dropping Duplicates']
#     toxic_percentages = [initial_toxic_percentage, toxic_percentage_dropna, toxic_percentage_drop_duplicates]
#     non_toxic_percentages = [initial_non_toxic_percentage, non_toxic_percentage_dropna, non_toxic_percentage_drop_duplicates]

#     plot_path = f'{save_dir}/toxic_vs_nontoxic_stats.png'
#     plt.figure(figsize=(10, 6))
#     plt.bar(categories, toxic_percentages, color='red', alpha=0.6, label='Toxic')
#     plt.bar(categories, non_toxic_percentages, bottom=toxic_percentages, color='green', alpha=0.6, label='Non-Toxic')
#     plt.ylabel('Percentage')
#     plt.title('Toxic vs Non-Toxic Products')
#     plt.legend()
#     plt.savefig(plot_path)
#     plt.show()
    
#     # Writing statistics to a file
#     stats = {
#         "dataset_size": len(df),
#         "dataset_size_after_dropping_duplicates": len(df_drop_duplicates),
#         "dataset_size_after_dropping_nan": len(df_dropna),
#         'Initial Toxic Percentage': initial_toxic_percentage,
#         'Initial Non-Toxic Percentage': initial_non_toxic_percentage,
#         'Toxic Percentage After Dropping NaN': toxic_percentage_dropna,
#         'Non-Toxic Percentage After Dropping NaN': non_toxic_percentage_dropna,
#         'Toxic Percentage After Dropping Duplicates': toxic_percentage_drop_duplicates,
#         'Non-Toxic Percentage After Dropping Duplicates': non_toxic_percentage_drop_duplicates,
#         'Duplicates Percentage': duplicates_percentage,
#         'NaN Values Percentage': nan_values_percentage
#     }
#     stats_path = f'{save_dir}/data_analysis_stats.txt'
#     with open(stats_path, 'w') as f:
#         for key, value in stats.items():
#             if 'percentage' in key.lower():
#                 f.write(f"{key}: {value}%\n")
#             else:
#                 f.write(f"{key}: {value}\n")

def load_and_prepare_dataframe(csv_path, subset):
    """
    Loads the CSV file and prepares the dataframe by filtering necessary columns
    and creating a combined text column.
    """
    df = pd.read_csv(csv_path)
    df = df[['final_product_name', 'final_product_description', 'toxicity']]
    
    return df

def calculate_statistics(df):
    """
    Calculates and returns various statistics from the dataframe.
    """
    initial_toxic_percentage = df['toxicity'].mean() * 100
    df_dropna = df.dropna()
    df_drop_duplicates = df.drop_duplicates()
    
    stats = {
        "dataset_size": len(df),
        "dataset_size_after_dropping_duplicates": len(df_drop_duplicates),
        "dataset_size_after_dropping_nan": len(df_dropna),
        "Initial Toxic Percentage": initial_toxic_percentage,
        "Toxic Percentage After Dropping NaN": df_dropna['toxicity'].mean() * 100,
        "Toxic Percentage After Dropping Duplicates": df_drop_duplicates['toxicity'].mean() * 100,
        "Duplicates Percentage": (len(df) - len(df_drop_duplicates)) / len(df) * 100,
        "NaN Values Percentage": (len(df) - len(df_dropna)) / len(df) * 100
    }
    return stats

def plot_statistics(stats, save_dir):
    """
    Plots and saves the statistics.
    """
    categories = ['Initial', 'After Dropping NaN', 'After Dropping Duplicates']
    toxic_percentages = [stats['Initial Toxic Percentage'], stats['Toxic Percentage After Dropping NaN'], stats['Toxic Percentage After Dropping Duplicates']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, toxic_percentages, color='red', alpha=0.6, label='Toxic')
    plt.ylabel('Percentage')
    plt.title('Toxic Products Percentage')
    plt.legend()
    plot_path = os.path.join(save_dir, 'toxic_vs_nontoxic_stats.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory
    return plot_path

def write_statistics_to_file(stats, save_dir):
    """
    Writes the calculated statistics to a file.
    """
    stats_path = os.path.join(save_dir, 'data_analysis_stats.txt')
    with open(stats_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}%\n" if 'Percentage' in key else f"{key}: {value}\n")
    return stats_path

def generate_word_cloud(df, category, save_dir, top_n=25):
    """
    Generates and saves a word cloud for the specified category (toxic or non-toxic).
    """
    text = " ".join(df[df['toxicity'] == category]['soup_of_text_clean'])
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='data/fonts/Sahel.ttf').generate(bidi_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    category_name = 'toxic' if category == 1 else 'non_toxic'
    cloud_path = os.path.join(save_dir, f'{category_name}_word_cloud.png')
    plt.savefig(cloud_path)
    plt.close()  # Close the plot to free up memory

    # Calculate top-N words
    words = bidi_text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(top_n)
    
    # Plotting the percentages of top-N words
    words, frequencies = zip(*most_common_words)
    percentages = [freq / sum(word_counts.values()) * 100 for freq in frequencies]
    
    plt.figure(figsize=(12, 8))
    plt.barh(words, percentages, color='skyblue')
    plt.xlabel('Percentage (%)')
    plt.title(f'Top-{top_n} Words in {category_name.capitalize()} Category')
    graph_path = os.path.join(save_dir, f'{category_name}_top_words_percentage.png')
    plt.savefig(graph_path)
    plt.close()
    return cloud_path

def prepare_df(df):
    """
    Prepare the dataframe by adding a column of combined text and cleaning the text.
    """
    df['tag'] = subset
    df = df.dropna(subset=['final_product_name'], how='all').reset_index(drop=True)
    df = df.fillna('')
    df['soup_of_text'] = df.apply(lambda row: f"{row['final_product_name']} {row['final_product_description']}", axis=1)
    df['soup_of_text_clean'] = df['soup_of_text'].apply(preprocess_data)  # Assuming preprocess_data is defined elsewhere
    df['toxicity'] = df['toxicity'].apply(int)
    return df
def preprocess_and_analyze(csv_path, subset='train'):
    """
    Main function to preprocess and analyze the data, including generating word clouds.
    """
    df = load_and_prepare_dataframe(csv_path, subset)
    stats = calculate_statistics(df)
    save_dir = f'stats/{subset}'
    os.makedirs(save_dir, exist_ok=True)
    df = prepare_df(df)
    
    plot_path = plot_statistics(stats, save_dir)
    stats_path = write_statistics_to_file(stats, save_dir)
    
    # Generate word clouds for both toxic and non-toxic categories
    toxic_cloud_path = generate_word_cloud(df, 1, save_dir)
    non_toxic_cloud_path = generate_word_cloud(df, 0, save_dir)
    
    print(f"Stats written to: {stats_path}")
    print(f"Graph saved to: {plot_path}")
    print(f"Toxic word cloud saved to: {toxic_cloud_path}")
    print(f"Non-Toxic word cloud saved to: {non_toxic_cloud_path}")



from bidi.algorithm import get_display
from arabic_reshaper import reshape
import random

def display_data(texts, labels, num_samples):
    label_map = {0: "Non-toxic", 1: "Toxic"}
    samples = random.sample(list(zip(texts, labels)), num_samples)
    for text, label in samples:
        reshaped_text = reshape(text)
        display_text = get_display(reshaped_text)
        print(f"Text: {display_text}")
        print(f"Label: {label_map[label]}")
        print()

def display_data_filter(texts, labels, num_samples, filter_label=1):
    label_map = {0: "Non-toxic", 1: "Toxic"}
    samples = random.sample(list(zip(texts, labels)), num_samples)
    for text, label in samples:
        if label != filter_label:
            continue
        reshaped_text = reshape(text)
        display_text = get_display(reshaped_text)
        print(f"Text: {display_text}")
        print(f"Label: {label_map[label]}")
        print()

# Example usage:
# data/processed_data/test_dataset/test_data_Nov_Dec_Oct_v3.csv
# data/processed_data/train_dataset/all_train_data_v3.csv
df_paths = ["data/processed_data/test_dataset/test_data_Nov_Dec_Oct_v3.csv", "data/processed_data/train_dataset/all_train_data_v3.csv"]
subsets = ["test", "train"]
tag = "all_train"
for df_path, subset in zip(df_paths, subsets):
    preprocess_and_analyze(df_path, subset)

# display_data_filter(train_texts, train_labels, 30, filter_label=1)
