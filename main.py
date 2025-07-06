import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import os
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from scipy.fft import fft
import pywt
from pyts.image import RecurrencePlot

# --- Controllable Parameter ---
ANALYSIS_RANGE = 100000            # Range of numbers to analyze in the Collatz conjecture

# Create directories if they don't exist
if not os.path.exists('graphs'):
    os.makedirs('graphs')
if not os.path.exists('advanced_analysis'):
    os.makedirs('advanced_analysis')

full_sequences_memo = {}
shortened_sequences_memo = {}
def _calculate_full_collatz_sequence(n):
    sequence = [n]
    current_n = n
    while current_n != 1:
        if current_n in full_sequences_memo:
            sequence.extend(full_sequences_memo[current_n][1:])
            break
        if current_n % 2 == 0:
            current_n = current_n // 2
        else:
            current_n = 3 * current_n + 1
        sequence.append(current_n)
    full_sequences_memo[n] = sequence
    return sequence

def _get_shortened_collatz_string(n):
    shortened_sequence = [n]
    current_n = n
    while current_n != 1:
        if current_n in shortened_sequences_memo and current_n != n: # If we encounter a number whose shortened sequence is known (and it's not the starting number itself)
            shortened_sequence.append(1) # Directly append 1 as the rest is known to lead to 1
            break
        if current_n % 2 == 0:
            current_n = current_n // 2
        else:
            current_n = 3 * current_n + 1
        shortened_sequence.append(current_n)
    shortened_str = " -> ".join(map(str, shortened_sequence))
    shortened_sequences_memo[n] = shortened_str # Store the newly generated shortened string
    return shortened_str

print(f"Generating Collatz sequences for numbers from 1 to {ANALYSIS_RANGE}...")

sequences = {i: _calculate_full_collatz_sequence(i) for i in range(1, ANALYSIS_RANGE + 1)}
print("Collatz sequences generated.")

max_len = max(len(s) for s in sequences.values())
longest_seq_start = max(sequences, key=lambda k: len(sequences[k]))
longest_sequence = sequences[longest_seq_start]

# --- Basic File Writing ---
print("Writing basic sequence data to .txt files...")

with open('collatz_sequences.txt', 'w') as f:
    f.write("{:<10} {:<20} {:<20}\n".format("Number", "Sequence Length", "Max Value"))
    f.write("-" * 50 + "\n")
    for i, sequence in sequences.items():
        f.write("{:<10} {:<20} {:<20}\n".format(i, len(sequence), max(sequence)))

with open('collatz_detailed_sequences.txt', 'w') as f:
    for i, sequence in sequences.items():
        f.write(f"Sequence for {i}:\n")
        f.write(" -> ".join(map(str, sequence)))
        f.write("\n" + "="*50 + "\n")

print("Collatz sequences and detailed sequences have been written to .txt files.")

# --- New Memoized Detailed Sequences File ---
print("Writing memoized detailed sequence data to a new .txt file...")

with open('collatz_memoized_detailed_sequences.txt', 'w') as f:
    for i, sequence in sequences.items():
        f.write(f"Sequence for {i}:\n")
        f.write(_get_shortened_collatz_string(i))
        f.write("\n" + "="*50 + "\n")

print("Memoized detailed sequences have been written to collatz_memoized_detailed_sequences.txt.")

# --- All Plotting and Analysis ---

print("Starting advanced analysis and plotting...")

# 1. Sequence Length Distribution (Histogram)
plt.figure(figsize=(10, 6))
sequence_lengths = [len(s) for s in sequences.values()]
plt.hist(sequence_lengths, bins=50, edgecolor='black')
plt.title('Distribution of Collatz Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.savefig('graphs/1_sequence_length_distribution.png')
plt.close()
pd.DataFrame(sequence_lengths, columns=['sequence_length']).to_csv('advanced_analysis/2_sequence_lengths.csv', index=False)

print("Basic analysis and initial plots completed.")

print("Starting advanced analysis and plotting...")

# 2. Max Value Distribution (Histogram)

print("Calculating max values for sequences...")

plt.figure(figsize=(10, 6))
max_values = [max(s) for s in sequences.values()]
plt.hist(max_values, bins=50, edgecolor='black')
plt.title('Distribution of Max Values in Collatz Sequences')
plt.xlabel('Max Value')
plt.ylabel('Frequency')
plt.savefig('graphs/3_max_value_distribution.png')
plt.close()
pd.DataFrame(max_values, columns=['max_value']).to_csv('advanced_analysis/4_max_values.csv', index=False)

print("Max values calculated and histogram plotted.")

# 3. Scatter plot of Length vs. Max Value
print("Creating scatter plots...")

plt.figure(figsize=(10, 8))
plt.scatter(sequence_lengths, max_values, alpha=0.5)
plt.title('Collatz Sequence Length vs. Max Value')
plt.xlabel('Sequence Length')
plt.ylabel('Max Value')
plt.savefig('graphs/5_length_vs_max_value_scatter.png')
plt.close()
pd.DataFrame({'sequence_length': sequence_lengths, 'max_value': max_values}).to_csv('advanced_analysis/6_length_vs_max_value.csv', index=False)

print("Scatter plot created.")


# 4. Steps to 1 Distribution (Histogram) - This is essentially sequence length - 1

print("Calculating steps to reach 1...")

plt.figure(figsize=(10, 6))
steps_to_one = [len(s) - 1 for s in sequences.values()]
plt.hist(steps_to_one, bins=50, edgecolor='black')
plt.title('Distribution of Steps to Reach 1')
plt.xlabel('Number of Steps')
plt.ylabel('Frequency')
plt.savefig('graphs/7_steps_to_one_distribution.png')
plt.close()
pd.DataFrame(steps_to_one, columns=['steps_to_one']).to_csv('advanced_analysis/8_steps_to_one.csv', index=False)

print("Steps to reach 1 calculated and histogram plotted.")


# 5. Hierarchical Clustering Dendrogram (on sequence lengths/max values)
# Using a subset for visualization due to potential computational intensity

print("Starting hierarchical clustering and autocorrelation analysis...")

subset_size = min(500, ANALYSIS_RANGE)
subset_indices = np.random.choice(range(1, ANALYSIS_RANGE + 1), subset_size, replace=False)
subset_data = np.array([[len(sequences[i]), max(sequences[i])] for i in subset_indices])
if len(subset_data) > 1: # Ensure there's enough data for clustering
    Z = linkage(pdist(subset_data), 'ward')
    plt.figure(figsize=(12, 7))
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    plt.title('Hierarchical Clustering Dendrogram of Collatz Sequences (Subset)')
    plt.xlabel('Sequence Index')
    plt.ylabel('Distance')
    plt.savefig('graphs/9_hierarchical_clustering_dendrogram.png')
    plt.close()
    pd.DataFrame(Z).to_csv('advanced_analysis/10_hierarchical_clustering_linkage.csv', index=False)

print("Hierarchical clustering dendrogram created.")

# 6. Autocorrelation Plot of Longest Sequence
print("Calculating autocorrelation for the longest sequence...")

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, pacf
plt.figure(figsize=(10, 6))
plot_acf(longest_sequence, lags=min(50, len(longest_sequence) - 1), ax=plt.gca())
plt.title('Autocorrelation Function of Longest Collatz Sequence')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.savefig('graphs/11_autocorrelation_plot.png')
plt.close()

print("Autocorrelation plot created.")

# Calculate and save autocorrelation values

print("Calculating autocorrelation values...")

acf_values = pd.Series(acf(longest_sequence, nlags=min(50, len(longest_sequence) - 1)))
acf_values.to_csv('advanced_analysis/12_autocorrelation_values.csv', index=False)

print("Autocorrelation values saved.")

# 7. Partial Autocorrelation Plot of Longest Sequence

print("Calculating partial autocorrelation for the longest sequence...")

from statsmodels.graphics.tsaplots import plot_pacf
plt.figure(figsize=(10, 6))
plot_pacf(longest_sequence, lags=min(50, len(longest_sequence) - 1), ax=plt.gca())
plt.title('Partial Autocorrelation Function of Longest Collatz Sequence')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.savefig('graphs/13_partial_autocorrelation_plot.png')
plt.close()
# Calculate and save partial autocorrelation values
pacf_values = pd.Series(pacf(longest_sequence, nlags=min(50, len(longest_sequence) - 1)))
pacf_values.to_csv('advanced_analysis/14_partial_autocorrelation_values.csv', index=False)

print("Partial autocorrelation values saved.")
# 8. Network Graph (for a small subset of numbers to visualize transitions)
# Create a directed graph for a small range to visualize transitions

print("Creating network graph of Collatz transitions...")
graph_range = min(100, ANALYSIS_RANGE) # Limit for visualization
G = nx.DiGraph()
for i in range(1, graph_range + 1):
    seq = _calculate_full_collatz_sequence(i)
    for j in range(len(seq) - 1):
        G.add_edge(seq[j], seq[j+1])

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20) # positions for all nodes
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.9)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6, arrows=True)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
plt.title(f'Collatz Transition Network (Numbers 1 to {graph_range})')
plt.axis('off')
plt.savefig('graphs/15_collatz_network_graph.png')
plt.close()

# Save edge list
with open('advanced_analysis/16_collatz_network_edges.csv', 'w') as f:
    f.write("source,target\n")
    for edge in G.edges():
        f.write(f"{edge[0]},{edge[1]}\n")

print("Network graph created.")
# 9. Shannon Entropy of Sequence Lengths

print("Calculating Shannon entropy of sequence lengths...")

from scipy.stats import entropy
counts = Counter(sequence_lengths)
probabilities = [count / len(sequence_lengths) for count in counts.values()]
shannon_entropy = entropy(probabilities, base=2)
with open('advanced_analysis/17_shannon_entropy_lengths.txt', 'w') as f:
    f.write(f"Shannon Entropy of Collatz Sequence Lengths: {shannon_entropy}\n")

print("Shannon entropy calculated and saved.")
# 10. First Passage Time Distribution (Histogram) - Time to reach a specific value (e.g., 4)
# For each sequence, find the index of the first occurrence of 4 (before 1)

print("Calculating first passage times to 4...")

first_passage_times_to_4 = []
for i, seq in sequences.items():
    try:
        # Find index of 4, but only if 1 is not encountered before it
        if 4 in seq:
            idx_4 = seq.index(4)
            idx_1 = len(seq) # Assume 1 is at the end if not found earlier
            if 1 in seq:
                idx_1 = seq.index(1)
            if idx_4 < idx_1: # 4 must appear before 1
                first_passage_times_to_4.append(idx_4)
    except ValueError:
        pass # 4 not in sequence

if first_passage_times_to_4:
    plt.figure(figsize=(10, 6))
    plt.hist(first_passage_times_to_4, bins=max(1, len(set(first_passage_times_to_4)) // 5), edgecolor='black')
    plt.title('Distribution of First Passage Times to 4')
    plt.xlabel('Steps to 4')
    plt.ylabel('Frequency')
    plt.savefig('graphs/17_first_passage_time_to_4_distribution.png')
    plt.close()
    pd.DataFrame(first_passage_times_to_4, columns=['first_passage_time_to_4']).to_csv('advanced_analysis/18_first_passage_times_to_4.csv', index=False)
else:
    print("No sequences reached 4 before 1 in the given range for First Passage Time analysis.")


print("First passage times to 4 calculated and histogram plotted.")

# --- Advanced Data Analysis & Plotting ---

print("Starting advanced data analysis...")

df = pd.DataFrame({
    'length': [len(s) for s in sequences.values()],
    'max_val': [max(s) for s in sequences.values()],
    'steps_to_max': [np.argmax(s) for s in sequences.values()],
})
df.describe().to_csv('advanced_analysis/1_statistical_summary.csv')

print("Statistical summary saved.")
# --- Ultra-Advanced Mathematical Analysis ---

print("Starting ultra-advanced mathematical analysis...")

# 1. Principal Component Analysis (PCA)

print("Performing PCA (Principal Component Analysis) on Collatz sequences...")

padded_sequences = np.array([s + [0] * (max_len - len(s)) for s in sequences.values()])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(padded_sequences)
pd.DataFrame(pca.explained_variance_ratio_).to_csv('advanced_analysis/15_pca_variance.csv')

plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['length'])
plt.title('PCA of Collatz Sequences')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Sequence Length')
plt.savefig('graphs/21_pca_plot.png')
plt.close()

print("PCA completed and plot saved.")

# 2. Time Series Analysis (ARIMA)

print("Performing Time Series Analysis using ARIMA on the longest sequence...")

model = ARIMA(longest_sequence, order=(5,1,0))
model_fit = model.fit()
with open('advanced_analysis/16_arima_summary.txt', 'w') as f:
    f.write(model_fit.summary().as_text())

print("ARIMA model fitted and summary saved.")

# 3. Fourier Transform

print("Performing Fourier Transform on the longest sequence...")

fft_vals = np.abs(fft(longest_sequence))[:len(longest_sequence)//2]
plt.figure(figsize=(10, 6))
plt.plot(fft_vals)
plt.title('Power Spectral Density of Longest Sequence')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.savefig('graphs/18_fourier_transform.png')
plt.close()

print("Fourier Transform completed and plot saved.")

# 4. Wavelet Transform

print("Performing Wavelet Transform on the longest sequence...")

coeffs = pywt.wavedec(longest_sequence, 'db1', level=5)
plt.figure(figsize=(12, 8))
for i, c in enumerate(coeffs):
    plt.subplot(len(coeffs), 1, i + 1)
    plt.plot(c)
    plt.title(f'Wavelet Coefficients - Level {i}')
plt.tight_layout()
plt.savefig('graphs/19_wavelet_transform.png')
plt.close()

print("Wavelet Transform completed and plot saved.")

# 5. Recurrence Plot
print("Creating Recurrence Plot for the longest sequence...")

rp = RecurrencePlot(threshold='point', percentage=20)
X_rp = rp.fit_transform([longest_sequence])
plt.figure(figsize=(8, 8))
plt.imshow(X_rp[0], cmap='binary', origin='lower')
plt.title('Recurrence Plot of Longest Sequence')
plt.savefig('graphs/20_recurrence_plot.png')
plt.close()

print("Recurrence Plot created and saved.")




print("All analyses and plots completed successfully.")