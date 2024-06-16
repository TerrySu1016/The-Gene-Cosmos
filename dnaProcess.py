import os
from scipy.sparse import dok_matrix, csr_matrix, save_npz
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def read_dna_sequence_from_fasta(file_path, start_index, sample_size):
    with open(file_path, 'r') as file:
        file_content = file.read().splitlines()
        dna_sequence = ''.join(file_content[1:])[start_index:start_index+sample_size]
        return dna_sequence

def update_frequency_matrix(dna_sequence, initial_size, buffer_size=500):
    movement = {'A': [1, 0], 'G': [0, 1], 'C': [0, -1], 'T': [-1, 0]}
    center = [initial_size // 2, initial_size // 2]
    frequency_matrix = dok_matrix((initial_size, initial_size), dtype=np.int32)
    current_pos = np.array(center)

    for nucleotide in tqdm(dna_sequence, desc='Processing DNA Sequence'):
        if nucleotide in movement:
            move = movement[nucleotide]
            current_pos += move

            if (current_pos < buffer_size).any() or (current_pos >= np.array(frequency_matrix.shape) - buffer_size).any():
                new_size = (frequency_matrix.shape[0] + 2 * buffer_size, frequency_matrix.shape[1] + 2 * buffer_size)
                new_frequency_matrix = dok_matrix(new_size, dtype=np.int32)
                new_frequency_matrix[buffer_size:-buffer_size, buffer_size:-buffer_size] = frequency_matrix
                frequency_matrix = new_frequency_matrix
                current_pos += buffer_size
            frequency_matrix[tuple(current_pos)] += 1

    return frequency_matrix.tocsr()
def save_frequency_matrix_xi(frequency_matrix, start_index, sample_size, folder_path):
    # Check if the folder_path exists, if not, create it
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"frequency_matrix_{start_index}_{sample_size}.npz"
    save_npz(os.path.join(folder_path, file_name), frequency_matrix)
    print(f"Matrix saved to: {os.path.join(folder_path, file_name)}")
    
def analyze_and_save_histogram(frequency_matrix, start_index, sample_size, description_folder, png_folder):
    length, width = frequency_matrix.shape
    total_elements = length * width
    zero_elements_ratio = (total_elements - frequency_matrix.nnz) / total_elements

    # 保存描述性文本
    txt_file_name = f"description_{start_index}_{sample_size}.txt"
    with open(os.path.join(description_folder, txt_file_name), 'w') as f:
        f.write(f"Matrix dimensions: Length = {length}, Width = {width}\n")
        f.write(f"Proportion of zero elements: {zero_elements_ratio:.4f}\n")

    # 生成直方图
    if frequency_matrix.nnz > 0:
        data = frequency_matrix.data
        plt.hist(data, bins=range(1, int(data.max()) + 2), align='left')
        plt.title('Histogram of Frequency Counts')
        plt.xlabel('Frequency Count')
        plt.ylabel('Number of Cells')
        plt.grid(axis='y')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置x轴刻度为整数
        # 保存直方图
        png_file_name = f"histogram_{start_index}_{sample_size}.png"
        plt.savefig(os.path.join(png_folder, png_file_name))
        plt.close()


# Example usage
sample_size = 30000000
num_samples = 1
initial_size = 1000000
fasta_file_path = 'D:/DNAdata/Data/Y_chr.fasta'
matrix_folder = 'D:/DNAdata/Data/Matrices'
description_folder = 'D:/DNAdata/Data/Descriptions/txt'
png_folder = 'D:/DNAdata/Data/Descriptions/png'
start_index = 0

for i in range(num_samples):
    dna_sequence = read_dna_sequence_from_fasta(fasta_file_path, start_index, sample_size)
    frequency_matrix = update_frequency_matrix(dna_sequence, initial_size)
    save_frequency_matrix_xi(frequency_matrix, start_index, sample_size, matrix_folder)
    analyze_and_save_histogram(frequency_matrix, start_index, sample_size, description_folder, png_folder)
    start_index += sample_size
