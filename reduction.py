import os
import numpy as np
from scipy.sparse import load_npz, coo_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import load_npz
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull

def find_non_zero_bounding_box(matrix):
    # Find the row and column indices of non-zero elements
    non_zero_rows, non_zero_cols = matrix.nonzero()
    
    # Get the minimum and maximum row and column indices
    row_min, row_max = np.min(non_zero_rows), np.max(non_zero_rows)
    col_min, col_max = np.min(non_zero_cols), np.max(non_zero_cols)
    
    return row_min, row_max, col_min, col_max


def preprocess_matrix(matrix):
    # Find the bounding box of non-zero elements
    row_min, row_max, col_min, col_max = find_non_zero_bounding_box(matrix)
    
    # Slice the matrix to focus only on non-zero elements
    preprocessed_matrix = matrix[row_min:row_max+1, col_min:col_max+1]
    
    return preprocessed_matrix

def load_and_reduce_matrix_sparse(npz_path, target_size):
    # 加载稀疏矩阵
    matrix = load_npz(npz_path)
    
    # Preprocess the matrix to focus only on non-zero elements
    preprocessed_matrix = preprocess_matrix(matrix)
    
    n_rows, n_cols = preprocessed_matrix.shape

    # 计算新的尺寸
    reduced_rows = 100
    reduced_cols = 100
    
    block_col = n_cols // 100
    block_row = n_rows // 100
    
    # 创建一个空的 COO 矩阵来保存降维后的结果
    data, rows, cols = [], [], []

    # 遍历每个块计算块平均值
    for i in tqdm(range(reduced_rows), desc='Reducing Rows'):
        for j in tqdm(range(reduced_cols), desc='Reducing Columns', leave=False):
            start_row = i * block_row
            end_row = (i + 1) * block_row
            start_col = j * block_col
            end_col = (j + 1) * block_col
            
            block = preprocessed_matrix[start_row:end_row, start_col:end_col]
            average = block.mean()
            
            if average > 0:
                rows.append(i)
                cols.append(j)
                data.append(average)
    
    reduced_matrix = coo_matrix((data, (rows, cols)), shape=(reduced_rows, reduced_cols))

    return reduced_matrix
#colors_hex = ['#FFC25A', '#B9562F', '#D96E44', '#2D517B', '#1B2C3C']
colors_hex =  ['#1B2C3C', '#FF0000', '#FFFF00']
rgb_colors_normalized = [(int(hex_color[1:3], 16)/255, int(hex_color[3:5], 16)/255, int(hex_color[5:], 16)/255) for hex_color in colors_hex]
custom_colormap = LinearSegmentedColormap.from_list("custom_colormap", rgb_colors_normalized, N=256)

def plot_zoomed_heatmap(matrix, output_folder, file_name):
    # Convert COO matrix to CSR format to support slicing
    matrix_csr = matrix.tocsr()
    
    # Slice the matrix and convert to dense array for plotting
    zoomed_matrix = matrix_csr.toarray()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(zoomed_matrix, cmap=custom_colormap, interpolation='nearest')
    plt.colorbar()
    plt.title("Zoomed Heatmap of Reduced Frequency Matrix")
    plt.savefig(os.path.join(output_folder, file_name))
    plt.close()
    print(f"Zoomed heatmap saved to: {os.path.join(output_folder, file_name)}")

# 示例使用
npz_file_path = 'D:/DNAdata/Data/Matrices/frequency_matrix_0_10000000.npz'
output_folder = 'D:/DNAdata/pic'
block_size = 0
reduced_matrix = load_and_reduce_matrix_sparse(npz_file_path, block_size)
plot_zoomed_heatmap(reduced_matrix, output_folder, 'reduced_heatmap_final.png')