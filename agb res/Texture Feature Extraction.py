# import numpy as np
# from skimage import io
# from skimage.feature import graycomatrix, graycoprops
#
# def compute_glcm_features(channel, levels=16, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
#     """计算指定通道的7个GLCM纹理特征"""
#     # 量化图像到指定灰度级
#     quantized = (channel // (256 // levels)).clip(0, levels - 1).astype(np.uint8)
#
#     # 计算灰度共生矩阵
#     glcm = graycomatrix(quantized,
#                         distances=distances,
#                         angles=angles,
#                         levels=levels,
#                         symmetric=True,
#                         normed=True)
#
#     # 初始化特征字典
#     features = {}
#
#     # 计算常规特征（取多方向均值）
#     properties = {
#         'Con': 'contrast',
#         'homm': 'homogeneity',
#         'Dis': 'dissimilarity',
#         'Cor': 'correlation',
#         'Sec': 'ASM'
#     }
#     for feat, prop in properties.items():
#         features[feat] = np.mean(graycoprops(glcm, prop))
#
#     # 计算方差和熵（需手动计算）
#     var_total = 0.0
#     entropy_total = 0.0
#     num_combinations = glcm.shape[2] * glcm.shape[3]
#
#     for d in range(glcm.shape[2]):
#         for a in range(glcm.shape[3]):
#             matrix = glcm[:, :, d, a]
#
#             # 计算方差
#             i, j = np.indices(matrix.shape)
#             mu = np.sum(i * matrix)
#             var = np.sum(matrix * (i - mu) ** 2)
#             var_total += var
#
#             # 计算熵
#             epsilon = 1e-12  # 避免log(0)
#             entropy = -np.sum(matrix * np.log(matrix + epsilon))
#             entropy_total += entropy
#
#     # 存储平均特征值
#     features['Var'] = var_total / num_combinations
#     features['Ent'] = entropy_total / num_combinations
#
#     return features
#
#
# # 读取图像
# # image = io.imread('C:\\Users\LW1\Desktop\2image220.jpg')
# image = io.imread('C:\\Users\LW1\Desktop\2image220.jpg')
#
# # 分离RGB通道
# r_channel = image[:, :, 0]
# g_channel = image[:, :, 1]
# b_channel = image[:, :, 2]
#
# # 设置计算参数
# params = {
#     'levels': 16,  # 量化灰度级数
#     'distances': [1],  # 像素对距离
#     'angles': [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 四个方向
# }
#
# # 计算各通道特征
# r_features = compute_glcm_features(r_channel, **params)
# g_features = compute_glcm_features(g_channel, **params)
# b_features = compute_glcm_features(b_channel, **params)
#
# # 组合所有特征（共21个）
# feature_names = ['Var', 'homm', 'Con', 'Dis', 'Ent', 'Sec', 'Cor']
# texture_features = {
#     f'{band}_{feat}': value
#     for band, features in zip(['R', 'G', 'B'], [r_features, g_features, b_features])
#     for feat, value in features.items()
# }
#
# # 打印结果
# print("提取的21个纹理特征：")
# for feature, value in texture_features.items():
#     print(f"{feature}: {value:.4f}")
#
# import numpy as np
# from skimage import io
# from skimage.feature import graycomatrix, graycoprops
# import os
#
#
# def compute_glcm_features(channel, levels=16, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
#     """计算指定通道的7个GLCM纹理特征"""
#     quantized = (channel // (256 // levels)).clip(0, levels - 1).astype(np.uint8)
#     glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
#
#     features = {}
#     properties = {'Con': 'contrast', 'homm': 'homogeneity', 'Dis': 'dissimilarity', 'Cor': 'correlation', 'Sec': 'ASM'}
#     for feat, prop in properties.items():
#         features[feat] = np.mean(graycoprops(glcm, prop))
#
#     var_total = entropy_total = 0.0
#     num_combinations = glcm.shape[2] * glcm.shape[3]
#     for d in range(glcm.shape[2]):
#         for a in range(glcm.shape[3]):
#             matrix = glcm[:, :, d, a]
#             i, j = np.indices(matrix.shape)
#             mu = np.sum(i * matrix)
#             var_total += np.sum(matrix * (i - mu) ** 2)
#             entropy_total += -np.sum(matrix * np.log(matrix + 1e-12))
#
#     features.update({'Var': var_total / num_combinations, 'Ent': entropy_total / num_combinations})
#     return features
#
#
# # 动态构建路径
# desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
# image_path = os.path.join(desktop, 'DJI_20240727121618_0030_07271.JPG')
#
# if not os.path.exists(image_path):
#     print(f"错误：文件未找到 → {image_path}")
#     print("解决方案：")
#     print("1. 确认文件已保存到桌面")
#     print("2. 右键文件属性检查完整名称")
#     print("3. 尝试将文件复制到代码所在目录并使用相对路径")
#     exit()
#
# image = io.imread(image_path)
#
# # 后续处理保持不变...
# r_channel = image[:, :, 0]
# g_channel = image[:, :, 1]
# b_channel = image[:, :, 2]
#
# params = {'levels': 16, 'distances': [1], 'angles': [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]}
# r_features = compute_glcm_features(r_channel, **params)
# g_features = compute_glcm_features(g_channel, **params)
# b_features = compute_glcm_features(b_channel, **params)
#
# texture_features = {
#     f'{band}_{feat}': value
#     for band, features in zip(['R', 'G', 'B'], [r_features, g_features, b_features])
#     for feat, value in features.items()
# }
#
# print("提取的21个纹理特征：")
# for feature, value in texture_features.items():
#     print(f"{feature}: {value:.4f}")

import numpy as np
from skimage import io
from skimage.feature import graycomatrix, graycoprops
import os
import pandas as pd


def compute_glcm_features(channel, levels=16, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    """计算指定通道的7个GLCM纹理特征"""
    quantized = (channel // (256 // levels)).clip(0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    features = {}
    properties = {'Con': 'contrast', 'homm': 'homogeneity', 'Dis': 'dissimilarity', 'Cor': 'correlation', 'Sec': 'ASM'}
    for feat, prop in properties.items():
        features[feat] = np.mean(graycoprops(glcm, prop))

    var_total = entropy_total = 0.0
    num_combinations = glcm.shape[2] * glcm.shape[3]
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            matrix = glcm[:, :, d, a]
            i, j = np.indices(matrix.shape)
            mu = np.sum(i * matrix)
            var_total += np.sum(matrix * (i - mu) ** 2)
            entropy_total += -np.sum(matrix * np.log(matrix + 1e-12))

    features.update({'Var': var_total / num_combinations, 'Ent': entropy_total / num_combinations})
    return features


def process_image(image_path):
    """
    处理单张图片，提取纹理特征
    :param image_path: 图像路径
    :return: 提取的纹理特征
    """
    # 读取图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    image = io.imread(image_path)

    # 提取RGB通道
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    # 设置GLCM计算的参数
    params = {'levels': 16, 'distances': [1], 'angles': [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]}

    # 计算每个通道的纹理特征
    r_features = compute_glcm_features(r_channel, **params)
    g_features = compute_glcm_features(g_channel, **params)
    b_features = compute_glcm_features(b_channel, **params)

    # 将特征整理到字典中
    texture_features = {
        f'{band}_{feat}': value
        for band, features in zip(['R', 'G', 'B'], [r_features, g_features, b_features])
        for feat, value in features.items()
    }

    return texture_features


def process_directory(directory_path, output_excel):
    """
    批量处理文件夹中的所有JPG文件并保存统计数据
    :param directory_path: 包含JPG图片的文件夹路径
    :param output_excel: 输出Excel文件路径
    """
    # 获取所有JPG文件
    jpg_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.jpg')]

    all_results = []

    # 遍历文件夹中的每个JPG文件进行处理
    for jpg_file in jpg_files:
        image_path = os.path.join(directory_path, jpg_file)
        try:
            print(f"正在处理 {jpg_file}...")
            texture_features = process_image(image_path)
            texture_features['Image'] = jpg_file  # 添加图片文件名作为标识
            all_results.append(texture_features)
        except Exception as e:
            print(f"处理 {jpg_file} 失败: {str(e)}")

    # 合并所有结果
    if all_results:
        final_df = pd.DataFrame(all_results)

        # 保存到Excel
        final_df.to_excel(output_excel, index=False, engine='openpyxl')
        print(f"所有结果已保存到 {output_excel}")
    else:
        print("没有成功处理任何图片。")


# 修改最后的主程序部分
if __name__ == "__main__":
    # 示例文件夹路径和输出文件
    folder_path = 'E:\\70M纵向\\images'  # 请替换为您的文件夹路径
    output_excel = 'E:\\70M纵向\\images\\texture_features_batch_results.xlsx'

    process_directory(folder_path, output_excel)

