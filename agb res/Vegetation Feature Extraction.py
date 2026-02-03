# import numpy as np
# import cv2
# import os
# import shapefile
# from osgeo import gdal, ogr, osr
# import matplotlib.pyplot as plt
# from skimage import io
#
#
# def normalize_rgb(image):
#     """归一化RGB通道"""
#     # 将图像转换为float32类型以避免溢出
#     img_float = image.astype(np.float32)
#
#     # 计算每个像素的RGB总和
#     sum_rgb = img_float.sum(axis=2)
#
#     # 避免除以零
#     sum_rgb[sum_rgb == 0] = 1e-10
#
#     # 归一化每个通道
#     r_norm = img_float[:, :, 0] / sum_rgb
#     g_norm = img_float[:, :, 1] / sum_rgb
#     b_norm = img_float[:, :, 2] / sum_rgb
#
#     return r_norm, g_norm, b_norm
#
#
# def calculate_vis(r, g, b):
#     """计算10种植被指数"""
#     # 1. Excess Green (EXG)
#     exg = 2 * g - r - b
#
#     # 2. Excess Blue (EXB)
#     exb = 2 * b - r - g
#
#     # 3. Green Leaf Index (GLI)
#     gli = (2 * g - r - b) / (2 * g + r + b)
#
#     # 4. Visible Atmospherically Resistant Index (VARI)
#     vari = (g - r) / (g + r - b)
#
#     # 5. Excess Green minus Excess Red (EXGR)
#     exgr = exg - (1.4 * r - g)
#
#     # 6. Red Green Blue Vegetation Index (RGBVI)
#     rgbvi = (g & zwnj; ** 2 - r * b) / (g ** & zwnj;2 + r * b)
#
#     # 7. Modified Green Red Vegetation Index (MGRVI)
#     mgrvi = (g & zwnj; ** 2 - r ** & zwnj;2) / (g & zwnj; ** 2 + r ** & zwnj;2)
#
#     # 8. Normalized Green-Red Difference Index (NGRDI)
#     ngrdi = (g - r) / (g + r)
#
#     # 9. Green-Red Ratio Index (GRRI)
#     grri = g / r
#
#     # 10. Normalized Difference Index (NDI)
#     ndi = (g - r) / (g + r)
#
#     return {
#         'EXG': exg,
#         'EXB': exb,
#         'GLI': gli,
#         'VARI': vari,
#         'EXGR': exgr,
#         'RGBVI': rgbvi,
#         'MGRVI': mgrvi,
#         'NGRDI': ngrdi,
#         'GRRI': grri,
#         'NDI': ndi
#     }
#
#
# def process_image(image_path, shapefile_path=None):
#     """处理单张图像"""
#     # 读取图像
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # 归一化RGB通道
#     r_norm, g_norm, b_norm = normalize_rgb(image)
#
#     # 计算植被指数
#     vis = calculate_vis(r_norm, g_norm, b_norm)
#
#     # 如果有shapefile，则按地块处理
#     if shapefile_path and os.path.exists(shapefile_path):
#         # 读取shapefile
#         sf = shapefile.Reader(shapefile_path)
#         shapes = sf.shapes()
#
#         # 为每个地块提取植被指数
#         for i, shape in enumerate(shapes):
#             # 这里需要实现地块边界内的像素提取和统计
#             # 由于实现较复杂，这里仅作示意
#             print(f"Processing plot {i + 1}")
#             # 实际应用中需要使用GDAL等库进行精确的栅格统计
#     else:
#         # 如果没有shapefile，返回整个图像的植被指数
#         return vis
#
#     return vis
#
#
# def batch_process_images(image_folder, output_folder):
#     """批量处理图像"""
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 获取所有JPG图像
#     image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
#
#     results = {}
#
#     for img_file in image_files:
#         img_path = os.path.join(image_folder, img_file)
#         print(f"Processing {img_file}...")
#
#         # 处理图像
#         vis = process_image(img_path)
#         results[img_file] = vis
#
#         # 可视化并保存结果
#         plot_vegetation_indices(vis, img_file, output_folder)
#
#     return results
#
#
# def plot_vegetation_indices(vis, img_name, output_folder):
#     """绘制并保存植被指数图"""
#     plt.figure(figsize=(15, 10))
#
#     for i, (name, index) in enumerate(vis.items()):
#         plt.subplot(4, 3, i + 1)
#         plt.imshow(index, cmap='viridis')
#         plt.title(name)
#         plt.colorbar()
#
#     plt.tight_layout()
#     output_path = os.path.join(output_folder, f"{os.path.splitext(img_name)}_vis.png")
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Saved vegetation indices plot to {output_path}")


# # 使用示例
# if __name__ == "__main__":
#     # 单张图像处理
#     image_path = "path_to_your_image.jpg"
#     shapefile_path = "path_to_your_shapefile.shp"  # 可选
#
#     # 处理单张图像
#     vis_results = process_image(image_path, shapefile_path)
#
#     # 批量处理
#     image_folder = "path_to_your_image_folder"
#     output_folder = "path_to_output_folder"
#     batch_results = batch_process_images(image_folder, output_folder)


# import numpy as np
# import rasterio
# import shapefile
# from rasterio.mask import mask
#
# # 计算VI的函数
# def calculate_vi(r, g, b):
#     # 归一化的计算
#     total_dn = r + g + b
#     r_norm = r / total_dn
#     g_norm = g / total_dn
#     b_norm = b / total_dn
#
#     # 计算植被指数
#     EXG = 2 * g_norm - r_norm - b_norm
#     EXB = b_norm - r_norm
#     GLI = (2 * g_norm - r_norm - b_norm) / (2 * g_norm + r_norm + b_norm)
#     VARI = (g_norm - r_norm) / (g_norm + r_norm - b_norm)
#     EXGR = g_norm - r_norm
#     RGBVI = (g_norm - b_norm) / (g_norm + b_norm)
#     MGRVI = (g_norm - r_norm) / (g_norm + r_norm)
#     NGRDI = (g_norm - b_norm) / (g_norm + b_norm)
#     GRRI = g_norm / r_norm
#     NDI = (g_norm - r_norm) / (g_norm + r_norm)
#
#     return EXG, EXB, GLI, VARI, EXGR, RGBVI, MGRVI, NGRDI, GRRI, NDI
#
#
# # 加载JPG图像
# def process_image(image_path, shapefile_path):
#     with rasterio.open(image_path) as src:
#         # 读取图像数据
#         img_data = src.read([1, 2, 3])  # R, G, B
#         img_data = np.moveaxis(img_data, 0, -1)  # 将通道维度移到最后
#
#         # 加载shapefile来提取地块边界
#         sf = shapefile.Reader(shapefile_path)
#         shapes = [shape.__geo_interface__ for shape in sf.shapes()]
#
#         # 通过mask提取地块区域
#         out_image, out_transform = mask(src, shapes, crop=True)
#
#         # 获取地块内的像素
#         r = out_image[0]
#         g = out_image[1]
#         b = out_image[2]
#
#         # 计算VI
#         VI = calculate_vi(r, g, b)
#
#         return VI
#
# # 示例调用
# image_path = 'C:\\Users\LW1\Desktop\2image.jpg'
# shapefile_path = '"C:\\Users\LW1\Desktop\fields".shp'
# VI_values = process_image(image_path, shapefile_path)
#
# # 打印或保存VI值
# for index, vi in zip(['EXG', 'EXB', 'GLI', 'VARI', 'EXGR', 'RGBVI', 'MGRVI', 'NGRDI', 'GRRI', 'NDI'], VI_values):
#     print(f"{index}: {vi}")

#
#
#
# import numpy as np
# from skimage import io
# import os
# import pandas as pd
#
#
# def normalize_dn_values(img):
#     """
#     对RGB通道进行归一化处理
#     :param img: 原始图像数组 (H, W, 3)
#     :return: 归一化的R、G、B通道
#     """
#     # 分离RGB通道
#     r = img[:, :, 0].astype(np.float32)
#     g = img[:, :, 1].astype(np.float32)
#     b = img[:, :, 2].astype(np.float32)
#
#     # 计算总DN值（添加极小值避免除零）
#     total = r + g + b + 1e-8
#
#     # 归一化处理
#     r_norm = r / total
#     g_norm = g / total
#     b_norm = b / total
#
#     return r_norm, g_norm, b_norm
#
#
# def calculate_vis(r_norm, g_norm, b_norm):
#     """
#     计算10种植被指数
#     :return: 包含各指数的字典
#     """
#     # 初始化结果字典
#     vis = {}
#
#     # 1. Excess Green Index (EXG)
#     vis['EXG'] = 2 * g_norm - r_norm - b_norm
#
#     # 2. Excess Blue Index (EXB)
#     vis['EXB'] = b_norm - r_norm
#
#     # 3. Green Leaf Index (GLI)
#     denominator_GLI = 2 * g_norm + r_norm + b_norm
#     vis['GLI'] = (2 * g_norm - r_norm - b_norm) / (denominator_GLI + 1e-8)
#
#     # 4. Visible Atmospherically Resistant Index (VARI)
#     denominator_VARI = g_norm + r_norm - b_norm
#     vis['VARI'] = (g_norm - r_norm) / (denominator_VARI + 1e-8)
#
#     # 5. Excess Green minus Excess Red (EXGR)
#     vis['EXGR'] = g_norm - r_norm
#
#     # 6. Red-Green-Blue Vegetation Index (RGBVI)
#     denominator_RGBVI = g_norm + b_norm
#     vis['RGBVI'] = (g_norm - b_norm) / (denominator_RGBVI + 1e-8)
#
#     # 7. Modified Green Red Vegetation Index (MGRVI)
#     denominator_MGRVI = g_norm + r_norm
#     vis['MGRVI'] = (g_norm - r_norm) / (denominator_MGRVI + 1e-8)
#
#     # 8. Normalized Green-Red Difference Index (NGRDI)
#     denominator_NGRDI = g_norm + b_norm
#     vis['NGRDI'] = (g_norm - b_norm) / (denominator_NGRDI + 1e-8)
#
#     # 9. Green-Red Ratio Index (GRRI)
#     vis['GRRI'] = g_norm / (r_norm + 1e-8)
#
#     # 10. Normalized Difference Index (NDI)
#     denominator_NDI = g_norm + r_norm
#     vis['NDI'] = (g_norm - r_norm) / (denominator_NDI + 1e-8)
#
#     return vis
#
#
# def process_image(image_path):
#     """
#     主处理函数
#     :param image_path: 图像路径
#     :return: 包含统计结果的DataFrame
#     """
#     # 读取图像
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"图像文件不存在: {image_path}")
#
#     img = io.imread(image_path)
#
#     # 归一化处理
#     r_norm, g_norm, b_norm = normalize_dn_values(img)
#
#     # 计算植被指数
#     vis_dict = calculate_vis(r_norm, g_norm, b_norm)
#
#     # 计算统计特征
#     stats = {}
#     for name, values in vis_dict.items():
#         stats[f"{name}_mean"] = np.nanmean(values)
#         stats[f"{name}_std"] = np.nanstd(values)
#         stats[f"{name}_median"] = np.nanmedian(values)
#
#     # 转换为DataFrame
#     df = pd.DataFrame([stats])
#
#     return df
#
#
# # 修改最后的主程序部分
# if __name__ == "__main__":
#     # 示例使用
#     image_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'DJI_20240727121618_0030_07271.JPG')
#
#     try:
#         result_df = process_image(image_path)
#         print("植被指数统计特征：")
#         print(result_df.to_string(index=False))
#
#         # 保存结果到Excel
#         excel_path = 'vegetation_indices.xlsx'
#         result_df.to_excel(excel_path, index=False, engine='openpyxl')
#         print(f"结果已保存到 {excel_path}")
#
#     except Exception as e:
#         print(f"处理失败: {str(e)}")

# if __name__ == "__main__":
#     # 示例使用
#     image_path = os.path.join(os.path.expanduser('~'), 'Desktop', '2image.jpg')
#
#     try:
#         result_df = process_image(image_path)
#         print("植被指数统计特征：")
#         print(result_df.to_string(index=False))
#
#         # 可选：保存结果到CSV
#         result_df.to_csv('vegetation_indices.csv', index=False)
#         print("结果已保存到 vegetation_indices.csv")
#
#     except Exception as e:
#         print(f"处理失败: {str(e)}")

import numpy as np
from skimage import io
import os
import pandas as pd


def normalize_dn_values(img):
    """
    对RGB通道进行归一化处理
    :param img: 原始图像数组 (H, W, 3)
    :return: 归一化的R、G、B通道
    """
    # 分离RGB通道
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)

    # 计算总DN值（添加极小值避免除零）
    total = r + g + b + 1e-8

    # 归一化处理
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total

    return r_norm, g_norm, b_norm


def calculate_vis(r_norm, g_norm, b_norm):
    """
    计算10种植被指数
    :return: 包含各指数的字典
    """
    # 初始化结果字典
    vis = {}

    # 1. Excess Green Index (EXG)
    vis['EXG'] = 2 * g_norm - r_norm - b_norm

    # 2. Excess Blue Index (EXB)
    vis['EXB'] = 1.4 * b_norm - g_norm

    # 3. Green Leaf Index (GLI)
    denominator_GLI = 2 * g_norm + r_norm + b_norm
    vis['GLI'] = (2 * g_norm - r_norm - b_norm) / (denominator_GLI + 1e-8)

    # 4. Visible Atmospherically Resistant Index (VARI)
    denominator_VARI = g_norm + r_norm - b_norm
    vis['VARI'] = (g_norm - r_norm) / (denominator_VARI + 1e-8)

    # 5. Excess Green minus Excess Red (EXGR)
    vis['EXGR'] = 3 * g_norm - 2.4 * r_norm -b_norm

    # 6. Red-Green-Blue Vegetation Index (RGBVI)
    denominator_RGBVI = g_norm * g_norm + b_norm * r_norm
    vis['RGBVI'] = (g_norm * g_norm - b_norm * r_norm) / (denominator_RGBVI + 1e-8)

    # 7. Modified Green Red Vegetation Index (MGRVI)
    denominator_MGRVI = g_norm * g_norm + r_norm * r_norm
    vis['MGRVI'] = (g_norm * g_norm - r_norm * r_norm) / (denominator_MGRVI + 1e-8)

    # 8. Normalized Green-Red Difference Index (NGRDI)
    denominator_NGRDI = g_norm + r_norm
    vis['NGRDI'] = (g_norm - r_norm) / (denominator_NGRDI + 1e-8)

    # 9. Green-Red Ratio Index (GRRI)
    vis['GRRI'] = r_norm / (g_norm + 1e-8)

    # 10. Normalized Difference Index (NDI)
    denominator_NDI = g_norm + r_norm
    vis['NDI'] = (r_norm - g_norm) / (denominator_NDI + 1e-8)

    return vis


def process_image(image_path):
    """
    主处理函数
    :param image_path: 图像路径
    :return: 包含统计结果的DataFrame
    """
    # 读取图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    img = io.imread(image_path)

    # 归一化处理
    r_norm, g_norm, b_norm = normalize_dn_values(img)

    # 计算植被指数
    vis_dict = calculate_vis(r_norm, g_norm, b_norm)

    # 计算统计特征
    stats = {}
    for name, values in vis_dict.items():
        stats[f"{name}_mean"] = np.nanmean(values)
        stats[f"{name}_std"] = np.nanstd(values)
        stats[f"{name}_median"] = np.nanmedian(values)

    # 转换为DataFrame
    df = pd.DataFrame([stats])

    return df


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
            result_df = process_image(image_path)
            result_df['Image'] = jpg_file  # 添加图片文件名作为标识
            all_results.append(result_df)
        except Exception as e:
            print(f"处理 {jpg_file} 失败: {str(e)}")

    # 合并所有结果
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # 保存到Excel
        final_df.to_excel(output_excel, index=False, engine='openpyxl')
        print(f"所有结果已保存到 {output_excel}")
    else:
        print("没有成功处理任何图片。")


# 修改最后的主程序部分
if __name__ == "__main__":
    # 示例文件夹路径和输出文件
    folder_path = 'E:\\70M纵向\\images'  # 请替换为您的文件夹路径
    output_excel = 'E:\\70M纵向\\images\\vegetation_indices_batch_results.xlsx'

    process_directory(folder_path, output_excel)
