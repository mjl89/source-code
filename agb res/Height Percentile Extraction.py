# import rasterio
# import numpy as np
# import os
#
# def calculate_percentiles(tif_file, percentiles=[1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]):
#     with rasterio.open(tif_file) as src:
#         data = src.read(1)  # 读取tif文件的第一波段
#         data = data.flatten()  # 将数据展平为一维数组，以便计算整个数据的百分位数
#
#     percentiles_data = {}
#     for percentile in percentiles:
#         percentile_value = np.percentile(data, percentile)
#         percentiles_data[f'{percentile}th_percentile'] = percentile_value
#
#     return percentiles_data
#
# def main():
#     folder_path = 'D:\\DJL1\DJITerra\\15373363338\\70M纵向\\lidars\\terra_las\\70M纵向去噪_归一化\\70M纵向去噪_归一化_高度变量_elev_AIH_90th.tif'  # 替换为你的tif文件所在文件夹路径
#     files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
#
#     for file in files:
#         file_path = os.path.join(folder_path, file)
#         percentiles = calculate_percentiles(file_path)
#         print(f"File: {file}")
#         for key, value in percentiles.items():
#             print(f"{key}: {value}")
#
# if __name__ == "__main__":
#     main()
#
# import rasterio
# import numpy as np
# import os
#
#
# def calculate_percentiles(tif_file, percentiles=[1, 20, 40, 50, 60, 80, 90, 99]):
#     with rasterio.open(tif_file) as src:
#         data = src.read(1)
#         data = data[data != src.nodata].flatten()  # 排除无效值:ml-citation{ref="3,7" data="citationList"}
#
#     return {f'{p}th_percentile': np.percentile(data, p) for p in percentiles}
#
#
# def main():
#     # 使用原始字符串或正斜杠路径
#     folder_path = r'D:\DJL1\DJITerra\15373363338\70M纵向\lidars\terra_las\70M纵向去噪_归一化\分割10'
#     if not os.path.isdir(folder_path):
#         raise ValueError(f"路径不存在或不是目录: {folder_path}")
#
#     for file in [f for f in os.listdir(folder_path) if f.endswith('.tif')]:
#         file_path = os.path.join(folder_path, file)
#         try:
#             results = calculate_percentiles(file_path)
#             print(f"\nFile: {file}")
#             for k, v in results.items():
#                 print(f"{k}: {v:.4f}")  # 保留4位小数
#         except Exception as e:
#             print(f"处理{file}时出错: {str(e)}")
#
#
# if __name__ == "__main__":
#     main()
# 以上代码输出结果不为表格，以下代码结果输出为表格
# import rasterio
# import numpy as np
# import os
# from tabulate import tabulate
#
#
# def calculate_percentiles(tif_file, percentiles=[1, 20, 40, 50, 60, 80, 90, 99]):
#     """计算TIFF文件的指定百分位数"""
#     with rasterio.open(tif_file) as src:
#         data = src.read(1)
#         # 排除无效值
#         data = data[data != src.nodata].flatten()
#
#     # 如果所有值都是无效值，返回空字典
#     if len(data) == 0:
#         return {}
#
#     return {f'{p}th_percentile': np.percentile(data, p) for p in percentiles}
#
#
# def main():
#     # 使用原始字符串或正斜杠路径
#     folder_path = r'D:\DJL1\DJITerra\15373363338\70M纵向\lidars\terra_las\70M纵向去噪_归一化\分割10'
#
#     if not os.path.isdir(folder_path):
#         raise ValueError(f"路径不存在或不是目录: {folder_path}")
#
#     # 获取所有TIFF文件
#     tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
#
#     if not tif_files:
#         print("指定文件夹中没有找到TIFF文件")
#         return
#
#     # 准备表格数据
#     table_data = []
#     # 表格头部：文件名 + 百分位数
#     percentiles = [1, 20, 40, 50, 60, 80, 90, 99]
#     headers = ["文件名"] + [f"{p}th百分位数" for p in percentiles]
#
#     # 处理每个文件
#     for file in tif_files:
#         file_path = os.path.join(folder_path, file)
#         try:
#             results = calculate_percentiles(file_path, percentiles)
#
#             if not results:  # 处理空数据情况
#                 row = [file] + ["无有效数据"] * len(percentiles)
#             else:
#                 row = [file] + [f"{results[f'{p}th_percentile']:.4f}" for p in percentiles]
#
#             table_data.append(row)
#
#         except Exception as e:
#             # 错误处理行
#             row = [file] + [f"错误: {str(e)}" for _ in percentiles]
#             table_data.append(row)
#
#     # 打印表格
#     print("\nTIFF文件百分位数统计结果：")
#     print(tabulate(table_data, headers=headers, tablefmt="grid"))
#
#
# if __name__ == "__main__":
#     main()
#
#
# import rasterio
# import numpy as np
# import os
#
#
# def calculate_percentiles(tif_file, percentiles=[1, 20, 40, 50, 60, 80, 90, 99]):
#     """计算TIFF文件的指定百分位数"""
#     with rasterio.open(tif_file) as src:
#         data = src.read(1)
#         # 排除无效值
#         data = data[data != src.nodata].flatten()
#
#     # 如果所有值都是无效值，返回空字典
#     if len(data) == 0:
#         return {}
#
#     return {f'{p}th_percentile': np.percentile(data, p) for p in percentiles}
#
#
# def print_table(data, headers):
#     """使用字符串格式化打印表格"""
#     # 计算每列的最大宽度
#     col_widths = [len(header) for header in headers]
#
#     for row in data:
#         for i, item in enumerate(row):
#             if len(str(item)) > col_widths[i]:
#                 col_widths[i] = len(str(item))
#
#     # 打印表头
#     header_row = "|".join(f" {header:{col_widths[i]}} " for i, header in enumerate(headers))
#     separator = "+".join("-" * (col_widths[i] + 2) for i in range(len(headers)))
#
#     print(separator)
#     print(header_row)
#     print(separator)
#
#     # 打印数据行
#     for row in data:
#         data_row = "|".join(f" {str(item):{col_widths[i]}} " for i, item in enumerate(row))
#         print(data_row)
#         print(separator)
#
#
# def main():
#     # 使用原始字符串或正斜杠路径
#     folder_path = r'D:\DJL1\DJITerra\15373363338\70M纵向\lidars\terra_las\70M纵向去噪_归一化\分割10'
#
#     if not os.path.isdir(folder_path):
#         raise ValueError(f"路径不存在或不是目录: {folder_path}")
#
#     # 获取所有TIFF文件
#     tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
#
#     if not tif_files:
#         print("指定文件夹中没有找到TIFF文件")
#         return
#
#     # 准备表格数据
#     table_data = []
#     # 表格头部：文件名 + 百分位数
#     percentiles = [1, 20, 40, 50, 60, 80, 90, 99]
#     headers = ["文件名"] + [f"{p}th百分位数" for p in percentiles]
#
#     # 处理每个文件
#     for file in tif_files:
#         file_path = os.path.join(folder_path, file)
#         try:
#             results = calculate_percentiles(file_path, percentiles)
#
#             if not results:  # 处理空数据情况
#                 row = [file] + ["无有效数据"] * len(percentiles)
#             else:
#                 row = [file] + [f"{results[f'{p}th_percentile']:.4f}" for p in percentiles]
#
#             table_data.append(row)
#
#         except Exception as e:
#             # 错误处理行
#             row = [file] + [f"错误: {str(e)}" for _ in percentiles]
#             table_data.append(row)
#
#     # 打印表格
#     print("\nTIFF文件百分位数统计结果：")
#     print_table(table_data, headers)
#
#
# if __name__ == "__main__":
#     main()


import rasterio
import numpy as np
import os
import csv  # 使用Python内置的csv模块


def calculate_percentiles(tif_file, percentiles=[1, 20, 40, 50, 60, 80, 90, 99]):
    """计算TIFF文件的指定百分位数"""
    with rasterio.open(tif_file) as src:
        data = src.read(1)
        # 排除无效值
        data = data[data != src.nodata].flatten()

    # 如果所有值都是无效值，返回空字典
    if len(data) == 0:
        return {}

    return {f'{p}th_percentile': np.percentile(data, p) for p in percentiles}


def main():
    # 输入文件夹路径
    folder_path = r'D:\DJL1\DJITerra\15373363338\70M纵向\lidars\terra_las\70M纵向去噪_归一化\分割99'
    # 输出CSV文件路径（可自定义修改）
    output_csv = os.path.join(folder_path, "百分位数统计结果.csv")

    if not os.path.isdir(folder_path):
        raise ValueError(f"路径不存在或不是目录: {folder_path}")

    # 获取所有TIFF文件
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    if not tif_files:
        print("指定文件夹中没有找到TIFF文件")
        return

    # 定义百分位数列表
    percentiles = [1, 20, 40, 50, 60, 80, 90, 99]
    # 表格头部
    headers = ["文件名"] + [f"{p}th百分位数" for p in percentiles]

    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(headers)

        # 处理每个文件并写入数据
        for file in tif_files:
            file_path = os.path.join(folder_path, file)
            try:
                results = calculate_percentiles(file_path, percentiles)

                if not results:  # 处理空数据情况
                    row = [file] + ["无有效数据"] * len(percentiles)
                else:
                    row = [file] + [round(results[f'{p}th_percentile'], 4) for p in percentiles]

                writer.writerow(row)
                print(f"已处理: {file}")

            except Exception as e:
                # 错误处理行
                row = [file] + [f"错误: {str(e)}" for _ in percentiles]
                writer.writerow(row)
                print(f"处理出错: {file} - {str(e)}")

    print(f"\n所有结果已保存至: {output_csv}")


if __name__ == "__main__":
    main()