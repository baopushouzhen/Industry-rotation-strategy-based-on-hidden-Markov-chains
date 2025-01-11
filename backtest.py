import os
import pandas as pd
import xlsxwriter
import os

import os


def get_first_level_subfolders(directory):
    subfolders = []  # 创建一个空列表来存储子文件夹路径

    # 遍历指定目录的第一层
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        # 检查路径是否是一个目录
        if os.path.isdir(entry_path):
            subfolders.append(entry_path)

    return subfolders



# directory = 'D:\\pythonProject6\\f_5_result\\f_5_result'
directory = r"D:\pythonProject6\f_5_change22_result_right"
file_list = get_first_level_subfolders(directory)
print(file_list)




# # 使用示例
# directory_path = r"D:\pythonProject6\f_4_result"
# file_list = get_files_in_first_level_folders(directory_path)

# # 打印文件路径列表（可选）
# print(file_list)
print(1)

for i in file_list:
    base_path = rf'{i}'
    last_part = base_path.split('\\')[-1]
    # file_path_result = rf'D:\pythonProject6\f_5_all_result\{last_part}_all_result.xlsx'
    file_path_result = rf'D:\pythonProject6\f_5_change22_result_right\{last_part}_all_result_right.xlsx'
    """
    ic循环
    """
    # 指定文件夹路径和输出文件路径
    for i in [1,2,3,4,5]:
        # folder_path = rf"D:\pythonProject6\19_42_new\19_42_new\19_{i}\ic"
        folder_path = rf"{base_path}\{i}\ic"
        # output_file_path = r"D:\pythonProject6\19_42\19_42\ic\1.xlsx"

        # 创建一个空列表来存储所有 DataFrame
        dfs = []

        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)

                # 读取 CSV 文件
                df = pd.read_csv(file_path)
                print(df.columns)
                df.set_index('CI005004.WI_y_m', inplace=True)
                df.index.name = 'Date'
                # 过滤包含 'return' 或 'benchmark' 的列名
                # filtered_columns = [col for col in df.columns if 'return' in col or 'benchmark' in col]
                # df.drop(columns = filtered_columns)

                # 取出不带后缀的文件名
                file_name_without_extension = os.path.splitext(filename)[0]

                # 取倒数第四个字符及之前的部分
                if len(file_name_without_extension) >= 4:
                    suffix = file_name_without_extension[:-3]  # 截取倒数第四个字符之前的部分
                else:
                    suffix = file_name_without_extension  # 文件名长度小于4时，取整个文件名
                print(suffix)
                # 为每列名添加后缀
                df.columns = [f'{col}_{suffix}' for col in df.columns]
                df.columns = [col.replace('_0', '_00') for col in df.columns]
                df.columns = [col.replace('_1', '_01') for col in df.columns]
                df.columns = [col.replace('_2', '_02') for col in df.columns]
                df.columns = [col.replace('_3', '_03') for col in df.columns]
                df.columns = [col.replace('_4', '_04') for col in df.columns]
                df.columns = [col.replace('_5', '_05') for col in df.columns]
                df.columns = [col.replace('_6', '_06') for col in df.columns]
                df.columns = [col.replace('_7', '_07') for col in df.columns]
                df.columns = [col.replace('_8', '_08') for col in df.columns]
                df.columns = [col.replace('_9', '_09') for col in df.columns]
                df.columns = [col.replace('_010', '_10') for col in df.columns]
                df.columns = [col.replace('_011', '_11') for col in df.columns]
                df.columns = [col.replace('_012', '_12') for col in df.columns]
                df.columns = [col.replace('_013', '_13') for col in df.columns]
                df.columns = [col.replace('_014', '_14') for col in df.columns]
                # 将处理后的 DataFrame 添加到列表中
                dfs.append(df)

                print(f'Processed file: {filename}')

        # 合并所有 DataFrame
        combined_df = pd.concat(dfs, axis=1, ignore_index=False)
        # print(1)
        # with pd.ExcelWriter('all_ceshi.xlsx', engine='openpyxl',mode='a') as writer:
        #     combined_df.to_excel(writer, sheet_name=f'ic_{i}', index=True)
        # print(1)
        # 定义 Excel 文件路径
        # file_path = f'{last_part}_result.xlsx'

        # 如果文件不存在，则创建一个新的 Excel 文件
        if not os.path.exists(file_path_result):
            with pd.ExcelWriter(file_path_result, engine='openpyxl') as writer:
                # 创建一个空的工作表，避免后续写入时出错
                pd.DataFrame().to_excel(writer, sheet_name='empty', index=False)

        # 获取当前的工作簿并添加新的工作表
        with pd.ExcelWriter(file_path_result, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:

                combined_df.to_excel(writer, sheet_name=f'ic_{i}', index=True)

        print("DataFrames have been successfully merged and written to Excel.")

    """
    return
    """
    # 取列名
    cum_return_df = pd.DataFrame()
    # folder_path = rf"D:\pythonProject6\19_42_new\19_42_new\19_1\return"
    folder_path = rf"{base_path}\1\return"

    # 创建一个空列表来存储所有 DataFrame
    dfs = []
    suffix_list = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            print(df.columns)
            df.set_index('Unnamed: 0', inplace=True)
            df.index.name = 'Date'
            # 过滤包含 'return' 或 'benchmark' 的列名
            # filtered_columns = [col for col in df.columns if 'return' in col or 'benchmark' in col]
            # df.drop(columns = filtered_columns)

            # 取出不带后缀的文件名
            file_name_without_extension = os.path.splitext(filename)[0]

            # 取倒数第四个字符及之前的部分
            if len(file_name_without_extension) >= 4:
                suffix = file_name_without_extension[:-7]  # 截取倒数第四个字符之前的部分
            else:
                suffix = file_name_without_extension  # 文件名长度小于4时，取整个文件名
            print(suffix)
            suffix_list.append(suffix)
            # 为每列名添加后缀
            df.columns = [f'{col}_{suffix}' for col in df.columns]
            df.columns = [col.replace('_0', '_00') for col in df.columns]
            df.columns = [col.replace('_1', '_01') for col in df.columns]
            df.columns = [col.replace('_2', '_02') for col in df.columns]
            df.columns = [col.replace('_3', '_03') for col in df.columns]
            df.columns = [col.replace('_4', '_04') for col in df.columns]
            df.columns = [col.replace('_5', '_05') for col in df.columns]
            df.columns = [col.replace('_6', '_06') for col in df.columns]
            df.columns = [col.replace('_7', '_07') for col in df.columns]
            df.columns = [col.replace('_8', '_08') for col in df.columns]
            df.columns = [col.replace('_9', '_09') for col in df.columns]
            df.columns = [col.replace('_010', '_10') for col in df.columns]
            df.columns = [col.replace('_011', '_11') for col in df.columns]
            df.columns = [col.replace('_012', '_12') for col in df.columns]
            df.columns = [col.replace('_013', '_13') for col in df.columns]
            df.columns = [col.replace('_014', '_14') for col in df.columns]


            dfs.append(df)

            print(f'Processed file: {filename}')

    # 合并所有 DataFrame
    combined_df = pd.concat(dfs, axis=1, ignore_index=False)
    print(suffix_list)
    cum_return_df = pd.DataFrame(index=suffix_list)

    # 遍历return里面的4个组，如果
    for i in [1,2,3,4,5]:

        # folder_path = rf"{base_path}\19_{i}\return"
        # output_file_path = rf"{base_path}\return\{i}.csv"
        folder_path = rf"{base_path}\{i}\return"
        output_file_path = rf"{base_path}\return\{i}.csv"
        # 创建一个空列表来存储所有 DataFrame
        dfs = []
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)

                # 读取 CSV 文件
                df = pd.read_csv(file_path)
                df.set_index('Unnamed: 0', inplace=True)
                df.index.name = 'Date'
                # 过滤包含 'return' 或 'benchmark' 的列名
                # filtered_columns = [col for col in df.columns if 'return' in col or 'benchmark' in col]
                # df.drop(columns = filtered_columns)

                # 取出不带后缀的文件名
                file_name_without_extension = os.path.splitext(filename)[0]

                # 取倒数第四个字符及之前的部分
                if len(file_name_without_extension) >= 4:
                    suffix = file_name_without_extension[:-7]  # 截取倒数第四个字符之前的部分
                else:
                    suffix = file_name_without_extension  # 文件名长度小于4时，取整个文件名
                x =((df['ex_return']+1).cumprod()-1).iloc[-1]
                print(suffix)  # 确认 suffix 是有效的行索引
                print(i)  # 确认 i 是有效的列标签
                print(x)  # 确认 x 的索引与 DataFrame 的行索引匹配
                cum_return_df.loc[suffix,i] = x
                # 为每列名添加后缀
                df.columns = [f'{col}_{suffix}' for col in df.columns]
                df.columns = [col.replace('_0', '_00') for col in df.columns]
                df.columns = [col.replace('_1', '_01') for col in df.columns]
                df.columns = [col.replace('_2', '_02') for col in df.columns]
                df.columns = [col.replace('_3', '_03') for col in df.columns]
                df.columns = [col.replace('_4', '_04') for col in df.columns]
                df.columns = [col.replace('_5', '_05') for col in df.columns]
                df.columns = [col.replace('_6', '_06') for col in df.columns]
                df.columns = [col.replace('_7', '_07') for col in df.columns]
                df.columns = [col.replace('_8', '_08') for col in df.columns]
                df.columns = [col.replace('_9', '_09') for col in df.columns]
                df.columns = [col.replace('_010', '_10') for col in df.columns]
                df.columns = [col.replace('_011', '_11') for col in df.columns]
                df.columns = [col.replace('_012', '_12') for col in df.columns]
                df.columns = [col.replace('_013', '_13') for col in df.columns]
                df.columns = [col.replace('_014', '_14') for col in df.columns]

                dfs.append(df)

                print(f'Processed file: {filename}')

        # 合并所有 DataFrame
        combined_df = pd.concat(dfs, axis=1, ignore_index=False)
        # 确保目录存在，如果不存在则创建
        directory = os.path.dirname(output_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 将 DataFrame 保存到 CSV 文件
        combined_df.to_csv(output_file_path)
        # combined_df.to_csv(output_file_path)


    # 假设 cum_return_df 已经定义并且包含索引
    replace_dict = {
        '_0': '_00',
        '_1': '_01',
        '_2': '_02',
        '_3': '_03',
        '_4': '_04',
        '_5': '_05',
        '_6': '_06',
        '_7': '_07',
        '_8': '_08',
        '_9': '_09',
        '_010': '_10',
        '_011': '_11',
        '_012': '_12',
        '_013': '_13',
        '_014': '_14'
    }

    # 更新索引
    cum_return_df.index = [index for index in cum_return_df.index]
    for old, new in replace_dict.items():
        cum_return_df.index = cum_return_df.index.str.replace(old, new)
    cum_return_df.sort_index()
    # 获取当前的工作簿并添加新的工作表
    with pd.ExcelWriter(file_path_result, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        cum_return_df.to_excel(writer, sheet_name='cum_return', index=True)
    print("DataFrames have been successfully merged and written to Excel.")




    """
    合并return的文件,制作daily_return
    """
    import os
    import pandas as pd

    # 文件夹路径
    folder_path = rf"{base_path}\return"

    # 存储所有 DataFrame 的列表
    dataframes = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # 假设文件是 CSV 格式
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            # 获取文件名的第一个字符
            prefix = filename[0]
            df.columns = [col.replace('_0', '_00') for col in df.columns]
            df.columns = [col.replace('_1', '_01') for col in df.columns]
            df.columns = [col.replace('_2', '_02') for col in df.columns]
            df.columns = [col.replace('_3', '_03') for col in df.columns]
            df.columns = [col.replace('_4', '_04') for col in df.columns]
            df.columns = [col.replace('_5', '_05') for col in df.columns]
            df.columns = [col.replace('_6', '_06') for col in df.columns]
            df.columns = [col.replace('_7', '_07') for col in df.columns]
            df.columns = [col.replace('_8', '_08') for col in df.columns]
            df.columns = [col.replace('_9', '_09') for col in df.columns]
            df.columns = [col.replace('_000', '_00') for col in df.columns]
            df.columns = [col.replace('_010', '_10') for col in df.columns]
            df.columns = [col.replace('_011', '_11') for col in df.columns]
            df.columns = [col.replace('_012', '_12') for col in df.columns]
            df.columns = [col.replace('_013', '_13') for col in df.columns]
            df.columns = [col.replace('_014', '_14') for col in df.columns]
            # 修改列名，将前缀加到列名上
            df.columns = [f'{prefix}_{col}' for col in df.columns]

            # 将处理后的 DataFrame 添加到列表中
            dataframes.append(df)

    # 合并所有 DataFrame

    combined_df = pd.concat(dataframes, axis=1)
    keep_columns = [col for col in combined_df.columns if 'ex_return' in col ]
    combined_df = combined_df[keep_columns]
    sorted_columns = sorted(combined_df.columns, key=lambda col: col[-4:])
    combined_df = combined_df[sorted_columns]

    with pd.ExcelWriter(file_path_result, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        combined_df.to_excel(writer, sheet_name='daily_return', index=True)
