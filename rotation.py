import pandas as pd
import numpy as np
import os
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
"""
读取数据
"""
open_price = r'C:\Users\cuixinpu\Desktop\rotation_data\ZXOpen.csv'
close_price = r'C:\Users\cuixinpu\Desktop\rotation_data\ZXPrice.csv'
turn_over = r'C:\Users\cuixinpu\Desktop\rotation_data\ZXTurnOver.csv'
open_price = pd.read_csv(open_price, encoding='gbk')
close_price = pd.read_csv(close_price,encoding='gbk')
turn_over = pd.read_csv(turn_over,encoding='gbk')
print(open_price)
print(close_price)
print(turn_over)


"""
合并前处理数据
"""
print(turn_over.columns)
# print(turn_over.iloc[0])

df_t = turn_over.drop(index=0)
df_o = open_price.drop(index=0)
df_c = close_price.drop(index=0)

print(df_c.columns)
df_c.set_index('Date',inplace= True)
df_o.set_index('Date',inplace= True)
df_t.set_index('Date',inplace= True)
print(df_c)

# 获取三个 DataFrame 中共有的列名
common_columns = df_c.columns.intersection(df_o.columns).intersection(df_t.columns)

# 为每一个共有的列名生成一个 DataFrame，并存储在字典中
dfs = {}

for column in common_columns:
    dfs[column] = pd.DataFrame({
        'close': df_c[column],
        'open': df_o[column],
        'turn_over': df_t[column]
    })

# 修改Date列格式，并将其设置为索引
for column, df in dfs.items():
    df.index = pd.to_datetime(df.index.astype(int).astype(str), format='%Y%m%d')


"""
生成剩下特征
"""
for column, df in dfs.items():
    df['close'] = df['close'].astype(float)
    df['turn_over'] = df['turn_over'].str.rstrip('%').astype('float')
    df['turn_over_diff'] = df['turn_over'].diff()
    df['close_return'] = df['close'].pct_change()
    df['20d_volatility'] = df['close_return'].rolling(window=20).std() * np.sqrt(252)
    df['20d_volatility_diff'] = df['20d_volatility'].diff()
#
# #相关系数矩阵，判断马尔可夫链使用什么
# # # 设置显示的最大行数
# pd.set_option('display.max_rows', 100)
# # #
# # 设置显示的最大列数
# pd.set_option('display.max_columns', 20)
#
# # 设置显示的最大宽度（字符数）
# pd.set_option('display.max_colwidth', 100)
# #打印看一下
# 设置最大行数为 10
pd.set_option('display.max_rows', None)
# 设置最大列数为 10
pd.set_option('display.max_columns', 30)
for key, df in dfs.items():
    # 列名列表
    columns_to_plot = [
        'close',
        'turn_over',
        'turn_over_diff',
        'close_return',
        '20d_volatility',
        '20d_volatility_diff'
    ]
    save_folder = r"C:\Users\cuixinpu\Desktop\picture"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 遍历要绘制的每一列
    for i, column in enumerate(columns_to_plot):
        plt.subplot(2, 3, i + 1)  # 创建 2x3 的子图布局
        sns.kdeplot(df[column].dropna(), fill=True)  # 使用 dropna() 以避免 NaN 值影响绘图
        plt.title(f'Probability Density of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')

    plt.tight_layout()
    plt.show()
    save_path = os.path.join(save_folder, f'{key}.png')
    plt.savefig(save_path)
    c = df.corr()
    print(f"\nDataFrame for column: {key}")
    print(c)
# # print(dfs)
# # 设置最大行数为 10
# pd.set_option('display.max_rows', None)
# # 设置最大列数为 10
# pd.set_option('display.max_columns', 30)
x = dfs['CI005030.WI']
print((x['close']==0).sum())
first_non_zero = x.loc[x['close'] != 0, 'close'].iloc[0]
x['close'] = x['close'].replace(0, first_non_zero)
x['close'] = x['close'].replace(0, first_non_zero)
"""
确定隐藏状态个数
"""
#注释部分
#
# def evaluate_hmm_models(data, n_states_range):
#
#     """
#     评估不同隐状态数量下的 HMM 模型，并计算 AIC 和 BIC 值。
#     输出 AIC 和 BIC 最小的隐状态数量。
#
#     Parameters:
#     data (np.ndarray): 输入数据。
#     n_states_range (range): 隐状态数量的范围。
#
#     Returns:
#     pd.DataFrame: 包含每个隐状态数量对应的 AIC 和 BIC 值的 DataFrame。
#     """
#     results = {'n_states': [], 'AIC': [], 'BIC': []}
#
#     for n_states in n_states_range:
#         # 定义 HMM 模型
#         model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=1000)
#
#         # 训练模型
#         model.fit(data)
#
#         # 获取对数似然
#         log_likelihood = model.score(data)
#
#         # 计算参数数量
#         n_params = n_states * (n_states - 1) + 2 * n_states * data.shape[1]
#
#         # 计算 AIC 和 BIC
#         aic = -2 * log_likelihood + 2 * n_params
#         bic = -2 * log_likelihood + np.log(len(data)) * n_params
#
#         results['n_states'].append(n_states)
#         results['AIC'].append(aic)
#         results['BIC'].append(bic)
#
#     results_df = pd.DataFrame(results)
#
#     # 找到 AIC 和 BIC 最小的隐状态数量
#     best_aic_index = results_df['AIC'].idxmin()
#     best_bic_index = results_df['BIC'].idxmin()
#
#     best_aic_states = results_df.loc[best_aic_index, 'n_states']
#     best_bic_states = results_df.loc[best_bic_index, 'n_states']
#
#     best_aic_value = results_df.loc[best_aic_index, 'AIC']
#     best_bic_value = results_df.loc[best_bic_index, 'BIC']
#
#     # 打印结果
#     print("Results DataFrame:")
#     print(results_df)
#     print("\nBest AIC:")
#     print(f"Number of states: {best_aic_states}, AIC: {best_aic_value}")
#     print("\nBest BIC:")
#     print(f"Number of states: {best_bic_states}, BIC: {best_bic_value}")
#
#     # # 绘制 AIC 和 BIC 曲线
#     # plt.figure(figsize=(12, 6))
#     # plt.plot(results_df['n_states'], results_df['AIC'], label='AIC', marker='o')
#     # plt.plot(results_df['n_states'], results_df['BIC'], label='BIC', marker='o')
#     # plt.xlabel('Number of States')
#     # plt.ylabel('Value')
#     # plt.title('AIC and BIC for Different Number of States')
#     # plt.legend()
#     # plt.grid(True)
#     # plt.show()
#     return results_df, best_aic_states, best_bic_states
#
# for column, df in dfs.items():
#     df = df.drop(columns='open')
#     df = df.dropna()
#     data = df.values
#     data = StandardScaler().fit_transform(data)
#     # 评估 HMM 模型
#     n_states_range = range(1, 15)  # 从 1 到 10 个隐状态
#     results_df, best_aic_states, best_bic_states = evaluate_hmm_models(data, n_states_range)

# #test
# df = dfs['CI005026.WI']
# df = df[df.index.year >= 2017]
# df = df.dropna()
#
# print(df)
#
# #假定隐状态数量是10
# n = 10
# # 定义并训练HMM模型
# # 这里我们假设有2个隐状态
# observations = df.drop(columns = 'open').values
# observations = StandardScaler().fit_transform(observations)
# model = hmm.GaussianHMM(n_components=n, covariance_type='full', n_iter=2000,min_covar=1e-3)
# model.fit(observations)
#
# # 使用模型生成状态序列
# hidden_states = model.predict(observations)
#
# # 将状态序列添加回DataFrame
# df['Hidden_State'] = hidden_states
# print(df)

"""
跳过隐藏状态的测试
"""


# 生成本月和下一个月的交易日数量
for column,df in dfs.items():
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['y_m'] = df['Date'].dt.to_period('M')
    monthly_counts = df['y_m'].value_counts().sort_index()
    monthly_counts_next = monthly_counts.shift(-1).fillna(23).astype(int) # fillna（）里面是7月份交易日
    df['cur_days'] = df['y_m'].map(monthly_counts)
    df['next_days'] = df['y_m'].map(monthly_counts_next)

print(1)

# 生成收益率的函数
# for column,df in dfs_test.items():
def cal_next_windows_return(next_days,df):
    """

    :param next_days: 下一个（需要预测）窗口的交易日数量
    :return: 下一个（需要预测）窗口的收益率，是否显著
    """
    df['next_window_last_close'] = df['close'].shift(-(next_days+1))
    df['next_window_last_return'] = (df['next_window_last_close'] - df['close'])/df['close']
    return df


def cal_pos(data,n,test_period,w_matrix,threshold  ):
    """

    :param data: 处理好的每个行业的dataframe
    :param n: Markov隐状态数量
    :param test_period: 样本外区间,ep:[2023-1,2024-2] 回测区间
    :param w_0: 计算距离的时候0偏移的权重
    :param w_1: 计算距离的时候1偏移的权重
    :param w__1: 计算距离的时候-1偏移的权重
    :param w_2: 计算距离的时候2偏移的权重
    :param w__2: 计算距离的时候-2偏移的权重
    :param up_threshold: 判定显著上涨的阈值
    :param down_threshold: 判定显著下跌的阈值，与上一个参数是相反数
    :return: 添加了显著上涨概率列的dataframe
    """


    df = data.dropna() # 不可少，输入不能有缺失值
    for index_l, threshold_l in enumerate(threshold):
        for index, row in enumerate(w_matrix):
            df.loc[:, f'{index}_{index_l}_pro'] = 0.0
    # df.loc[:, 'pro'] = 0.0
    df = df.copy()
    # df.loc[:, 'pro'] = df['pro'].astype(float)
    cols_to_drop = [col for col in df.columns if 'pro' in col]
    # 创建表示 2018 年 1 月的 Period 对象 #标记
    period = pd.Period('2018-01', freq='M')

    # 获取 YYYY-MM 格式的字符串
    timestamp_str = period.strftime('%Y-%m') #标记    # distance = {}
    for i in test_period:

        """
        标准化数据
        """

        train = df[(df['y_m'] > timestamp_str)& (df['y_m'] < i)] # 生成当前月的训练数据，不包含当前月

        print(train)
        train_corr =  train.drop(columns = ['open','Date','y_m','cur_days','next_days']+cols_to_drop)
        corr = train_corr.corr()
        print('zheshixiangguanxishujvzhen')
        print(corr)
        test = df[df['y_m'] == i] # 生成当前月的测试数据，即当前月
        print(test)
        train_v = train.drop(columns = ['open','Date','y_m','cur_days','next_days']+cols_to_drop).values # 输入的数据没有开盘价这一特征
        test_v = test.drop(columns = ['open','Date','y_m','cur_days','next_days']+cols_to_drop).values  # 输入的数据没有开盘价这一特征
        print('这是标准化训练数据')
        print(train_v)
        print('这是标准化测试数据')
        print(test_v)
        scaler = StandardScaler() # 训练数据标准化模型
        train_s = scaler.fit_transform(train_v) # 标准化训练数据
        print('这是训练数据')
        print(i)
        print(train_s)
        test_s = scaler.transform(test_v)



        """
        模型训练
        """
        model = hmm.GaussianHMM(n_components=n,covariance_type='diag',n_iter=1000,random_state = 42) # 用train训练马尔可夫模型
        # print(train_v)
        model.fit(train_s) #
        #使用模型生成状态序列
        hidden_state_train = model.predict(train_s)
        hidden_state_test = model.predict(test_s)

        """
        数据合并
        """
        df_return_1 = pd.DataFrame({'hidden_state':hidden_state_train,
                                         'close':train['close'] ,
        'turn_over':train['turn_over'],
        'turn_over_diff':train['turn_over_diff'] ,
        'close_return':train['close_return'],
        '20d_volatility':train['20d_volatility'] ,
        '20d_volatility_diff':train['20d_volatility_diff'],
        'open':train['open'],
        'Date':train['Date'],
        'y_m': train['y_m'],
        'next_days': train['next_days'],
        'cur_days': train['cur_days'],

        })
        print('train')
        print(df_return_1['hidden_state'])
        df_return_2 = pd.DataFrame({'hidden_state': hidden_state_test,
                                    'close': test['close'],
                                    'turn_over': test['turn_over'],
                                    'turn_over_diff': test['turn_over_diff'],
                                    'close_return':  test['close_return'],
                                    '20d_volatility':  test['20d_volatility'],
                                    '20d_volatility_diff':  test['20d_volatility_diff'],
                                    'open':  test['open'],
                                    'Date':  test['Date'],
                                    'y_m': test['y_m'],
                                    'next_days':  test['next_days'],
                                    'cur_days':  test['cur_days'],

                                    })
        print('test')
        print(df_return_2['hidden_state'])

        combined_df = pd.concat([df_return_1, df_return_2], ignore_index=True)
        print(combined_df)
        """
        计算数列匹配分数
        """

        combined_df['hidden_state_s1'] = combined_df['hidden_state'].shift(1)
        combined_df['hidden_state_s_1'] = combined_df['hidden_state'].shift(-1)
        combined_df['hidden_state_s2'] = combined_df['hidden_state'].shift(2)
        combined_df['hidden_state_s_2'] = combined_df['hidden_state'].shift(-2)
        long = len(train)  # + len(test)
        # 获取待匹配数列长度
        win = len(hidden_state_test)


        #生成长度是win。最后5个数字是1的序列
        ta = np.zeros(win)
        ta[-5:] = 1
        #生成长度是win,衰减的序列
        sequence = np.exp(-np.log(2) / 10 * np.arange(win))
        all_ones = np.ones(win)
        matrix = np.vstack((all_ones,ta, sequence))

        for idx , weight_list in enumerate(matrix):

            # 生成列记录匹配数量
            combined_df[f'{idx}_match_count_0'] = 0  # 完全匹配
            combined_df[f'{idx}_match_count_1'] = 0  # 差一位匹配
            combined_df[f'{idx}_match_count__1'] = 0  # 差一位匹配
            combined_df[f'{idx}_match_count_2'] = 0  # 差二位匹配
            combined_df[f'{idx}_match_count__2'] = 0  # 差二位匹配



            for j in range(win - 1, long):
                window = combined_df['hidden_state'].iloc[j - (win - 1):j + 1].values
                match_array = [hidden_state_test[k] == window[k] for k in range(win)]
                match_array = np.array(match_array, dtype=int)
                match_count = sum(match_array * weight_list)
                # match_count = sum(hidden_state_test[k] == window[k] for k in range(win))
                combined_df.iloc[j, combined_df.columns.get_loc(f'{idx}_match_count_0')] = match_count

            for j in range(win - 1, long):
                window = combined_df['hidden_state_s1'].iloc[j - (win - 1):j + 1].values
                match_array = [hidden_state_test[k] == window[k] for k in range(win)]
                match_array = np.array(match_array, dtype=int)
                match_count = sum(match_array * weight_list)
                # match_count = sum(hidden_state_test[k] == window[k] for k in range(win))
                combined_df.iloc[j, combined_df.columns.get_loc(f'{idx}_match_count_1')] = match_count

            for j in range(win - 1, long):
                window = combined_df['hidden_state_s_1'].iloc[j - (win - 1):j + 1].values
                match_array = [hidden_state_test[k] == window[k] for k in range(win)]
                match_array = np.array(match_array, dtype=int)
                match_count = sum(match_array * weight_list)
                # match_count = sum(hidden_state_test[k] == window[k] for k in range(win))
                combined_df.iloc[j, combined_df.columns.get_loc(f'{idx}_match_count__1')] = match_count

            for j in range(win - 1, long):
                window = combined_df['hidden_state_s2'].iloc[j - (win - 1):j + 1].values
                match_array = [hidden_state_test[k] == window[k] for k in range(win)]
                match_array = np.array(match_array, dtype=int)
                match_count = sum(match_array * weight_list)
                # match_count = sum(hidden_state_test[k] == window[k] for k in range(win))
                combined_df.iloc[j,combined_df.columns.get_loc(f'{idx}_match_count_2')] = match_count
            for j in range(win - 1, long):
                window = combined_df['hidden_state_s_2'].iloc[j - (win - 1):j + 1].values
                match_array = [hidden_state_test[k] == window[k] for k in range(win)]
                match_array = np.array(match_array, dtype=int)
                match_count = sum(match_array * weight_list)
                # match_count = sum(hidden_state_test[k] == window[k] for k in range(win))
                combined_df.iloc[j, combined_df.columns.get_loc(f'{idx}_match_count__2')] = match_count
            for index, row in enumerate(w_matrix):

                combined_df[f'{idx}_{index}_pro'] = combined_df[f'{idx}_match_count_0'] * row[0] \
                                     + combined_df[f'{idx}_match_count_1'] * row[1] \
                                     + combined_df[f'{idx}_match_count__1'] * row[2] \
                                     + combined_df[f'{idx}_match_count_2'] * row[3] \
                                     + combined_df[f'{idx}_match_count__2'] * row[4]

        """
        计算收益率
        """

        # 计算下一个月的交易天数
        def increment_month(date_str):
            # 将字符串转换为 Period 对象
            period = pd.Period(date_str, freq='M')
            # 增加一个月
            next_period = period + 1
            # 返回下一个月份的字符串形式
            return next_period.strftime('%Y-%m')

        next_month = increment_month(i)
        next = df[df['y_m'] == next_month]
        next_days = next.shape[0]

        # 循环计算某一交易日未来窗口的收益率
        combined_df = cal_next_windows_return(next_days, combined_df )
        combined_df = combined_df.head(df_return_1.shape[0])

        # 根据收益率判断上涨/下跌概率
        for idx,weight_list in enumerate(matrix):
            for index_l,threshold_l in enumerate(threshold):
                for index, row in enumerate(w_matrix):
                    up = combined_df[combined_df['next_window_last_return'] > threshold_l][f'{idx}_{index}_pro'].mean()
                    down = combined_df[combined_df['next_window_last_return'] < - threshold_l][f'{idx}_{index}_pro'].mean()
                    up_pro = up / (up + down)
                    i = pd.Period(i, freq='M')
                    df.loc[df['y_m'] == i, f'{idx}_{index}_{index_l}_pro'] = up_pro



        # # 显著上涨和下跌数量
        #
        # up_pro = up / (up + down)
        # i = pd.Period(i, freq='M')
        # df.loc[df['y_m'] == i, 'pro'] = up_pro
        # ic

    # print(test_period)
    # 将字符串列表转换为 DatetimeIndex
    datetime_index = pd.to_datetime(test_period, format='%Y-%m')

    # 将 DatetimeIndex 转换为 PeriodIndex（月份频率）
    test_period = datetime_index.to_period('M')
    df_f = df[df['y_m'].isin(test_period)]
    #
    #     """
    #     计算数列匹配分数
    #     """
    #     # 获取待匹配数列长度
    #     win = len(hidden_state_test)
    #     # 生成列记录匹配数量
    #     df_return_1['match_count_0'] = 0 # 完全匹配
    #     df_return_1['match_count_1'] = 0 # 差一位匹配
    #     df_return_1['match_count__1'] = 0  # 差一位匹配
    #     df_return_1['match_count_2'] = 0  # 差二位匹配
    #     df_return_1['match_count__2'] = 0  # 差二位匹配
    #
    #     df_return_1['hidden_state_s1'] = df_return_1['hidden_state'].shift(1)
    #     df_return_1['hidden_state_s_1'] = df_return_1['hidden_state'].shift(-1)
    #     df_return_1['hidden_state_s2'] = df_return_1['hidden_state'].shift(2)
    #     df_return_1['hidden_state_s_2'] = df_return_1['hidden_state'].shift(-2)
    #
    #     for j in range(win-1,len(train)):
    #         window = df_return_1['hidden_state'].iloc[j-(win-1):j+1].values
    #         match_count = sum(hidden_state_test[j] == window[j] for j in range(win))
    #         df_return_1.iloc[j, df_return_1.columns.get_loc('match_count_0')] = match_count
    #
    #     for j in range(win-1,len(train)):
    #         window = df_return_1['hidden_state_s1'].iloc[j-(win-1):j+1].values
    #         match_count = sum(hidden_state_test[j] == window[j] for j in range(win))
    #         df_return_1.iloc[j, df_return_1.columns.get_loc('match_count_1')] = match_count
    #
    #     for j in range(win-1,len(train)):
    #         window = df_return_1['hidden_state_s_1'].iloc[j-(win-1):j+1].values
    #         match_count = sum(hidden_state_test[j] == window[j] for j in range(win))
    #         df_return_1.iloc[j, df_return_1.columns.get_loc('match_count__1')] = match_count
    #
    #     for j in range(win-1,len(train)):
    #         window = df_return_1['hidden_state_s2'].iloc[j-(win-1):j+1].values
    #         match_count = sum(hidden_state_test[j] == window[j] for j in range(win))
    #         df_return_1.iloc[j, df_return_1.columns.get_loc('match_count_2')] = match_count
    #     for j in range(win-1,len(train)):
    #         window = df_return_1['hidden_state_s_2'].iloc[j-(win-1):j+1].values
    #         match_count = sum(hidden_state_test[j] == window[j] for j in range(win))
    #         df_return_1.iloc[j, df_return_1.columns.get_loc('match_count__2')] = match_count
    #
    #
    #     df_return_1['pro'] =  df_return_1['match_count_0'] * w_0\
    #                      +df_return_1['match_count_1']* w_1 \
    #                     +df_return_1['match_count__1']* w__1\
    #                     +df_return_1['match_count_2'] * w_2\
    #                      +df_return_1['match_count__2'] * w__2
    #
    #     """
    #     计算收益率
    #     """
    #
    #     # 计算下一个月的交易天数
    #     def increment_month(date_str):
    #         # 将字符串转换为 Period 对象
    #         period = pd.Period(date_str, freq='M')
    #         # 增加一个月
    #         next_period = period + 1
    #         # 返回下一个月份的字符串形式
    #         return next_period.strftime('%Y-%m')
    #     next_month = increment_month(i)
    #     next = df[df['y_m'] == next_month ]
    #     next_days = next.shape[0]
    #
    #     # 循环计算某一交易日未来窗口的收益率
    #     train = cal_next_windows_return(next_days,df_return_1)
    #
    #
    #     # 根据收益率判断上涨/下跌概率
    #
    #     up = df_return_1[df_return_1['next_window_last_return'] > up_threshold]['pro'].mean()
    #
    #     down = df_return_1[df_return_1['next_window_last_return'] < down_threshold]['pro'].mean()
    #     # 显著上涨和下跌数量
    #
    #     up_pro = up/(up+down)
    #     i = pd.Period(i, freq='M')
    #     df.loc[df['y_m'] == i,'pro'] = up_pro
    #     # ic
    #
    # # print(test_period)
    # # 将字符串列表转换为 DatetimeIndex
    # datetime_index = pd.to_datetime(test_period, format='%Y-%m')
    #
    # # 将 DatetimeIndex 转换为 PeriodIndex（月份频率）
    # test_period = datetime_index.to_period('M')
    # df_f = df[df['y_m'].isin(test_period)]

    return df_f


"""
函数的参数定义
"""
n = 6

# 生成从2023年1月到2024年6月的所有月份
date_range = pd.date_range(start='2020-12-01', end='2024-06-30', freq='MS')
# 将日期格式化为可比较的字符串格式
formatted_dates = date_range.strftime('%Y-%m')
# 输出结果
test_period = formatted_dates.tolist()

w_matrix = np.array([[1,0,0,0,0], [1,0.5,0.5,0.25,0.25], [1,0.5,0.5,0,0]])
# w_0 = 1  #
# w_1 = 0
# w__1 = 0
# w_2 = 0
# w__2 = 0
threshold  = np.arange(0.05, 0.105 ,0.005)

# 获取待匹配数列长度
# win = len(hidden_state_test)
# 生成长度是win。最后5个数字是1的序列
# ta = np.zeros(win)
# ta[-5:] = 1
# # 生成长度是win,衰减的序列
# sequence = np.exp(-np.log(2) / 10 * np.arange(win))
# x = cal_pos(df_29,n,test_period)
# print('这是x')
# pd.set_option('display.max_rows', None)
# print(x)

dfs_for_backtest = {}
for column, df in dfs.items():
    print(column)
    dfs_for_backtest[column] = cal_pos(df,n,test_period,w_matrix,threshold )
# print(dfs_for_backtest)

# y = cal_pos(dfs['CI005030.WI'],n,test_period,w_0,w_1,w__1,w_2,w__2,up_threshold ,down_threshold)
# print(y)

"""
按月换仓轮动
"""
# 把所有的文件合并到一个大文件里面
merged_df = pd.concat({key:df.add_prefix(f'{key}_')for key,df in dfs_for_backtest.items()},axis=1)
print(merged_df.head())
merged_df.to_csv('33_2.csv')

print('1')









