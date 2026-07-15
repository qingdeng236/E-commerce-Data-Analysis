"""
电商用户行为数据分析
===================
数据集：user_personalized_features.csv
字段说明：
- User_ID: 用户唯一标识
- Age: 年龄
- Gender: 性别 (Male/Female)
- Location: 地区 (Urban/Suburban/Rural)
- Income: 收入水平
- Interests: 兴趣标签 (Sports/Technology/Fashion/Travel/Food)
- Last_Login_Days_Ago: 上次登录距今天数
- Purchase_Frequency: 购买频率
- Average_Order_Value: 平均订单价值
- Total_Spending: 总消费金额
- Product_Category_Preference: 产品类别偏好
- Time_Spent_on_Site_Minutes: 网站停留时间(分钟)
- Pages_Viewed: 浏览页面数
- Newsletter_Subscription: 邮件订阅 (True/False)

分析维度：
1. 用户画像分析 - 人口统计学特征分布
2. 用户行为分析 - 登录/浏览/停留行为
3. 消费行为分析 - 购买频次/客单价/总消费
4. 用户分群分析 - RFM模型 + K-Means聚类
5. 兴趣偏好分析 - 兴趣与品类偏好交叉
6. 用户活跃度分析 - 登录间隔/订阅率/活跃度分级
7. 相关性分析 - 关键指标相关性矩阵
8. 高价值用户画像 - TOP20%用户特征提取
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据加载与预处理
# ============================================================
def load_and_preprocess_data(file_path):
    """
    负责部分：数据加载、清洗、类型转换
    - 处理缺失值
    - 转换数据类型
    - 创建衍生特征
    """
    # 加载数据
    df = pd.read_csv(file_path)
    
    # 清理列名（去除首尾空格）
    df.columns = df.columns.str.strip()
    
    # 清理User_ID（去除#号）
    df['User_ID'] = df['User_ID'].str.replace('#', '').astype(int)
    
    # 转换布尔值（处理CSV中的字符串布尔值）
    df['Newsletter_Subscription'] = df['Newsletter_Subscription'].apply(
        lambda x: str(x).strip().lower() == 'true'
    )
    
    # 创建衍生特征
    # 1. 用户活跃度评分（登录越近、停留越长、浏览越多 = 越活跃）
    df['Activity_Score'] = (
        (30 - df['Last_Login_Days_Ago']) / 30 * 0.4 +  # 登录新鲜度（40%权重）
        df['Time_Spent_on_Site_Minutes'] / df['Time_Spent_on_Site_Minutes'].max() * 0.3 +  # 停留时间（30%）
        df['Pages_Viewed'] / df['Pages_Viewed'].max() * 0.3  # 浏览深度（30%）
    )
    
    # 2. 用户价值等级（基于总消费分位数）
    df['Value_Level'] = pd.qcut(df['Total_Spending'], q=5, labels=['低价值', '中低价值', '中等价值', '中高价值', '高价值'])
    
    # 3. 活跃度等级
    df['Activity_Level'] = pd.qcut(df['Activity_Score'], q=3, labels=['低活跃', '中活跃', '高活跃'])
    
    # 4. 客单价分级
    df['AOV_Level'] = pd.qcut(df['Average_Order_Value'], q=3, labels=['低客单', '中客单', '高客单'])
    
    print(f"数据加载完成：{df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"缺失值统计：\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    return df


# ============================================================
# 1. 用户画像分析
# ============================================================
def analyze_user_profile(df):
    """
    负责部分：人口统计学特征分析
    - 年龄分布与分段
    - 性别比例
    - 地区分布
    - 收入分布
    - 交叉分析（性别×地区、年龄×收入等）
    """
    print("\n" + "="*60)
    print("1. 用户画像分析")
    print("="*60)
    
    # 年龄分段
    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['18-25岁', '26-35岁', '36-45岁', '46-55岁', '55岁以上']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    # 基础统计
    print("\n【年龄分布】")
    print(df['Age_Group'].value_counts().sort_index())
    
    print("\n【性别比例】")
    gender_ratio = df['Gender'].value_counts(normalize=True) * 100
    print(gender_ratio)
    
    print("\n【地区分布】")
    print(df['Location'].value_counts(normalize=True) * 100)
    
    print("\n【收入统计】")
    print(f"平均收入：¥{df['Income'].mean():.0f}")
    print(f"收入中位数：¥{df['Income'].median():.0f}")
    print(f"收入标准差：¥{df['Income'].std():.0f}")
    
    # 交叉分析：性别 × 地区
    print("\n【性别 × 地区分布】")
    cross_gender_location = pd.crosstab(df['Gender'], df['Location'], normalize='index') * 100
    print(cross_gender_location.round(1))
    
    # 交叉分析：年龄段 × 收入水平
    print("\n【年龄段 × 平均收入】")
    age_income = df.groupby('Age_Group')['Income'].mean()
    print(age_income.round(0))
    
    return df


# ============================================================
# 2. 用户行为分析
# ============================================================
def analyze_user_behavior(df):
    """
    负责部分：用户平台行为分析
    - 登录频率分布
    - 停留时间分析
    - 浏览深度分析
    - 行为模式识别（高频短停留 vs 低频长停留）
    """
    print("\n" + "="*60)
    print("2. 用户行为分析")
    print("="*60)
    
    # 登录行为分析
    print("\n【登录行为分布】")
    login_bins = [0, 7, 14, 21, 30]
    login_labels = ['7天内登录', '7-14天', '14-21天', '21-30天']
    df['Login_Frequency_Group'] = pd.cut(df['Last_Login_Days_Ago'], bins=login_bins, labels=login_labels)
    print(df['Login_Frequency_Group'].value_counts().sort_index())
    
    # 停留时间分析
    print("\n【网站停留时间】")
    print(f"平均停留时间：{df['Time_Spent_on_Site_Minutes'].mean():.0f} 分钟")
    print(f"中位数：{df['Time_Spent_on_Site_Minutes'].median():.0f} 分钟")
    time_bins = [0, 120, 300, 600]
    time_labels = ['2小时内', '2-5小时', '5小时以上']
    df['Time_Group'] = pd.cut(df['Time_Spent_on_Site_Minutes'], bins=time_bins, labels=time_labels)
    print(df['Time_Group'].value_counts())
    
    # 浏览深度分析
    print("\n【浏览页面数分布】")
    print(f"平均浏览页面：{df['Pages_Viewed'].mean():.1f} 页")
    print(f"中位数：{df['Pages_Viewed'].median():.0f} 页")
    
    # 行为模式识别
    # 高频短停留用户 vs 低频长停留用户
    df['Behavior_Pattern'] = np.where(
        (df['Last_Login_Days_Ago'] <= 7) & (df['Time_Spent_on_Site_Minutes'] < df['Time_Spent_on_Site_Minutes'].median()),
        '高频短停留',
        np.where(
            (df['Last_Login_Days_Ago'] > 14) & (df['Time_Spent_on_Site_Minutes'] >= df['Time_Spent_on_Site_Minutes'].median()),
            '低频长停留',
            '其他模式'
        )
    )
    print("\n【行为模式分布】")
    print(df['Behavior_Pattern'].value_counts())
    
    return df


# ============================================================
# 3. 消费行为分析
# ============================================================
def analyze_purchase_behavior(df):
    """
    负责部分：消费行为深度分析
    - 购买频率分布
    - 客单价分析
    - 总消费分布
    - 消费频次 × 客单价矩阵
    - 高消费用户特征
    """
    print("\n" + "="*60)
    print("3. 消费行为分析")
    print("="*60)
    
    # 购买频率分析
    print("\n【购买频率分布】")
    print(df['Purchase_Frequency'].describe())
    freq_bins = [-1, 2, 5, 8, 10]
    freq_labels = ['低频(0-2次)', '中频(3-5次)', '高频(6-8次)', '极高频(9-10次)']
    df['Purchase_Freq_Group'] = pd.cut(df['Purchase_Frequency'], bins=freq_bins, labels=freq_labels)
    print(df['Purchase_Freq_Group'].value_counts().sort_index())
    
    # 客单价分析
    print("\n【客单价分布】")
    print(f"平均客单价：¥{df['Average_Order_Value'].mean():.0f}")
    print(f"客单价中位数：¥{df['Average_Order_Value'].median():.0f}")
    
    # 总消费分析
    print("\n【总消费分布】")
    print(f"平均总消费：¥{df['Total_Spending'].mean():.0f}")
    print(f"总消费中位数：¥{df['Total_Spending'].median():.0f}")
    print(f"消费TOP10%阈值：¥{df['Total_Spending'].quantile(0.9):.0f}")
    
    # 消费矩阵：购买频率 × 客单价
    print("\n【购买频率 × 客单价 交叉分析】")
    purchase_aov_matrix = df.groupby(['Purchase_Freq_Group', 'AOV_Level'])['Total_Spending'].mean()
    print(purchase_aov_matrix.round(0))
    
    # 高消费用户特征（TOP20%）
    top_spenders = df[df['Total_Spending'] >= df['Total_Spending'].quantile(0.8)]
    print("\n【高消费用户(TOP20%)特征】")
    print(f"平均年龄：{top_spenders['Age'].mean():.1f}岁")
    print(f"平均收入：¥{top_spenders['Income'].mean():.0f}")
    print(f"平均购买频率：{top_spenders['Purchase_Frequency'].mean():.1f}次")
    print(f"平均客单价：¥{top_spenders['Average_Order_Value'].mean():.0f}")
    print(f"邮件订阅率：{top_spenders['Newsletter_Subscription'].mean()*100:.1f}%")
    
    return df


# ============================================================
# 4. 用户分群分析（RFM + K-Means）
# ============================================================
def cluster_users(df):
    """
    负责部分：用户分群建模
    - RFM模型构建（Recency/Frequency/Monetary）
    - K-Means聚类（自动确定最佳簇数）
    - 各簇用户特征画像
    - 分群运营建议
    """
    print("\n" + "="*60)
    print("4. 用户分群分析（RFM + K-Means）")
    print("="*60)
    
    # 构建RFM特征
    # R: 最近登录天数（越小越好，取反）
    # F: 购买频率
    # M: 总消费金额
    rfm = df[['Last_Login_Days_Ago', 'Purchase_Frequency', 'Total_Spending']].copy()
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # 数据标准化
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # K-Means聚类（假设分为4个群体）
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 分析各簇特征
    print("\n【各簇用户特征】")
    cluster_profile = df.groupby('Cluster').agg({
        'Last_Login_Days_Ago': 'mean',
        'Purchase_Frequency': 'mean',
        'Total_Spending': 'mean',
        'Average_Order_Value': 'mean',
        'Time_Spent_on_Site_Minutes': 'mean',
        'Income': 'mean',
        'User_ID': 'count'
    }).round(1)
    cluster_profile.columns = ['平均登录间隔(天)', '平均购买频次', '平均总消费', '平均客单价', '平均停留时间(分)', '平均收入', '用户数']
    print(cluster_profile)
    
    # 为各簇命名
    # 根据RFM特征判断簇类型
    cluster_names = {}
    for cluster_id in range(4):
        cluster_data = df[df['Cluster'] == cluster_id]
        r_score = cluster_data['Last_Login_Days_Ago'].mean()
        f_score = cluster_data['Purchase_Frequency'].mean()
        m_score = cluster_data['Total_Spending'].mean()
        
        # 简单规则命名
        if f_score > df['Purchase_Frequency'].median() and m_score > df['Total_Spending'].median():
            cluster_names[cluster_id] = '高价值活跃用户'
        elif f_score > df['Purchase_Frequency'].median():
            cluster_names[cluster_id] = '频次型用户'
        elif m_score > df['Total_Spending'].median():
            cluster_names[cluster_id] = '高客单用户'
        else:
            cluster_names[cluster_id] = '低价值用户'
    
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
    
    print("\n【各簇命名】")
    for cid, name in cluster_names.items():
        count = len(df[df['Cluster'] == cid])
        print(f"簇{cid}: {name} ({count}人, {count/len(df)*100:.1f}%)")
    
    return df


# ============================================================
# 5. 兴趣偏好分析
# ============================================================
def analyze_interests(df):
    """
    负责部分：兴趣与品类偏好分析
    - 兴趣标签分布
    - 品类偏好分布
    - 兴趣 × 品类交叉分析
    - 兴趣 × 消费行为分析
    """
    print("\n" + "="*60)
    print("5. 兴趣偏好分析")
    print("="*60)
    
    # 兴趣分布
    print("\n【兴趣标签分布】")
    interest_dist = df['Interests'].value_counts()
    print(interest_dist)
    
    # 品类偏好分布
    print("\n【产品类别偏好分布】")
    category_dist = df['Product_Category_Preference'].value_counts()
    print(category_dist)
    
    # 兴趣 × 品类交叉分析
    print("\n【兴趣 × 品类偏好 交叉表】")
    interest_category = pd.crosstab(df['Interests'], df['Product_Category_Preference'])
    print(interest_category)
    
    # 不同兴趣用户的消费特征
    print("\n【不同兴趣用户的消费特征】")
    interest_spending = df.groupby('Interests').agg({
        'Total_Spending': 'mean',
        'Purchase_Frequency': 'mean',
        'Average_Order_Value': 'mean',
        'Time_Spent_on_Site_Minutes': 'mean'
    }).round(1)
    interest_spending.columns = ['平均总消费', '平均购买频次', '平均客单价', '平均停留时间']
    print(interest_spending)
    
    return df


# ============================================================
# 6. 用户活跃度分析
# ============================================================
def analyze_activity(df):
    """
    负责部分：用户活跃度与留存分析
    - 活跃度分级
    - 邮件订阅率分析
    - 活跃度与消费关系
    - 流失风险用户识别
    """
    print("\n" + "="*60)
    print("6. 用户活跃度分析")
    print("="*60)
    
    # 活跃度分布
    print("\n【活跃度等级分布】")
    print(df['Activity_Level'].value_counts())
    
    # 邮件订阅率
    print("\n【邮件订阅率】")
    sub_rate = df['Newsletter_Subscription'].mean() * 100
    print(f"总体订阅率：{sub_rate:.1f}%")
    
    # 不同活跃度用户的订阅率
    print("\n【各活跃度等级订阅率】")
    sub_by_activity = df.groupby('Activity_Level')['Newsletter_Subscription'].mean() * 100
    print(sub_by_activity.round(1))
    
    # 活跃度与消费关系
    print("\n【活跃度 × 平均消费】")
    activity_spending = df.groupby('Activity_Level')['Total_Spending'].mean()
    print(activity_spending.round(0))
    
    # 流失风险用户识别（30天未登录）
    churn_risk = df[df['Last_Login_Days_Ago'] >= 25]
    print(f"\n【流失风险用户（25天+未登录）】")
    print(f"数量：{len(churn_risk)}人 ({len(churn_risk)/len(df)*100:.1f}%)")
    print(f"平均历史消费：¥{churn_risk['Total_Spending'].mean():.0f}")
    print(f"平均购买频次：{churn_risk['Purchase_Frequency'].mean():.1f}次")
    
    return df


# ============================================================
# 7. 相关性分析
# ============================================================
def analyze_correlations(df):
    """
    负责部分：关键指标相关性分析
    - 数值变量相关性矩阵
    - 强相关性识别
    - 关键驱动因素发现
    """
    print("\n" + "="*60)
    print("7. 相关性分析")
    print("="*60)
    
    # 选择数值变量
    numeric_cols = ['Age', 'Income', 'Last_Login_Days_Ago', 'Purchase_Frequency', 
                    'Average_Order_Value', 'Total_Spending', 'Time_Spent_on_Site_Minutes', 
                    'Pages_Viewed', 'Activity_Score']
    
    # 计算相关性矩阵
    corr_matrix = df[numeric_cols].corr()
    
    print("\n【关键相关性】（|r| > 0.3）")
    # 提取强相关性
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:
                strong_corr.append({
                    '变量1': corr_matrix.columns[i],
                    '变量2': corr_matrix.columns[j],
                    '相关系数': corr_val
                })
    
    strong_corr_df = pd.DataFrame(strong_corr).sort_values('相关系数', key=abs, ascending=False)
    print(strong_corr_df)
    
    # 重点发现
    print("\n【关键发现】")
    # 收入与消费的关系
    income_spending_corr = corr_matrix.loc['Income', 'Total_Spending']
    print(f"收入与总消费相关性：{income_spending_corr:.3f}")
    
    # 停留时间与消费的关系
    time_spending_corr = corr_matrix.loc['Time_Spent_on_Site_Minutes', 'Total_Spending']
    print(f"停留时间与总消费相关性：{time_spending_corr:.3f}")
    
    # 登录频率与购买频率的关系
    login_purchase_corr = corr_matrix.loc['Last_Login_Days_Ago', 'Purchase_Frequency']
    print(f"登录间隔与购买频率相关性：{login_purchase_corr:.3f}")
    
    return corr_matrix


# ============================================================
# 8. 高价值用户画像
# ============================================================
def analyze_high_value_users(df):
    """
    负责部分：高价值用户深度画像
    - TOP20%用户特征提取
    - 与平均用户对比
    - 可操作的运营建议
    """
    print("\n" + "="*60)
    print("8. 高价值用户画像（TOP20%）")
    print("="*60)
    
    # 定义高价值用户
    threshold = df['Total_Spending'].quantile(0.8)
    high_value = df[df['Total_Spending'] >= threshold]
    avg_users = df[df['Total_Spending'] < threshold]
    
    # 对比分析
    comparison = pd.DataFrame({
        '高价值用户(TOP20%)': high_value[['Age', 'Income', 'Purchase_Frequency', 
                                          'Average_Order_Value', 'Time_Spent_on_Site_Minutes',
                                          'Pages_Viewed']].mean(),
        '其他用户': avg_users[['Age', 'Income', 'Purchase_Frequency', 
                               'Average_Order_Value', 'Time_Spent_on_Site_Minutes',
                               'Pages_Viewed']].mean()
    }).round(1)
    
    print("\n【高价值用户 vs 其他用户对比】")
    print(comparison)
    
    # 高价值用户特征
    print("\n【高价值用户特征分布】")
    print(f"性别比例：\n{high_value['Gender'].value_counts(normalize=True)*100}")
    print(f"\n地区分布：\n{high_value['Location'].value_counts(normalize=True)*100}")
    print(f"\n兴趣分布：\n{high_value['Interests'].value_counts(normalize=True)*100}")
    print(f"\n品类偏好：\n{high_value['Product_Category_Preference'].value_counts(normalize=True)*100}")
    print(f"\n邮件订阅率：{high_value['Newsletter_Subscription'].mean()*100:.1f}%")
    
    # 运营建议
    print("\n【运营建议】")
    # 找出高价值用户最显著的特征
    top_interest = high_value['Interests'].mode()[0]
    top_category = high_value['Product_Category_Preference'].mode()[0]
    top_location = high_value['Location'].mode()[0]
    
    print(f"1. 重点运营兴趣标签：{top_interest}")
    print(f"2. 重点推荐品类：{top_category}")
    print(f"3. 重点投放地区：{top_location}")
    print(f"4. 高价值用户邮件订阅率更高，建议加强邮件营销")
    print(f"5. 高价值用户停留时间更长，建议优化用户体验延长停留")
    
    return high_value


# ============================================================
# 主函数
# ============================================================
def main():
    """
    负责部分：流程编排
    - 按顺序执行所有分析模块
    - 输出汇总报告
    """
    # 数据路径
    file_path = r'/data/user_personalized_features.csv'
    
    # 1. 加载数据
    df = load_and_preprocess_data(file_path)
    
    # 2. 用户画像分析
    df = analyze_user_profile(df)
    
    # 3. 用户行为分析
    df = analyze_user_behavior(df)
    
    # 4. 消费行为分析
    df = analyze_purchase_behavior(df)
    
    # 5. 用户分群
    df = cluster_users(df)
    
    # 6. 兴趣偏好分析
    df = analyze_interests(df)
    
    # 7. 活跃度分析
    df = analyze_activity(df)
    
    # 8. 相关性分析
    corr_matrix = analyze_correlations(df)
    
    # 9. 高价值用户画像
    high_value_users = analyze_high_value_users(df)
    
    # 保存分析结果
    output_path = r'/data/analysis/analysis_result.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n分析结果已保存至：{output_path}")
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == '__main__':
    main()
