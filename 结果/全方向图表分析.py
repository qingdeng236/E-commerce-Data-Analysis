"""
电商用户行为数据 - 全方向图表分析
运行后自动生成所有分析图表并弹出展示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv(r'/data/user_personalized_features.csv')


# ============================================================
# 图表1：用户人口统计属性四宫格图
# ============================================================
def plot_demographics(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('用户人口统计属性分析', fontsize=18, fontweight='bold', y=0.98)

    # 1-1 年龄分布直方图 + KDE密度曲线
    sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0, 0],
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0, 0].axvline(df['Age'].mean(), color='red', linestyle='--',
                        label=f'均值={df["Age"].mean():.1f}')
    axes[0, 0].set_title('用户年龄分布', fontsize=14)
    axes[0, 0].set_xlabel('年龄')
    axes[0, 0].set_ylabel('人数')
    axes[0, 0].legend()

    # 1-2 性别比例饼图
    gender_counts = df['Gender'].value_counts()
    colors = ['#66b3ff', '#ff9999']
    axes[0, 1].pie(gender_counts, labels=gender_counts.index,
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   textprops={'fontsize': 14})
    axes[0, 1].set_title('用户性别比例', fontsize=14)

    # 1-3 地区分布柱状图
    location_order = ['Urban', 'Suburban', 'Rural']
    sns.countplot(data=df, x='Location', order=location_order,
                  ax=axes[1, 0], palette='Set2', edgecolor='white')
    for p in axes[1, 0].patches:
        axes[1, 0].text(p.get_x() + p.get_width() / 2, p.get_height() + 3,
                        int(p.get_height()), ha='center', fontsize=12)
    axes[1, 0].set_title('用户地区分布', fontsize=14)
    axes[1, 0].set_xlabel('地区')
    axes[1, 0].set_ylabel('人数')

    # 1-4 性别×地区分组收入箱线图
    sns.boxplot(data=df, x='Location', y='Income', hue='Gender',
                ax=axes[1, 1], palette='Pastel1', order=location_order)
    axes[1, 1].set_title('性别×地区 收入分布对比', fontsize=14)
    axes[1, 1].set_xlabel('地区')
    axes[1, 1].set_ylabel('收入')
    axes[1, 1].legend(title='性别')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(r'D:\big data\data\chart_01_demographics.png', bbox_inches='tight')
    plt.show()


# ============================================================
# 图表2：用户活跃度三指标分布图
# ============================================================
def plot_engagement(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('用户活跃度与参与行为分析', fontsize=16, fontweight='bold')

    sns.histplot(df['Last_Login_Days_Ago'], bins=30, kde=True,
                 ax=axes[0], color='coral', edgecolor='white')
    axes[0].set_title('上次登录距今天数', fontsize=13)
    axes[0].set_xlabel('天数')

    sns.histplot(df['Time_Spent_on_Site_Minutes'], bins=30, kde=True,
                 ax=axes[1], color='mediumseagreen', edgecolor='white')
    axes[1].set_title('网站停留时长（分钟）', fontsize=13)
    axes[1].set_xlabel('分钟')

    sns.histplot(df['Pages_Viewed'], bins=30, kde=True,
                 ax=axes[2], color='mediumpurple', edgecolor='white')
    axes[2].set_title('浏览页面数', fontsize=13)
    axes[2].set_xlabel('页面数')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(r'D:\big data\data\chart_02_engagement.png', bbox_inches='tight')
    plt.show()


# ============================================================
# 图表3：消费模式分析（购买频率 + AOV + 总消费 + 散点）
# ============================================================
def plot_spending(df):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('消费模式分析', fontsize=18, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 购买频率柱状图
    ax1 = fig.add_subplot(gs[0, 0])
    freq_counts = df['Purchase_Frequency'].value_counts().sort_index()
    ax1.bar(freq_counts.index, freq_counts.values, color='royalblue', edgecolor='white')
    ax1.set_title('购买频率分布')
    ax1.set_xlabel('购买次数')
    ax1.set_ylabel('人数')

    # 平均订单金额分布
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df['Average_Order_Value'], bins=30, kde=True,
                 ax=ax2, color='darkorange', edgecolor='white')
    ax2.set_title('平均订单金额分布')
    ax2.set_xlabel('金额')

    # 总消费金额分布
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(df['Total_Spending'], bins=30, kde=True,
                 ax=ax3, color='teal', edgecolor='white')
    ax3.set_title('总消费金额分布')
    ax3.set_xlabel('金额')

    # 购买频率 vs 总消费散点图（按产品类别着色）
    ax4 = fig.add_subplot(gs[1, :])
    categories = df['Product_Category_Preference'].unique()
    colors_map = plt.cm.Set1(np.linspace(0, 1, len(categories)))
    for cat, color in zip(categories, colors_map):
        mask = df['Product_Category_Preference'] == cat
        ax4.scatter(df.loc[mask, 'Purchase_Frequency'],
                    df.loc[mask, 'Total_Spending'],
                    c=[color], label=cat, s=40, alpha=0.6, edgecolors='white', linewidth=0.5)
    ax4.legend(title='产品类别', loc='upper left', fontsize=9, ncol=3)
    ax4.set_title('购买频率 vs 总消费金额（按产品类别着色）', fontsize=13)
    ax4.set_xlabel('购买频率')
    ax4.set_ylabel('总消费金额')

    plt.savefig(r'D:\big data\data\chart_03_spending.png', bbox_inches='tight')
    plt.show()


# ============================================================
# 图表4：产品类别偏好（横向柱状图 + 兴趣×品类热力图）
# ============================================================
def plot_product_preference(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('产品类别偏好分析', fontsize=16, fontweight='bold')

    # 各品类偏好人数横向柱状图
    cat_counts = df['Product_Category_Preference'].value_counts()
    colors = plt.cm.Paired(np.linspace(0, 1, len(cat_counts)))
    axes[0].barh(cat_counts.index, cat_counts.values, color=colors, edgecolor='white')
    for i, (cat, count) in enumerate(cat_counts.items()):
        axes[0].text(count + 2, i, str(count), va='center', fontsize=11)
    axes[0].set_title('各产品类别偏好人数', fontsize=14)
    axes[0].set_xlabel('人数')

    # 兴趣 × 产品类别 交叉热力图
    cross_tab = pd.crosstab(df['Interests'], df['Product_Category_Preference'])
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd',
                ax=axes[1], linewidths=0.5, cbar_kws={'shrink': 0.8})
    axes[1].set_title('兴趣 × 产品类别偏好 热力图', fontsize=14)
    axes[1].set_xlabel('产品类别')
    axes[1].set_ylabel('兴趣')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(r'D:\big data\data\chart_04_product_preference.png', bbox_inches='tight')
    plt.show()


# ============================================================
# 图表5：数值特征相关性热力图
# ============================================================
def plot_correlation(df):
    numeric_cols = ['Age', 'Income', 'Last_Login_Days_Ago', 'Purchase_Frequency',
                    'Average_Order_Value', 'Total_Spending',
                    'Time_Spent_on_Site_Minutes', 'Pages_Viewed']
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.8, ax=ax,
                cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1)
    ax.set_title('数值特征 Pearson 相关系数热力图', fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(r'D:\big data\data\chart_05_correlation.png', bbox_inches='tight')
    plt.show()


# ============================================================
# 图表6：Newsletter 订阅分析（柱状图 + 小提琴图 + 分组对比）
# ============================================================
def plot_newsletter(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Newsletter 订阅行为分析', fontsize=16, fontweight='bold')

    # 订阅比例
    sub_counts = df['Newsletter_Subscription'].map({True: '订阅', False: '未订阅'}).value_counts()
    bars = axes[0].bar(sub_counts.index, sub_counts.values,
                       color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.5)
    for bar, val in zip(bars, sub_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{val}\n({val / len(df) * 100:.1f}%)', ha='center', fontsize=12)
    axes[0].set_title('Newsletter 订阅比例')
    axes[0].set_ylabel('人数')

    # 订阅 vs 未订阅总消费小提琴图
    df_plot = df.copy()
    df_plot['Sub_Label'] = df_plot['Newsletter_Subscription'].map({True: '订阅', False: '未订阅'})
    sns.violinplot(data=df_plot, x='Sub_Label', y='Total_Spending',
                   palette=['#e74c3c', '#2ecc71'], ax=axes[1], inner='quartile')
    axes[1].set_title('订阅 vs 未订阅 消费分布')
    axes[1].set_xlabel('')

    # 订阅 vs 未订阅 各指标雷达对比（简化为柱状对比）
    metrics = ['Purchase_Frequency', 'Average_Order_Value', 'Time_Spent_on_Site_Minutes', 'Pages_Viewed']
    metric_labels = ['购买频率', '客单价', '停留时长', '浏览页数']
    sub_mean = df[df['Newsletter_Subscription'] == True][metrics].mean()
    unsub_mean = df[df['Newsletter_Subscription'] == False][metrics].mean()
    x_pos = np.arange(len(metrics))
    width = 0.35
    axes[2].bar(x_pos - width / 2, sub_mean, width, label='订阅', color='#2ecc71')
    axes[2].bar(x_pos + width / 2, unsub_mean, width, label='未订阅', color='#e74c3c')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(metric_labels)
    axes[2].set_title('订阅 vs 未订阅 行为指标对比')
    axes[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(r'D:\big data\data\chart_06_newsletter.png', bbox_inches='tight')
    plt.show()


# ============================================================
# 图表7：用户分群（肘部法则 + PCA降维散点 + 各簇画像雷达图）
# ============================================================
def plot_user_segmentation(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    features = ['Age', 'Income', 'Purchase_Frequency', 'Average_Order_Value',
                'Total_Spending', 'Time_Spent_on_Site_Minutes', 'Pages_Viewed',
                'Last_Login_Days_Ago']
    x = df[features].copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # 肘部法则
    sse_values = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(x_scaled)
        sse_values.append(km.inertia_)

    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_seg = df.copy()
    df_seg['Cluster'] = kmeans.fit_predict(x_scaled)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('用户分群分析（KMeans 聚类）', fontsize=18, fontweight='bold')
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # 肘部法则折线图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(list(k_range), sse_values, 'bo-', linewidth=2.5, markersize=8)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'选择 K={optimal_k}')
    ax1.set_xlabel('聚类数 K', fontsize=12)
    ax1.set_ylabel('SSE（簇内误差平方和）', fontsize=12)
    ax1.set_title('肘部法则确定最优聚类数')
    ax1.legend()

    # PCA 2D 散点图
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(x_pca[:, 0], x_pca[:, 1], c=df_seg['Cluster'],
                          cmap='viridis', s=40, alpha=0.7, edgecolors='white', linewidth=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('簇标签')
    ax2.set_xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title(f'用户分群 PCA 可视化（K={optimal_k}）')

    # 各簇特征画像对比柱状图
    ax3 = fig.add_subplot(gs[1, :])
    cluster_profile = df_seg.groupby('Cluster')[features].mean()
    # 标准化到 0-1 范围方便对比
    cluster_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
    x_bar = np.arange(len(features))
    width = 0.18
    cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, cluster_id in enumerate(cluster_norm.index):
        ax3.bar(x_bar + i * width, cluster_norm.loc[cluster_id], width,
                label=f'群体 {cluster_id}', color=cluster_colors[i])
    ax3.set_xticks(x_bar + width * 1.5)
    ax3.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=9)
    ax3.set_ylabel('标准化值 (0-1)')
    ax3.set_title('各用户群体特征画像对比')
    ax3.legend()

    plt.savefig(r'D:\big data\data\chart_07_segmentation.png', bbox_inches='tight')
    plt.show()

    return df_seg


# ============================================================
# 图表8：购买行为预测（混淆矩阵 + 特征重要性）
# ============================================================
def plot_purchase_prediction(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix, classification_report

    median_spending = df['Total_Spending'].median()
    df_enc = df.copy()
    df_enc['High_Spender'] = np.where(df_enc['Total_Spending'] > median_spending, 1, 0)

    for col in ['Gender', 'Location', 'Interests', 'Product_Category_Preference']:
        le = LabelEncoder()
        df_enc[col + '_enc'] = le.fit_transform(df_enc[col])

    feature_cols = ['Age', 'Income', 'Last_Login_Days_Ago', 'Purchase_Frequency',
                    'Average_Order_Value', 'Time_Spent_on_Site_Minutes', 'Pages_Viewed',
                    'Gender_enc', 'Location_enc', 'Interests_enc',
                    'Product_Category_Preference_enc']
    df_enc['Newsletter_enc'] = df_enc['Newsletter_Subscription'].astype(int)
    feature_cols.append('Newsletter_enc')

    x = df_enc[feature_cols]
    y = df_enc['High_Spender']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('购买行为预测（高消费用户分类）', fontsize=18, fontweight='bold')
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # 随机森林混淆矩阵
    ax1 = fig.add_subplot(gs[0, 0])
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['低消费', '高消费'], yticklabels=['低消费', '高消费'])
    ax1.set_title('随机森林 混淆矩阵')
    ax1.set_xlabel('预测标签')
    ax1.set_ylabel('真实标签')

    # 逻辑回归混淆矩阵
    ax2 = fig.add_subplot(gs[0, 1])
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                xticklabels=['低消费', '高消费'], yticklabels=['低消费', '高消费'])
    ax2.set_title('逻辑回归 混淆矩阵')
    ax2.set_xlabel('预测标签')
    ax2.set_ylabel('真实标签')

    # 特征重要性排名
    ax3 = fig.add_subplot(gs[1, :])
    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    importance_sorted = importance.sort_values(ascending=True)
    bars = ax3.barh(importance_sorted.index, importance_sorted.values,
                    color='steelblue', edgecolor='white')
    ax3.set_title('随机森林 - 特征重要性排名', fontsize=14)
    ax3.set_xlabel('重要性')

    plt.savefig(r'D:\big data\data\chart_08_prediction.png', bbox_inches='tight')
    plt.show()

    print("随机森林分类报告：")
    print(classification_report(y_test, y_pred_rf))
    print("逻辑回归分类报告：")
    print(classification_report(y_test, y_pred_lr))


# ============================================================
# 图表9：消费金额回归预测（预测 vs 真实散点图）
# ============================================================
def plot_spending_regression(df):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import r2_score

    df_enc = df.copy()
    for col in ['Gender', 'Location', 'Interests', 'Product_Category_Preference']:
        le = LabelEncoder()
        df_enc[col + '_enc'] = le.fit_transform(df_enc[col])

    feature_cols = ['Age', 'Income', 'Last_Login_Days_Ago', 'Purchase_Frequency',
                    'Average_Order_Value', 'Time_Spent_on_Site_Minutes', 'Pages_Viewed',
                    'Gender_enc', 'Location_enc', 'Interests_enc',
                    'Product_Category_Preference_enc']
    df_enc['Newsletter_enc'] = df_enc['Newsletter_Subscription'].astype(int)
    feature_cols.append('Newsletter_enc')

    x = df_enc[feature_cols]
    y = df_enc['Total_Spending']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('消费金额回归预测（预测值 vs 真实值）', fontsize=16, fontweight='bold')

    r2_lr = r2_score(y_test, y_pred_lr)
    r2_rf = r2_score(y_test, y_pred_rf)

    # 线性回归
    axes[0].scatter(y_test, y_pred_lr, alpha=0.5, s=25, color='steelblue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='理想预测线')
    axes[0].set_title(f'线性回归 (R²={r2_lr:.4f})')
    axes[0].set_xlabel('真实消费金额')
    axes[0].set_ylabel('预测消费金额')
    axes[0].legend()

    # 随机森林回归
    axes[1].scatter(y_test, y_pred_rf, alpha=0.5, s=25, color='darkorange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='理想预测线')
    axes[1].set_title(f'随机森林回归 (R²={r2_rf:.4f})')
    axes[1].set_xlabel('真实消费金额')
    axes[1].set_ylabel('预测消费金额')
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(r'D:\big data\data\chart_09_regression.png', bbox_inches='tight')
    plt.show()


# ============================================================
# 图表10：兴趣与行为关联分析（四指标柱状/箱线图）
# ============================================================
def plot_interest_behavior(df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('用户兴趣与行为关联分析', fontsize=18, fontweight='bold')

    interests_order = df['Interests'].value_counts().index
    palette = 'Set3'

    sns.barplot(data=df, x='Interests', y='Purchase_Frequency',
                order=interests_order, ax=axes[0, 0], palette=palette,
                errorbar='ci', capsize=0.1)
    axes[0, 0].set_title('不同兴趣用户的平均购买频率')
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.barplot(data=df, x='Interests', y='Total_Spending',
                order=interests_order, ax=axes[0, 1], palette=palette,
                errorbar='ci', capsize=0.1)
    axes[0, 1].set_title('不同兴趣用户的平均总消费')
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.boxplot(data=df, x='Interests', y='Time_Spent_on_Site_Minutes',
                order=interests_order, ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title('不同兴趣用户网站停留时长')
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.boxplot(data=df, x='Interests', y='Pages_Viewed',
                order=interests_order, ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title('不同兴趣用户浏览页面数')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(r'D:\big data\data\chart_10_interest_behavior.png', bbox_inches='tight')
    plt.show()

    groups = [g['Total_Spending'].values for _, g in df.groupby('Interests')]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"ANOVA 兴趣→消费: F={f_stat:.3f}, p={p_val:.4f}")


# ============================================================
# 图表11：收入分层消费分析
# ============================================================
def plot_income_tier(df):
    df_tier = df.copy()
    df_tier['Income_Tier'] = pd.qcut(df_tier['Income'], q=4,
                                     labels=['低收入', '中低收入', '中高收入', '高收入'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('收入分层与消费能力分析', fontsize=16, fontweight='bold')

    tier_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

    # 各层级平均总消费
    tier_spending = df_tier.groupby('Income_Tier', observed=True)['Total_Spending'].mean()
    bars1 = axes[0].bar(tier_spending.index, tier_spending.values, color=tier_colors, edgecolor='white')
    for bar, val in zip(bars1, tier_spending.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                     f'{val:.0f}', ha='center', fontsize=11)
    axes[0].set_title('各收入层级平均总消费')
    axes[0].set_ylabel('平均总消费')

    # 各层级平均客单价
    tier_aov = df_tier.groupby('Income_Tier', observed=True)['Average_Order_Value'].mean()
    bars2 = axes[1].bar(tier_aov.index, tier_aov.values, color=tier_colors, edgecolor='white')
    for bar, val in zip(bars2, tier_aov.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}', ha='center', fontsize=11)
    axes[1].set_title('各收入层级平均客单价')
    axes[1].set_ylabel('平均客单价')

    # 各层级购买频率
    tier_freq = df_tier.groupby('Income_Tier', observed=True)['Purchase_Frequency'].mean()
    bars3 = axes[2].bar(tier_freq.index, tier_freq.values, color=tier_colors, edgecolor='white')
    for bar, val in zip(bars3, tier_freq.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{val:.2f}', ha='center', fontsize=11)
    axes[2].set_title('各收入层级平均购买频率')
    axes[2].set_ylabel('平均购买频率')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(r'D:\big data\data\chart_11_income_tier.png', bbox_inches='tight')
    plt.show()


# ============================================================
# 图表12：性别×地区 消费行为多维度对比
# ============================================================
def plot_gender_location_spending(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('性别 × 地区 消费行为多维度对比', fontsize=16, fontweight='bold')

    location_order = ['Urban', 'Suburban', 'Rural']

    # 平均总消费
    sns.barplot(data=df, x='Location', y='Total_Spending', hue='Gender',
                order=location_order, ax=axes[0], palette='Set1', errorbar='ci')
    axes[0].set_title('性别×地区 平均总消费')
    axes[0].set_xlabel('地区')

    # 平均客单价
    sns.barplot(data=df, x='Location', y='Average_Order_Value', hue='Gender',
                order=location_order, ax=axes[1], palette='Set1', errorbar='ci')
    axes[1].set_title('性别×地区 平均客单价')
    axes[1].set_xlabel('地区')
    axes[1].get_legend().remove()

    # 平均购买频率
    sns.barplot(data=df, x='Location', y='Purchase_Frequency', hue='Gender',
                order=location_order, ax=axes[2], palette='Set1', errorbar='ci')
    axes[2].set_title('性别×地区 平均购买频率')
    axes[2].set_xlabel('地区')
    axes[2].get_legend().remove()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(r'D:\big data\data\chart_12_gender_location.png', bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("开始生成全部 12 组分析图表...")
    print(f"图表将保存至: D:\\big data\\data\\")
    print("=" * 60)

    print("\n[1/12] 用户人口统计属性...")
    plot_demographics(df)

    print("[2/12] 用户活跃度...")
    plot_engagement(df)

    print("[3/12] 消费模式...")
    plot_spending(df)

    print("[4/12] 产品类别偏好...")
    plot_product_preference(df)

    print("[5/12] 相关性热力图...")
    plot_correlation(df)

    print("[6/12] Newsletter 订阅...")
    plot_newsletter(df)

    print("[7/12] 用户分群聚类...")
    df = plot_user_segmentation(df)

    print("[8/12] 购买行为预测...")
    plot_purchase_prediction(df)

    print("[9/12] 消费金额回归...")
    plot_spending_regression(df)

    print("[10/12] 兴趣与行为关联...")
    plot_interest_behavior(df)

    print("[11/12] 收入分层消费...")
    plot_income_tier(df)

    print("[12/12] 性别×地区 消费对比...")
    plot_gender_location_spending(df)

    print("\n" + "=" * 60)
    print("全部 12 组图表生成完毕！")
    print("保存文件列表：")
    for i in range(1, 13):
        print(f"  chart_{i:02d}_*.png")
    print("=" * 60)
