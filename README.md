# 电商用户行为数据分析文件夹说明

本目录是对 Kaggle 电商用户行为数据集（`user_personalized_features.csv`）进行端到端数据分析的工程文件夹，包含原始数据、数据清洗脚本、分析脚本、Jupyter Notebook、分析报告以及可视化图表。

---

## 一、数据来源与背景

- **数据集名称**：电商用户行为数据集（E-commerce User Personalized Features）
- **样本量**：1000 名用户
- **数据来源**：[Kaggle - ecommerce-product-recommendation-collaborative](https://www.kaggle.com/datasets/kartikeybartwal/ecommerce-product-recommendation-collaborative)
- **分析目标**：购买行为分析、用户分群、活跃度分析、个性化推荐预测

### 字段说明

| 字段 | 说明 |
| --- | --- |
| User_ID | 用户唯一标识符 |
| Age | 年龄 |
| Gender | 性别（Male/Female） |
| Location | 所在地区（Urban/Suburban/Rural） |
| Income | 收入水平 |
| Interests | 兴趣标签（Sports/Technology/Fashion/Travel/Food） |
| Last_Login_Days_Ago | 上次登录距今天数 |
| Purchase_Frequency | 购买频率 |
| Average_Order_Value | 平均订单价值 |
| Total_Spending | 总消费金额 |
| Product_Category_Preference | 偏好产品类别 |
| Time_Spent_on_Site_Minutes | 网站停留时间（分钟） |
| Pages_Viewed | 浏览页面数 |
| Newsletter_Subscription | 是否订阅营销邮件（True/False） |

---

## 二、文件清单

### 1. 数据文件

| 文件 | 说明 |
| --- | --- |
| `user_personalized_features.csv` | 原始数据文件，包含 1000 行 × 14 列用户特征数据。列名已由 `convert_headers.py` 翻译为中文。 |
| `analysis_result.csv` | `ecommerce_analysis.py` 分析后的完整结果，包含原始字段 + 派生特征（活跃度评分、价值等级、聚类标签等）。 |

### 2. 数据处理脚本

| 文件 | 说明 |
| --- | --- |
| `convert_headers.py` | 将 `user_personalized_features.csv` 的英文列名翻译为中文列名，便于后续中文报告输出。 |
| `ecommerce_analysis.py` | 核心分析脚本，覆盖 8 大分析维度：用户画像、行为分析、消费分析、RFM + K-Means 聚类、兴趣偏好、活跃度、相关性、高价值用户画像。 |
| `read/01.py` | 可视化脚本，生成 12 组分析图表（人口统计、活跃度、消费模式、产品偏好、相关性、订阅、聚类、预测、回归、兴趣行为、收入分层、性别×地区）。 |

### 3. 分析报告

| 文件 | 说明 |
| --- | --- |
| `ecommerce_analysis_report.md` | 基于 `ecommerce_analysis.py` 输出的中文数据分析报告，包含核心发现与运营建议。 |
| `电商用户行为数据集_readme.md` | 数据集原始说明文档（来自 Kaggle），包含字段说明和问题描述。 |

### 4. Notebook

| 文件 | 说明 |
| --- | --- |
| `ecommerce_analysis_notebook.ipynb` | Jupyter Notebook 版本的分析工程，共 34 个单元格（15 个 Markdown + 19 个 Code），包含交互式分析、代码和图表输出。 |

### 5. 可视化图表

位于 `read/` 目录下，共 12 张 PNG 图表：

| 图表 | 文件名 | 内容 |
| --- | --- | --- |
| 1 | `chart_01_demographics.png` | 用户人口统计属性四宫格图（年龄、性别、地区、收入） |
| 2 | `chart_02_engagement.png` | 用户活跃度三指标分布（登录间隔、停留时长、浏览页数） |
| 3 | `chart_03_spending.png` | 消费模式分析（购买频率、客单价、总消费、散点图） |
| 4 | `chart_04_product_preference.png` | 产品类别偏好（横向柱状图 + 兴趣×品类热力图） |
| 5 | `chart_05_correlation.png` | 数值特征 Pearson 相关系数热力图 |
| 6 | `chart_06_newsletter.png` | Newsletter 订阅行为分析 |
| 7 | `chart_07_segmentation.png` | 用户分群分析（K-Means 聚类、肘部法则、PCA） |
| 8 | `chart_08_prediction.png` | 购买行为预测（随机森林 + 逻辑回归混淆矩阵、特征重要性） |
| 9 | `chart_09_regression.png` | 消费金额回归预测（线性回归 vs 随机森林） |
| 10 | `chart_10_interest_behavior.png` | 用户兴趣与行为关联分析 |
| 11 | `chart_11_income_tier.png` | 收入分层与消费能力分析 |
| 12 | `chart_12_gender_location.png` | 性别 × 地区 消费行为多维度对比 |

---

## 三、分析框架

本次分析从 8 个核心维度展开：

1. **用户画像分析**：年龄、性别、地区、收入分布
2. **用户行为分析**：登录频率、停留时间、浏览深度、行为模式
3. **消费行为分析**：购买频次、客单价、总消费、高消费用户特征
4. **用户分群分析**：RFM 模型 + K-Means 聚类
5. **兴趣偏好分析**：兴趣标签、品类偏好、交叉分析
6. **用户活跃度分析**：活跃度分级、订阅率、流失风险识别
7. **相关性分析**：关键指标相关性矩阵
8. **高价值用户画像**：TOP20% 用户特征提取与运营建议

---

## 四、核心结论

1. **用户价值与收入无关**：高价值用户平均收入反而低于其他用户，消费意愿比消费能力更重要。
2. **活跃度 ≠ 消费力**：低活跃用户平均消费最高，可能因为他们登录时带有明确购买目的。
3. **客单价是关键区分指标**：高价值用户的核心特征是客单价更高（¥111 vs ¥102）。
4. **17.9% 用户面临流失风险**：25 天以上未登录用户共 179 人，平均历史消费 ¥2,703。
5. **转化路径需优化**：停留时间、登录频率与消费几乎无关，需优化从浏览到购买的转化漏斗。

---

## 五、使用建议

1. **数据准备**：运行 `convert_headers.py` 将列名转为中文（如已翻译则跳过）。
2. **执行分析**：运行 `ecommerce_analysis.py` 生成完整分析结果，输出至 `analysis_result.csv`。
3. **生成图表**：运行 `read/01.py` 自动生成 12 组分析图表。
4. **交互式分析**：打开 `ecommerce_analysis_notebook.ipynb` 进行交互式探索和调参。

### 注意事项

- `ecommerce_analysis.py` 和 `read/01.py` 中的数据路径目前为 `/data/user_personalized_features.csv`（Linux 风格），在 Windows 本地运行时请根据实际路径修改为 `d:\big data\data\analysis\user_personalized_features.csv`。
- `ecommerce_analysis.py` 的输出路径 `/data/analysis/analysis_result.csv` 同样需要按实际环境修改。
- `read/01.py` 的图表保存路径为 `D:\big data\data\`，如需保存到 `analysis` 子目录，请修改保存路径。

---

## 六、依赖环境

主要依赖以下 Python 库：

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`

---

*分析日期：2026-05-27*
