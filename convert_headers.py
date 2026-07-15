import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'd:\big data\data\analysis\user_personalized_features.csv')

# 根据README.md中的字段说明进行翻译
column_mapping = {
    'User_ID': '用户ID',
    'Age': '年龄',
    'Gender': '性别',
    'Location': '所在地区',
    'Income': '收入水平',
    'Interests': '兴趣偏好',
    'Last_Login_Days_Ago': '上次登录天数',
    'Purchase_Frequency': '购买频率',
    'Average_Order_Value': '平均订单价值',
    'Total_Spending': '总消费金额',
    'Product_Category_Preference': '偏好产品类别',
    'Time_Spent_on_Site_Minutes': '网站停留时间_分钟',
    'Pages_Viewed': '浏览页面数',
    'Newsletter_Subscription': '是否订阅营销通知'
}

# 重命名列
df = df.rename(columns=column_mapping)

# 保存修改后的CSV文件
df.to_csv(r'd:\big data\data\analysis\user_personalized_features.csv', index=False, encoding='utf-8-sig')

print("列标题翻译完成！")
print("新的列标题：")
print(list(df.columns))