
import pandas as pd
import numpy as np
df=pd.read_csv('supplychain_cleaned.csv')


df.head()


# ##Converting the date columns to the datetime


df['order_date']=pd.to_datetime(df['order_date'])
df['shipping_date'] = pd.to_datetime(df['shipping_date'])


# ##Extracting the time based features


df['order_year']=df['order_date'].dt.year
df['order_month']=df['order_date'].dt.month
df['order_day']=df['order_date'].dt.day
df['order_weekday']=df['order_date'].dt.weekday
df['order_weekdate']=df['order_date'].dt.isocalendar().week


# Checking if there is  a weekend or not


df['is_weekend'] = df['order_weekday'].apply(lambda x: 1 if x >= 5 else 0)


# The new features that are created are as follows:
# 
# -weekend demand trends
# 
# -weekend shopping behaviour


# ##Delivery Performance Features


# The Delivery time
df['delivery_time'] = (df['shipping_date'] - df['order_date']).dt.days

# The Shipping delay
df['shipping_delay'] = df['days_for_shipping_(real)'] - df['days_for_shipment_(scheduled)']

# The Late delivery flag
df['late_delivery_flag'] = df['shipping_delay'].apply(lambda x: 1 if x > 0 else 0)


# This helps to predict the late deliveries


# ##Profitability Features


# The profit per product
df['profit_per_item'] = df['order_profit_per_order'] / (df['order_item_quantity'] + 1)


# This shows the business profitability patterns


# ##Product Demand Feature


#The product popularity
product_orders = df.groupby('product_name')['order_item_quantity'].transform('sum')
df['product_popularity'] = product_orders


# This helps to detect high_demand products


# ##Market Level Feature


#The total orders per region
df['region_order_volume'] = df.groupby('order_region')['order_id'].transform('count')


# This is useful for regional demand forecasting


# ##Order Size Category


# Order size category
df['order_size'] = pd.cut(
    df['order_item_quantity'],
    bins=[0,2,5,10,50],
    labels=['small','medium','large','bulk']
)


# Encoding it:


df = pd.get_dummies(df, columns=['order_size'], drop_first=True)


# This helps because sales pattern differes between small orders and bulk orders.


# ##Log Transform Skewed Variable


df['quantity_log'] = np.log1p(df['order_item_quantity'])
df['profit_log'] = np.log1p(df['order_profit_per_order'])


# This helps to reduce skewness


# ##Encoding Categorical Variables


df=pd.get_dummies(df,drop_first=True)


# This step is done because the ML models require numerical input


y = df['sales']
X = df.drop('sales', axis=1)


print("Saving dataset...")

df.to_parquet("supplychain_feature_engineered.csv")

print("Saved successfully")
