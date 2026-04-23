# Data Dictionary — Supply Chain Dataset

## 1. Introduction
This document provides a Data Dictionary for the cleaned Supply Chain dataset. It describes the meaning, format, and details of each variable to help analysts understand the dataset before performing analysis or building models.

## 2. Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| type | str | Type of payment method used |
| days_for_shipping_(real) | int | Actual number of days taken for shipping |
| days_for_shipment_(scheduled) | int | Planned number of days for shipment |
| delivery_status | str | Status of order delivery |
| late_delivery_risk | int | Indicates risk of late delivery (0 = No, 1 = Yes) |
| category_id | int | Unique ID of product category |
| category_name | str | Name of the product category |
| customer_city | str | City of the customer |
| customer_country | str | Country of the customer |
| customer_segment | str | Type of customer segment |
| customer_state | str | State of the customer |
| customer_zipcode | int | Postal code of the customer |
| department_id | int | Unique ID of department |
| department_name | str | Name of the department |
| market | str | Market region of the order |
| order_city | str | City where order was placed |
| order_country | str | Country where order was placed |
| order_customer_id | int | Unique ID of the customer |
| order_date | date | Date and time when order was placed |
| order_id | int | Unique identifier for each order |
| order_item_discount | float | Discount amount applied on order item |
| order_item_discount_rate | float | Percentage of discount applied |
| order_item_profit_ratio | float | Profit ratio of the order item |
| order_item_quantity | int | Quantity of items ordered |
| sales | float | Total sales value of the order |
| order_item_total | float | Total price of order item after discount |
| order_profit_per_order | float | Profit earned from the order |
| order_region | str | Region where order belongs |
| order_state | str | State where order was placed |
| order_status | str | Current status of the order |
| product_name | str | Name of the product |
| product_price | float | Price of the product |
| product_status | int | Status of product availability |
| shipping_date | date | Date and time when order was shipped |
| shipping_mode | str | Method of shipping used |
| delivery_days | int | Number of days taken to deliver the order |
| order_month | int | Month when the order was placed |
| delay_flag | int | Indicates if delivery was delayed (0 = No, 1 = Yes) |
| sales_log | float | Log transformed value of sales |
| profit_log | float | Log transformed value of profit |
| order_day | int | Day of the month when order was placed |
| order_week | int | Week number of the year |
| is_weekend | bool | Indicates if order was placed on weekend |

## 3. Purpose of the Dataset
The dataset can be used to analyze supply chain performance, order processing, customer behavior, shipping efficiency, and sales trends. It supports data analysis and decision-making in logistics and retail operations.

## 4. Conclusion
This data dictionary provides a clear explanation of each column in the dataset. Proper documentation improves data understanding and helps teams work more efficiently during analysis and reporting.
