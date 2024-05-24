import requests
import json

"""
'Customer_Age': long (required),
'Gender': string (required), 
'Dependent_count': long (required), 
'Education_Level': string (required), 
'Marital_Status': string (required), 
'Income_Category': string (required), 
'Card_Category': string (required), 
'Months_on_book': long (required), 
'Total_Relationship_Count': long (required), 
'Months_Inactive_12_mon': long (required), 
'Contacts_Count_12_mon': long (required), 
'Credit_Limit': double (required), 
'Total_Revolving_Bal': long (required), 
'Avg_Open_To_Buy': double (required), 
'Total_Amt_Chng_Q4_Q1': double (required), 
'Total_Trans_Amt': long (required), 
'Total_Trans_Ct': long (required), 
'Total_Ct_Chng_Q4_Q1': double (required), 
'Avg_Utilization_Ratio': double (required)]'
"""
payload = json.dumps({
    'inputs':{
    'Customer_Age': 45,
    'Gender': "M", 
    'Dependent_count': 3, 
    'Education_Level': "High School", 
    'Marital_Status': "Married", 
    'Income_Category': "$60K - $80K", 
    'Card_Category': "Blue", 
    'Months_on_book': 39, 
    'Total_Relationship_Count': 5, 
    'Months_Inactive_12_mon': 1, 
    'Contacts_Count_12_mon': 1, 
    'Credit_Limit': 12691, 
    'Total_Revolving_Bal': 777, 
    'Avg_Open_To_Buy': 11914, 
    'Total_Amt_Chng_Q4_Q1': 1.335, 
    'Total_Trans_Amt': 1144, 
    'Total_Trans_Ct': 42, 
    'Total_Ct_Chng_Q4_Q1': 1.625, 
    'Avg_Utilization_Ratio': 0.061
        }
    })


health_check = requests.get("http://127.0.0.1:5000//health")
print(health_check)

response = requests.post(
    url=f"http://127.0.0.1:5000//invocations",
    data=payload,
    headers={"Content-Type": "application/json"},
)
print(response.json())

