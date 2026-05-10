import pandas as pd

import os



# 1. ระบุชื่อไฟล์ (กรณีอยู่ในโฟล์เดอร์เดียวกับโค้ด ไม่ต้องใส่ Path ยาว)

file_name = 'owid-energy-data.csv'



# 2. อ่านข้อมูล

# ถ้าเป็น CSV

df = pd.read_csv(file_name)



# ถ้าเป็น Excel (.xlsx) ให้ใช้คำสั่งนี้แทน:

# df = pd.read_excel('owid-energy-data.xlsx')



# 3. Filter ข้อมูลตามกลยุทธ์ที่เราวางไว้ (Thailand + ตั้งแต่ปี 2010)

th_energy = df[(df['country'] == 'Thailand') & (df['year'] >= 2010)].copy()



# 4. เลือกเฉพาะ Columns ที่สำคัญต่อการทำ Synthetic Sensing (Track A)

target_columns = [

'year', 'country',

'oil_consumption', # ปริมาณการใช้น้ำมัน (Baseline หลัก)

'oil_share_energy', # % ความพึ่งพาน้ำมัน (Sensitivity Index)

'energy_per_gdp', # ประสิทธิภาพ (Vulnerability Index)

'fossil_fuel_consumption'

]



th_energy = th_energy[target_columns]



# 5. ตรวจสอบข้อมูลเบื้องต้น

print(th_energy.head())

print(th_energy.info())

