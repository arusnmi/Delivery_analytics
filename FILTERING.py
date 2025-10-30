import pandas as pd


dataset=pd.read_csv('Last mile Delivery Data.csv')



filt_dataset1=dataset[dataset['Store_Latitude']!= 0]
filt_dataset2=filt_dataset1[filt_dataset1['Store_Longitude']!= 0]
filt_dataset3=filt_dataset2[filt_dataset2['Agent_Rating']!= 6]



filt_dataset3.dropna(inplace=True)

filt_dataset3.to_csv('no_na_Last_mile_Delivery_Data.csv', index=False)


# print(filt_dataset2.describe())
# print(filt_dataset2.info())