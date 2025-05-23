from imp_data import load_maastricht_data  # adjust path / package name as needed
data = load_maastricht_data()
print(data.keys())
# what I want to do 
# create a deamande forcasting model
# this model will consider each service point as a cluster of customers
# the model will predict the demand for each service point using logistic regression
