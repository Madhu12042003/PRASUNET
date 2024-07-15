# Property Price Prediction

## Dataset

This novel dataset is collected from real
estate advertisements published in *www.NoBroker.com* for the purpose of this project. It includes a
comprehensive compilation of almost **2000** property listings
gathered from more than **60** localities dispersed throughout the metropolitan city of
Bangalore, Karnataka, India. Data extraction was limited to
publicly available listing details and refrained from accessing
any proprietary or restricted content.

The 26 features used for the purpose of this
study are - price, sqft, price per sqft, age, locality, bedrooms,
bathrooms, balcony, parking, security, flooring, furnishing,
facing, children friendly, clubhouse, swimming pool, gym, lift,
internet, fire safety, intercom, gas provision, park, shopping
complex, sewage provision, and visitor friendly. A derived
property called **price per square foot** is calculated by dividing
the property’s price by the property’s total are in square feet. 

## Preprocessing

The features of about 25–30 listings from each locality were
first taken out and put into a CSV file. These files were later
consolidated, and redundant entries were eliminated to ensure
dataset coherence. Data cleaning methods were applied to rectify any inaccuracies or omissions. 

Amenities list
was split into separate columns and their presence are encoded
as binary values (1 for presence, 0 for absence), streamlining
the analysis of property amenities. These preprocessing steps
were instrumental in refining the dataset, enhancing its quality
and usability for subsequent analysis and modeling endeavors.

## Property Price Prediction

 A Linear Regression model is then trained on the standardized training data. The model's predictions are evaluated using Root Mean Squared Error (RMSE) and the R² score, which provide insights into the model's accuracy and explanatory power. This approach ensures a systematic and efficient prediction of property prices based on the available features.

 The model results in a Root Mean Square Error(RMSE) of 0.012 and Linear Regression score i.e. R² score of 0.84. This suggests that Linear Regression model is a reliable predictor for this dataset.
