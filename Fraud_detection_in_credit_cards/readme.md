<p style = "font-size : 42px; color : #458B00 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #FF9912; border-radius: 5px 5px;"><strong>Fraud Detection on Credit Cards</strong></p>

![pie_plot](images/Ecommerce.jpg?raw=true)

# 1. Datset
The dataset consists in 1 year of historical transactional data and fraud flags. The bank is interested in a model which predicts the likelihood that a transaction is later marked as fraud. 
Unfortunately this dataset is not open source. However, I could consider to share the dataset by means of a formal request. In such a case please send a request email to [@yuniorcabrales](yuniorcabrales@gmail.com). I hope the procedure used here could serve as inspiration for readers when analyzing similar datasets.

## 1.1. Data Dictionary

| Column                                                                                                                                                                                                          | Explanation                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| transactionTime                                                                                                                                                                                                     | The time the transaction was requested.                                                                              |
| eventId                                                                                                                                                                                                      | A unique identifying string for this transaction                                                                                   |
| accountNumber                                                        | The account number which makes the transaction                                                                               |
| merchantId                                                          | A unique identifying string for this merchant                                  |
| mcc                                                              | The merchant category code of the merchant                                                                              |
| transactionAmount                                                                                                                                                                                               | The value of the transaction in GBP                                                                     |
| posEntryMode                                                                                                                                                                                         | The Point Of Sale entry mode                                                                           |
| availableCash                                                                                                                                                                                  | The (rounded) amount available to spend prior to the transaction                                                                        |
| merchantCountry                                                                                                                                                                                     | A unique identifying string for the merchant's country                                                                                     |
| merchantZip                                                                                                                                                                                       | A truncated zip code for the merchant's postal region    |
|                                                      
posEntryMode_Values                                                                                    |
| 00                                                                                                                                                                                              | Entry Mode Unknown                                                                       |
| 01                                                                                                                                                                                                 | POS Entry Mode Manual                                                                         |
| 02                                                                                                                                                                                                      | POS Entry Model Partial MSG Stripe                                                                 |
| 05                                                                                                                                                                                                    | POS Entry Circuit Card                                                                |
| 07                                                                                                                                                                                                         | RFID Chip (Chip card processed using chip)                                                                   |
| 80                                                                                                                                                                                                         | Chip Fallback to Magnetic Stripe                                                                        |
| 81                                                                                                                                                                                                         | POS Entry E-Commerce                                                                        |
| 90                                                                                                                                                                                                    | POS Entry Full Magnetic Stripe Read                                                                    |
| 91                          | POS Entry Circuit Card Partial                                                                |               


# 2. Project Motivation:

The goal of the project is to predict fraudulent transactions in order to banks to make decisions on time when risky-looking purchases are detected. First of all we did an exploratory data analysis and proceed to test eight different models. Because the dataset is highly imbalanced we used the SMOTE function from imbalanced-learn library to overcome the data imbalance. Although all the tested models showed a good accuracy score, Five of were above 0.99.

# 3. Data Insights
* 72% of the fraudulent transactions are performed by E-Commerce, while 24% are performed by manual mode (see dictionary). The rest are less frequent (4% all together)

* It can be noticed that 81% of fraudulent transactions are performed on three different countries. ~36% of the fraudulent transactions are performed the country whose code is 826 (Not clear on dictionary). 29% for country with code 840, while ~16% for the one with 442 country code. Other countries have less than 1% of incidence on average 

![pie_plot](images/img1.png?raw=true)

* Although all the tested models showed a good accuracy score, Five of were above 0.99.

![plot](images/img2.png?raw=true)

# 4. Libraries

Pandas, Numpy, imblearn, Sklearn, Seaborn, matplotlib, keras