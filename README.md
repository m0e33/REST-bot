# REST-bot

## Getting Started
setup for training.
setup for inference.
prerequesites for inference.
## Abstract
Stock trend forecasting, aiming at predicting the stock future trends, is crucial for investors to seek maximized profits from the stock market. Many event-driven methods utilized the events extracted from news, social media, and discussion board to forecast the stock trend in recent years. However, existing event-driven methods have some shortcomings, one of which is overlooking the influence of event information differentiated by the stock-dependent properties. Our model tries to prevent exactly that by learning the behavior of stocks in different contexts. 

## 1. Motivation & Related Work
Our model is the paper "REST: Relational Event-driven Stock Trend Forecasting" by Wentao Xu and his colleagues, who built and evaluated a similar model. Our goal was to build a profitable strategy around a model that incorporates the stock context, so that the model predictions serve as an indicator (buy and sell signal) for the strategy.

In recent years, stock trend forecasting has attracted much attention because of its vital stock investment role. We can categorize most of the existing stock trend forecasting work into two categories: the Event-driven Methods and the Technical Analysis.

Technical analysis [10] is a category of methods for stock trend forecasting, which is orthogonal to event-driven methods. The technical analysis predicts the stock trend based on the historical time-series of market data, such as trading price and volume. This type of approach aims to discover the trading patterns that can leverage for future predictions. Technical analysis is not sensitive to the abrupt changes in stock price caused by external event information of stock [7], limiting its performance on the stock trend forecasting.

According to the efficient market hypothesis [24], people know that an event that happens on a stock would change the stock information of this stock, affecting its stock price. An event can have different sources, for example social media (Twitter, Reddit, Facebook), forums (Yahoo Finance Forum) or news (Google News, Press Releases, Bloomberg News). 

XXX (Hier kann man auch aus den Paper nochmal für jeden Absatz paar Beispielpaper raussuchen)

## 2. Approach
### 2.1. Data
For our model we need on the one hand price data (simple timeseries) and on the other hand stock news and press releases of the companies. For the companies we have chosen the 50 companies with the largest market capitalization (to train the model). All our data can be obtained from the API "Financialmodelingprep". A big limitation is that we can only obtain data for the last 2 years. These datasets are stored as simple time series as shown in the following:

![](https://github.com/m0e33/REST-bot/blob/report/assets/image1.jpg?raw=true)

The data preprocessing and the associated data format are quite complicated. The input for a forward pass can be described quite well by the following picture:

![](https://github.com/m0e33/REST-bot/blob/report/assets/image2.jpg?raw=true)

The input for a forward pass consists of 5 dimensions. First, we have a window of 30 days in dimension one. These are later separated in the model between the first 27 days, which is the stock context, and the last 3 days, which is practically the current information situation about the companies. A day now includes the event information from all symbols. Since we are looking at 50 companies, the 2nd dimension consists of a list of symbols. For each of these symbols, there is a set of events that are stored for a symbol (dimension 3). Each event has an event type, with which we distinguish between different news types ( just stock news or press releases) and a set of words (dimension 4):

![](https://github.com/m0e33/REST-bot/blob/report/assets/image3.jpg?raw=true)

The event type and the event words are word embeddings of length 300 (dimension 5). This can also be seen in the following diagram:

![](https://github.com/m0e33/REST-bot/blob/report/assets/image4.jpg?raw=true)

So, in summary, we have the following input shapes: [30, 50, 10, 50, 300].

![](https://github.com/m0e33/REST-bot/blob/report/assets/image5.jpg?raw=true)

The ground truth looks much simpler. Since we only want to predict the relative price range of the next day for each symbol, this is only 3 dimensions. 

![](https://github.com/m0e33/REST-bot/blob/report/assets/image6.jpg?raw=true)

To create the training data, we build sliding windows with a stride of 1. Since we work with daily prices, we have a window with the dimensions described above for each day of our training time series. The resulting data set is enormous, as the following calculation shows:

![](https://github.com/m0e33/REST-bot/blob/report/assets/image7.jpg?raw=true)


### 2.2. Architecture
see slides
## 3. Evaluation
Training stats
## 4. Conclusion
spannendes projekt, 
## 5. Future Work
### 5.1 Graph Convolution
### 5.2 Different Contexts
Crypto currencies, andere Zeitauflösung

---

[7] Shumin Deng, Ningyu Zhang, Wen Zhang, Jiaoyan Chen, Jeff Z Pan, and Huajun Chen. 2019. Knowledge-Driven Stock Trend Prediction and Explanation via Temporal Convolutional Network. In Companion Proceedings of The 2019 World Wide Web Conference. ACM, 678–685.

[10] Robert D Edwards, John Magee, and WH Charles Bassetti. 2018. Technical analysis of stock trends. CRC press. 

[24] Burton G Malkiel and Eugene F Fama. 1970. Efficient capital markets: A review
of theory and empirical work. The journal of Finance 25, 2 (1970), 383–417
