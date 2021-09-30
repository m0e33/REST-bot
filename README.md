# REST-bot

## Getting Started
setup for training.
setup for inference.
prerequesites for inference.

## Abstract
Stock trend forecasting, aiming at predicting the stock future trends, is crucial for investors to seek maximized profits from the stock market. Many event-driven methods utilized the events extracted from news, social media, and discussion board to forecast the stock trend in recent years. However, existing event-driven methods have some shortcomings, one of which is overlooking the influence of event information differentiated by the stock-dependent properties. Our model tries to prevent exactly that by learning the behavior of stocks in different contexts.

## Problem
We want to predict stock prices using event data. Given the stock-specific information (e.g., the textual information from news and social media, the historical stock price) of stock <a href="https://www.codecogs.com/eqnedit.php?latex=s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /></a> at date <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a>, the goal of stock trend forecasting is to forecast the stock price trend <a href="https://www.codecogs.com/eqnedit.php?latex=d^t_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d^t_i" title="d^t_i" /></a>.

Here, we define the stock price trend for stock <a href="https://www.codecogs.com/eqnedit.php?latex=s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /></a> at date <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a> as the stock price change rate of the next day:  
<a href="https://www.codecogs.com/eqnedit.php?latex=d_{i}^{t}=\frac{\text&space;{&space;Price&space;}_{i}^{t&plus;1}-\text&space;{&space;Price&space;}_{i}^{t}}{\text&space;{&space;Price&space;}_{i}^{t}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d_{i}^{t}=\frac{\text&space;{&space;Price&space;}_{i}^{t&plus;1}-\text&space;{&space;Price&space;}_{i}^{t}}{\text&space;{&space;Price&space;}_{i}^{t}}" title="d_{i}^{t}=\frac{\text { Price }_{i}^{t+1}-\text { Price }_{i}^{t}}{\text { Price }_{i}^{t}}"/></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=Price^t_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Price^t_i" title="Price^t_i" /></a> could be specified by different values, such as opening price, closing price and volume weighted average price (VWAP), and we use closing price in our work.

We also need to define what we mean with stock context. Stock context is defined as the combination of stock’s historical events and these events’ feedback, where event feedback is defined as the relative change of price on the stock that this event happened.

## 1. Motivation & Related Work
Our model is the paper "REST: Relational Event-driven Stock Trend Forecasting" by Wentao Xu and his colleagues, who built and evaluated a similar model. Our goal was to build a profitable strategy around a model that incorporates the stock context, so that the model predictions serve as an indicator (buy and sell signal) for the strategy.

In recent years, stock trend forecasting has attracted much attention because of its vital stock investment role. We can categorize most of the existing stock trend forecasting work into two categories: the Event-driven Methods and the Technical Analysis.

Technical analysis [10] is a category of methods for stock trend forecasting, which is orthogonal to event-driven methods. The technical analysis predicts the stock trend based on the historical time-series of market data, such as trading price and volume. This type of approach aims to discover the trading patterns that can leverage for future predictions. Technical analysis is not sensitive to the abrupt changes in stock price caused by external event information of stock [7], limiting its performance on the stock trend forecasting.

According to the efficient market hypothesis [24], people know that an event that happens on a stock would change the stock information of this stock, affecting its stock price. An event can have different sources, for example social media (Twitter, Reddit, Facebook), forums (Yahoo Finance Forum) or news (Google News, Press Releases, Bloomberg News). 

Paper examples for technical analysis:
- Autoregressive Model (linear apporach) [19]
- ARIMA model (linear apporach) [2]
- Deep learning [28, 35]
- LSTM [14]

Paper examples for event driven stock trend forcasting:
- Hierarchical attention mechanism [15]
- Knowledge-driven temporal convolutional network [7]
- Cross-model attention based Hybrid Recurrent Neural Network [39]

## 2. Approach
### 2.1. Data
For our model we need on the one hand price data (simple timeseries) and on the other hand stock news and press releases of the companies. For the companies we have chosen the 50 companies with the largest market capitalization (to train the model). All our data can be obtained from the API of [financialmodelingprep](https://financialmodelingprep.com/developer). A big limitation is that we can only obtain data for the last 2 years. These datasets are stored as simple time series as shown in the following:

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

[19] Lili Li, Shan Leng, Jun Yang, and Mei Yu. 2016. Stock Market Autoregressive Dynamics: A Multinational Comparative Study with Quantile Regression. Math- ematical Problems in Engineering 2016 (2016).

[2] Adebiyi A Ariyo, Adewumi O Adewumi, and Charles K Ayo. 2014. Stock price prediction using the ARIMA model. In 2014 UKSim-AMSS 16th International Conference on Computer Modelling and Simulation. IEEE, 106–112.

[28] Jigar Patel, Sahil Shah, Priyank Thakkar, and Ketan Kotecha. 2015. Predicting stock market index using fusion of machine learning techniques. Expert Systems with Applications 42, 4 (2015), 2162–2172.

[35] Jonathan L Ticknor. 2013. A Bayesian regularized artificial neural network
for stock market forecasting. Expert Systems with Applications 40, 14 (2013),
5501–5506.

[14] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9, 8 (1997), 1735–1780.

[15] ZiniuHu,WeiqingLiu,JiangBian,XuanzheLiu,andTie-YanLiu.2018.Listening to chaotic whispers: A deep learning framework for news-oriented stock trend prediction. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. ACM, 261–269.

[39] Huizhe Wu, Wei Zhang, Weiwei Shen, and Jun Wang. 2018. Hybrid Deep Se-
quential Modeling for Social Text-Driven Stock Prediction. In Proceedings of the 27th ACM International Conference on Information and Knowledge Man- agement, CIKM 2018, Torino, Italy, October 22-26, 2018. 1627–1630. https: //doi.org/10.1145/3269206.3269290
