# REST BOT - Predicting stock price trends based on news and press releases.

## Getting Started

### Prerequisites
To contribute to this project, train the model or try inference, we recommend to work with python 3.7 and a virtual environment.
So first, create a virtual environment, activate it and install the required dependencies.

```console
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

This project was planned to run on a bakdata kubeflow cluster within the google cloud environments, and so, currently  - even if not using the cloud environment - you have to set the google cloud authentication environment variables when running the main script.

```console
export GOOGLE_APPLICATION_CREDENTIALS=./gcp-bakdata-kubeflow-cluster.json
```

### Training

For pre-set training with our configurations, you can simply call the main script and the application will download the necessary data, build the datasets and train the model.

```
python3 main.py
```

If you want to configure your own training parameters, you have to take a look into the `configuration/confiuguration.py`. There you can find the training parameter as well as the hyper parameter configuration for training.
Within the `train_model()` method of the `model/rest_kubeflow_adapter.py` you can find the data configuration, which declares parameter that shape the dataset that is built when running training.

The data configuration needs a list of symbols the model is trained on. This list of symbols can be defined in the `symbols.yml`

We provide a sample configuration here. All of these settings will become clear when reading the following report.

```python
data_cfg = DataConfiguration(
    symbols=load_symbols(limit=None),
    start="2019-01-01",
    end="2021-01-01",
    feedback_metrics=["open", "close", "high", "low", "vwap"],
    stock_news_fetch_limit=20000,
    events_per_day_limit=10
)

train_cfg = TrainConfiguration(
    val_split=0.2,
    test_split=0.1,
    batch_size=8
)

hp_cfg = HyperParameterConfiguration(
    num_epochs=1000,
    attn_cnt=4,
    lstm_units_cnt=80,
    sliding_window_size=10,
    offset_days=2
)

```

### Inference
Currently, due to hardware constraints (RAM), we are not able to provide a model with decent performance. However, all the infrastructure for model inference has been integrated. 
Mandatory for inference to work is a _successfull_ training run, whether it trained the model up to a good test loss value or not. A successfull training run yields the following assets:
- Serialized configurations, which determin the configuration for the model to load
- Serialized model weights, the core for inference
- Serialized word-embedding weights, an important part for preprocessing.
#### Locally
If you want to play around locally with inference, you can use the simple `inference_test.py` script and adapt the parameter given there.

#### Server
To run inference from a cloud machine, we provide a simple flask server with an inference endpoint.
For setting this up you have to export the required flask environment variables:
```console
export FLASK_ENV=development
export FLASK_APP=rest_bot_server.py
```

After that you can start the server by running:

```console
flask run --host=0.0.0.0
```

The server now listens on port 5000 for requests. A sample request could be:

```
http://127.0.0.1:5000/inference?date=2021-08-01&symbols=TSLA-AMZN
```

This will give you the relative price change prediction for the tesla and amazon stock on the first of August, 2021.

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

The ground truth looks much simpler. Since we only want to predict the relative price change of the next day for each symbol, this is only 3 dimensions. 

![](https://github.com/m0e33/REST-bot/blob/report/assets/image6.jpg?raw=true)

To create the training data, we build sliding windows with a stride of 1. Since we work with daily prices, we have a window with the dimensions described above for each day of our training time series. The resulting data set is enormous, as the following calculation shows:

![](https://github.com/m0e33/REST-bot/blob/report/assets/image7.jpg?raw=true)


### 2.2. Architecture

The models architecture is highly custom with each part serving its own unique function.
It can be divided into four parts: An event-information encoder, a stock-context encoder, a stock-dependent influence 
and a prediction layer. The original paper includes one more module thats part of this architecture, which we haven't been able to incorporate yet, and for now briefly describe it in the future work section. 
Each of these layers work on all symbols and their respective events simultaneously, however,
in the following sub-sections, we go through them from the perspective of one symbol.

![](assets/architecture-overview.png?raw=true)

#### 2.2.1 Event Information Encoder

The event information encoders job is to build an expressive representation of the information that exists right
as the day we want to predict takes place. It therefore utilizes an attention mechanism as well as a small LSTM net.

![](assets/architecture-event-information-encoder.png?raw=true)

The input to this component are the last three days with their events for one symbol.
The attention mechanism then gives different attention values to each word in the events content, with regards to the event type (PRESS or NEWS etc.). Configurable here are the number of attention heads, (ref attention) and the nunmber of hidden LSTM units.
The LSTM then flattens the sequence of weighted events over the last three days into one single dimension representation vector.
It thereby learns and incorporates how long one event's effect lasts on the price development. 
An event that has occurred three days ago might not have the same impact as an event which took place yesterday.

#### 2.2.2 Stock Context Encoder
Parallel to the event-information encoder, works the stock context encoder. Its job is to encode the stocks events over the past 30 days alongside its price development over this time span.
It uses the same mechanics as the event-information encoder - an attention mechanism and a series of LSTMs.
The events of the past 30 days are weighted with learned attention filters, and then passed through an LSTM - as is the stocks price development information - where they are squashed into one single dimension representing information about the kind of events, how long they lasted and their respective price change.

#### 2.2.3 Stock Dependent Influence

![](assets/architecture-stock-dependent-influence.png?raw=true)

The stock dependent influence now takes the event information for the current day and the event information over the past 30 days as well as their impact on the stock's price change, and calculates / learns the impact of the current event information on the current price change.
It does so, by feeding the two representations through one dense layer. The output is a representation of the strength of the impact of current day's information landscape concerning the symbol on the price development of this symbol.

#### 2.2.4 Prediction
The final layer now takes care of converting the strength of the impact into a real relative price change prediction.

![](assets/architecture-price-prediction.png?raw=true)


### 2.3 Training
We implemented a distributed strategy to utilize all CPU / GPU devices provided by the host for training. This can be experienced in this [screencast]({https://github.com/m0e33/REST-bot/blob/main/assets/training_run_memory_leak_h.mp4} "Link Title")
.

#### 2.3.1 Hardware Limits
As seen in 2.1, the dataset for as short as two years of training data (the paper uses eight years), can get very demanding, when loaded all at once into memory.
We therefore implemented a streaming solution for the sliding window systematic of this dataset. In theory, this should lead to only ~14 GB for two years of events data residing at the same time in memory, 
while the windows into this timeframe are generated during training and released from memory after training. In practice, we experience a slow increase in memory consumption, which leads to even very potent machines killing the training process. This also can be experienced in the screencast linked above.
We found, that decreasing the sliding window size, or the amount of days which make up the stock context information, leads to longer training time before beeing killed.
The furthest we got with an acceptable data configuration was eleven epochs, which took around 45 minutes per epoch.

![](assets/training_step_duration.png?raw=true)
![](assets/training_train_loss.png?raw=true)



## 3. Evaluation
For the reasons explained in section 2.3.1, we are currently not able to train a model until sufficient and performant loss values. However, we have implemented a way of evaluating a model, if it happens to be sufficiently trained.
We implemented a simple threshold strategy (again, the paper "REST: Relational Event-driven Stock Trend Forecasting" was used as a model). This takes the model-predictions as a buy indicator and works as follows. Since our data is at daily granularity, the strategy first sorts the predictions for all symbols in a day. It then buys the first k symbols (those with highest predicted relative price change on the next day) (if an open position does not already exist), and sells the remaining symbols (if an open position exists). At the end of a strategy run, all open positions are closed. This strategy can be used for classical backtesting as well as for paper trading. We calculate an additional 0.15% fee for a buy transaction and 0.25% fee for a sell transaction. 

## 4. Future Work
### 4.1 Graph Convolution
A possible extension would be to learn the mutual influence of stocks with a graph convolution and to include it in the calculation. The paper implementation has shown that even better results can be achieved. The basis of the Graph Convolution would be to place the companies in different relations, for example relations like "same industry", "same shareholder", upstream and downstream relations etc. Through the different relations a stock graph can be built, which is defined as follows: 

Stock graph is defined as a directed graph <a href="https://www.codecogs.com/eqnedit.php?latex=G=\langle\mathcal{S},&space;\mathcal{R},&space;\mathcal{A}\rangle" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G=\langle\mathcal{S},&space;\mathcal{R},&space;\mathcal{A}\rangle" title="G=\langle\mathcal{S}, \mathcal{R}, \mathcal{A}\rangle" /></a>, where S denote the set of stocks in the market and R is the set of relations between two stocks. <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{A}" title="\mathcal{A}" /></a> is the set of adjacent matrices. For an adjacent matrix <a href="https://www.codecogs.com/eqnedit.php?latex=A^{r}&space;\in&space;\mathcal{A}(r&space;\in&space;\mathcal{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A^{r}&space;\in&space;\mathcal{A}(r&space;\in&space;\mathcal{R}" title="A^{r} \in \mathcal{A}(r \in \mathcal{R}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=A^{r}&space;\in&space;\mathbb{R}|\mathcal{S}|&space;\times|\mathcal{S}|)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A^{r}&space;\in&space;\mathbb{R}|\mathcal{S}|&space;\times|\mathcal{S}|)" title="A^{r} \in \mathbb{R}|\mathcal{S}| \times|\mathcal{S}|)" /></a> of relation <a href="https://www.codecogs.com/eqnedit.php?latex=r,&space;A_{i&space;j}^{r}=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r,&space;A_{i&space;j}^{r}=1" title="r, A_{i j}^{r}=1" /></a> means there is a relation <a href="https://www.codecogs.com/eqnedit.php?latex=r" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r" title="r" /></a> from stocks <a href="https://www.codecogs.com/eqnedit.php?latex=s_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_j" title="s_j" /></a> to stock <a href="https://www.codecogs.com/eqnedit.php?latex=s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=A_{i&space;j}^{r}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A_{i&space;j}^{r}=0" title="A_{i j}^{r}=0" /></a> indicates there is no a relation <a href="https://www.codecogs.com/eqnedit.php?latex=r" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r" title="r" /></a> from stock <a href="https://www.codecogs.com/eqnedit.php?latex=s_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_j" title="s_j" /></a> to stock <a href="https://www.codecogs.com/eqnedit.php?latex=s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /></a>.

![](https://github.com/m0e33/REST-bot/blob/report/assets/image8.jpg?raw=true)

### 4.2 Different Assets, different Event Types
Another possible extension would be the use of a new data basis. So far, we have only used blue-chip stocks, so to speak, which experience shows are usually not very volatile and react less strongly to individual events than other assets. So we could also try to use stocks with smaller market capitalization or even a completely new asset, such as cryptocurrency. Since our model only works with a timeseries of prices and simple text data, one could just as well train the model on cryptocurrencies. It may be possible to make specific statements about which assets and investment variants are most influenced by events.

Another possibility would be to include additional news data. Other papers have already pointed out that social media data from Twitter, for example, can also contribute to model performance. With our model it is easy to add further event types. This would allow us to make statements about which event types have the greatest influence on the price of an asset. 

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
