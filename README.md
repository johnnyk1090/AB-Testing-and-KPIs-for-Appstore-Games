<img src="assets/banner.svg" width="100%" alt="App-Store Games: KPIs & A/B Tests. What makes a game succeed? Sentiment, engagement and monetisation, tested properly and turned into decisions.">

## The question

This project identifies the **key performance indicators** in a games app-store dataset and builds a **predictive model for game prices**. The goal is insight for the business teams responsible for decisions, with techniques that apply to any marketing dataset.

## The chapters

1. User sentiment analysis
2. Most popular platform metric
3. Language support metric
4. User engagement: playtime
5. NLP for gaming categories
6. Monetisation analysis: distribution of game prices
7. Monetisation analysis: price and engagement metrics
8. Monetisation analysis: DLC count and user engagement
9. Isolation Forest for outlier detection
10. Traditional A/B testing
11. Dimensionality reduction
12. Predictive modelling
13. Containerisation

## Methods

- Interactive dashboards with Plotly
- Regression with an **Extra Trees** regressor
- **Natural language processing** on game descriptions
- **Isolation Forest** outlier detection
- **Shapiro-Wilk** normality testing and the **Mann-Whitney U** test for A/B comparisons
- **Principal component analysis**

## Figures

<table>
<tr><td width="50%" valign="top"><img src="graphs/Distribution_of_prices.png" width="100%" alt="Distribution of game prices"><br><sub>Distribution of game prices</sub></td><td width="50%" valign="top"><img src="graphs/DLC%20Count%20vs.%20Positive%20Comments%20with%20Estimated%20Owners%20as%20Hue.png" width="100%" alt="DLC count vs positive comments, coloured by estimated owners"><br><sub>DLC count vs positive comments, coloured by estimated owners</sub></td></tr>
<tr><td width="50%" valign="top"><img src="graphs/Average_playtime.png" width="100%" alt="Word cloud of categories for average playtime"><br><sub>Word cloud of categories for average playtime</sub></td><td width="50%" valign="top"><img src="graphs/Variance_explained_by_components.png" width="100%" alt="Variance explained by number of principal components"><br><sub>Variance explained by number of principal components</sub></td></tr>
<tr><td width="50%" valign="top"><img src="graphs/heatmap_corr.png" width="100%" alt="Correlation heatmap"><br><sub>Correlation heatmap</sub></td><td width="50%" valign="top"><img src="graphs/Game_sentiment_ratio.png" width="100%" alt="Sentiment ratio per game"><br><sub>Sentiment ratio per game</sub></td></tr>
</table>

## Run it

1. Download `requirements.txt`.
2. Follow the instructions in [`Conda_Virtual_Env.txt`](Conda_Virtual_Env.txt).

`requirements.txt` is a full freeze of the original environment, and some entries point to local conda builds. If pip cannot resolve them, install the packages listed under *Built with* below.

A [`Dockerfile`](docker/Dockerfile) is included for running the project in a container.

## Built with

`Python` · `NumPy` · `pandas` · `Matplotlib` · `Seaborn` · `SciPy` · `Plotly` · `scikit-learn` · `Joblib` · `WordCloud` · `LabelEncoder` · `PCA` · `delayed functions` · `itertools` · `Pingouin` · `PyCaret`

<br>

<p align="center"><sub><a href="https://github.com/johnnyk1090">← back to the profile</a></sub></p>
