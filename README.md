
# Machine Learning Volatility Modeling: An Empirical Study On Swiss Listed Companies

## Università della Svizzera italiana (USI)
### Faculty of Economics

---

## About

This repository contains the Master Thesis conducted as part of the Master’s Degree Programme in Finance - Quantitative Finance at the Università della Svizzera italiana. The thesis, titled "Machine Learning Volatility Modeling: An Empirical Study On Swiss Listed Companies", explores the application of machine learning (ML) and econometric models for forecasting financial volatility. The primary focus is on evaluating whether ML models provide marginal performance advantages over traditional econometric models, considering the additional efforts required for implementation.

---

## Research Objectives

The primary research question addressed in this study is:

*Does machine learning in volatility forecasting yield marginal performance advantages over econometric models, taking into consideration the additional efforts required for its implementation?*

To answer this question, the research conducts a comprehensive analysis and comparison of various models. These models fall into two main categories:

1. **Heterogeneous Autoregressive (HAR) Models:** Various iterations of the HAR model are implemented to assess their forecasting performance.

2. **Neural Networks (NNs):** Different architectures of neural networks are employed to investigate their effectiveness in predicting financial volatility.
   
3. **XGboost** 

---

## Dataset

The research utilizes an unprecedented dataset in the field, comprising price observations at a 10-minute frequency for 17 companies listed on the SIX Swiss Stock Exchange and included in the Swiss Market Index (SMI). The dataset also incorporates additional market variables and macroeconomic factors to enhance the volatility forecast through a multivariate setting. Neural network models take partial advantage of this approach. 

### List of Variables

The following table provides an overview of the key variables used in the analysis:

| Data Category | Name | Data | Frequency | Source |
| --- | --- | --- | --- | --- |
| Trading Variables | PRICE | Trade Price on the exchange. Returns and Realized Volatility are calculated from prices. | 10 min | BMLL, SIX |
|  | AVERAGE_BID_ASK_SPREAD_% | Average of all bid/ask spreads as a percentage of the mid-price | 1 day | Bloomberg |
|  | PX_VOLUME | Total number of shares traded on a security on the current day | 1 day | Bloomberg |
|  | RSK_BB_IMPLIED_CDS_SPREAD | Bloomberg Issuer Default Risk Implied CDS Spread | 1 day | Bloomberg |
|  | HIST_PUT_IMP_VOL | At-the-money put implied volatility based on the Listed Implied Volatility Engine (LIVE) calculator | 1 day | Bloomberg |
|  | NEWS_SENTIMENT_DAILY_AVG | Value of news sentiment for the parent company over an 8-hour period, value range [-1, 1] | 1 day | Bloomberg |
| Market Data | VSMI | Volatility Index of SMI (VSMI) | 1 day | SIX |
|  | CHFEUR | First difference CHF-EUR exchange rate | 1 day | Bloomberg |
|  | CHFUSD | First difference CHF-USD exchange rate | 1 day | Bloomberg |
|  | GSWISS10 | First difference 10y Swiss Gov Bond Yield | 1 day | Bloomberg |
|  | SFSNTC | First difference of Swiss OIS | 1 day | Bloomberg |
|  | SSARON | Swiss Average Rate O/N Intraday Value, volume-weighted reading based on CHF repo transactions | 1 day | Bloomberg |
|  | EURCHFV3M | EURCHF 3 Month ATM Implied Volatility | 1 day | Bloomberg |
|  | USDCHFV3M | USDCHF 3 Month ATM Implied Volatility | 1 day | Bloomberg |
| Macro Variables | CCFASZE | Switzerland Financial Market Survey - ZEW economic expectations | 1 day | Bloomberg |

---

### List of Equities 

| Short Name          | Ticker | ISIN          | Short Name          | Ticker | ISIN          |
|---------------------|--------|---------------|---------------------|--------|---------------|
| ABB LTD-REG         | ABBN   | CH0012221716  | PARTNERS GROUP J    | PGHN   | CH0024608827  |
| CIE FINANCI-REG     | CFR    | CH0210483332  | ROCHE HLDG-GENUS    | ROG    | CH0012032048  |
| GEBERIT AG-REG      | GEBN   | CH0030170408  | SWISSCOM AG-REG     | SCMN   | CH0008742519  |
| GIVAUDAN-REG        | GIVN   | CH0010645932  | SIKA AG             | SIKA   | CH0418792922  |
| HOLCIM LTD          | HOLN   | CH0012214059  | SONOVA HOLDING AG   | SOON   | CH0012549785  |
| KUEHNE + NAGEL REG  | KNIN   | CH0025238863  | SWISS RE AG         | SREN   | CH0126881561  |
| LONZA GROUP -REG    | LONN   | CH0013841017  | UBS GROUP AG        | UBSG   | CH0244767585  |
| NESTLE SA-REG       | NESN   | CH0038863350  | ZURICH INSURANCE    | ZURN   | CH0011075394  |
| NOVARTIS AG-REG     | NOVN   | CH0012005267  |                     |        |               |


## Thesis Details

- **Title:** Machine Learning Volatility Modeling: An Empirical Study On Swiss Listed Companies
- **Type:** Master Thesis
- **Degree Programme:** Finance - Quantitative Finance
- **Institution:** Università della Svizzera italiana
- **Faculty:** Faculty of Economics

---

### Note:

- For any inquiries or collaboration opportunities, please contact Pietro Bonazzi at [pietro.bonazzi@usi.ch].
- Data is available upon request. 


