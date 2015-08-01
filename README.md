#Yu Solution for Avito Context Ad Clicks

This reposition is the archive repository for my solutions for Kaggle competition [Avito Context Ad Clicks](https://www.kaggle.com/c/avito-context-ad-clicks). Before this, there has been several competitions regarding click through rate(CTR). And there are several methods have proved to be effective for this kind of problem. (Factorization machine)FM, vowpal wabbit and FTRL(regularized logistic regression). Here I tried FTRL with one-hot encoded features. Final score is 56 among 414, with log loss of 0.045. Top score is close to 0.040. There are many things learned from this competition, in summary,

- Online learning or model updating with stream data. Since the training data has total of over 390 million rows and 190 million context ads, loading the whole data into RAM to train is not possible. Logistic regression is easy and fast to train and regularization can produce sparsity. More detail about the model can look to Google's paper for reference.

- How to select feature/ what is the relevant informations. A lot of insights were shared on the [Forum](https://www.kaggle.com/c/avito-context-ad-clicks/forums/t/15606/congratulations). Including how to combine information from the the same search(All the displayed Ads in a search), how to order the data by date(which I tried but couldn't get it to work) since information is very time dependent.

- And finally due to the long training time(one day for me). Methods to speed up the process are needed. Due to my server constrain(python was not built with sqlite3), I used pandas for loading data. People reported speedup from using PyPy. And as a result, I didn't do any validation. There are many ideas about how to validate the model(Sort data by date). Since the model is fairly simple and the dataset is huge, public score is a good metric.

All in all, the limited skill in code organization and the slow iteration process are the biggest issues in the competition. Improving efficiency through better pipelining would be the most valuable investment for future's tasks. 
  