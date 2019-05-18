# Finding Trend in Stock Market with RobustPCA

(For more details, please refer to the report.pdf)

Here we will use RobustPCA to decompose stock prices into a low rank component and sparse component. We regard the first one as the main trends of the stocks while the second one as their individual noise.

We will evalute this decompositon by converting the problem into a classification task. That is, we try to recognize the underlying classes (10 classes in total, e.g. Industries, IT, Healt Care, etc.) of the given stocks. 

This is based on the basic assumption that stocks from the same classes will have similar main trends while stocks from different classes will have different trends.

If our model can better characterize the trends of stocks, it will achieve higher accuracy in classification.

Here we use 10 stocks from IT to visualize how RobustPCA decompose stocks into low-rank component and sparse component.

![Decomposition of 10 Stocks from IT with RobustPCA](https://github.com/ZhicongLiang/SNP500-Stock-Trend/blob/master/figs/IT-decompose.jpg)
