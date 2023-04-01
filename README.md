# Red-Wine-Quality-ML-with-logistic-regression-and-RF-I-

# Introduction

This case study is using Wine Quality Data Set from the UCI machine learning repository, in which the two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
Our research will focus on the red wine's quality, and it will be divided into two parts.

**1** -will attempt to predict whether the wine is good or not.

**2** -will try to estimate the precise quality of the wine as it appears in the dataset.

every part can be read by it's own, and this is the part one of the study.

![Red_Wine_Glass](https://user-images.githubusercontent.com/41892582/227959684-1d4e9efc-56ff-440d-87a2-6483851273cc.jpg)

**Loading libraries**

Firstly I will start by loading some packages that I will use during the analysis

~~~
library(tidyverse)
library(Hmisc)
library(rms)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggcorrplot)
library(corrplot)
library(ppcor)
library(EFAtools)
library(splines)
~~~

**Getting the data**

~~~
wine<-read.csv("winequality-red.csv")
~~~

**Exploration of the data**

~~~
##the structure of the data
str(wine)
~~~

~~~
'data.frame':	1599 obs. of  12 variables:
 $ fixed.acidity       : num  7.4 7.8 7.8 11.2 7.4 7.4 7.9 7.3 7.8 7.5 ...
 $ volatile.acidity    : num  0.7 0.88 0.76 0.28 0.7 0.66 0.6 0.65 0.58 0.5 ...
 $ citric.acid         : num  0 0 0.04 0.56 0 0 0.06 0 0.02 0.36 ...
 $ residual.sugar      : num  1.9 2.6 2.3 1.9 1.9 1.8 1.6 1.2 2 6.1 ...
 $ chlorides           : num  0.076 0.098 0.092 0.075 0.076 0.075 0.069 0.065 0.073 0.071 ...
 $ free.sulfur.dioxide : num  11 25 15 17 11 13 15 15 9 17 ...
 $ total.sulfur.dioxide: num  34 67 54 60 34 40 59 21 18 102 ...
 $ density             : num  0.998 0.997 0.997 0.998 0.998 ...
 $ pH                  : num  3.51 3.2 3.26 3.16 3.51 3.51 3.3 3.39 3.36 3.35 ...
 $ sulphates           : num  0.56 0.68 0.65 0.58 0.56 0.56 0.46 0.47 0.57 0.8 ...
 $ alcohol             : num  9.4 9.8 9.8 9.8 9.4 9.4 9.4 10 9.5 10.5 ...
 $ quality             : int  5 5 5 6 5 5 5 7 7 5 ...
~~~

When turning the quality variable into a 0 (bad wine) or 1 (good wine), it is considered bad wine if the quality is 6 or lower and good wine if the quality is 7 or higher.

~~~
wine$quality<-ifelse(wine$quality>6,1,0)          
~~~

basic description of the data


~~~
##basic description of the data
describe(wine)
~~~

~~~
wine 

 12  Variables      1599  Observations
---------------------------------------------------------------------------
fixed.acidity 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0       96    0.999     8.32    1.893      6.1      6.5 
     .25      .50      .75      .90      .95 
     7.1      7.9      9.2     10.7     11.8 

lowest :  4.6  4.7  4.9  5.0  5.1, highest: 14.3 15.0 15.5 15.6 15.9
---------------------------------------------------------------------------
volatile.acidity 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0      143        1   0.5278    0.199    0.270    0.310 
     .25      .50      .75      .90      .95 
   0.390    0.520    0.640    0.745    0.840 

lowest : 0.120 0.160 0.180 0.190 0.200, highest: 1.180 1.185 1.240 1.330 1.580
---------------------------------------------------------------------------
citric.acid 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0       80    0.999    0.271   0.2227    0.000    0.010 
     .25      .50      .75      .90      .95 
   0.090    0.260    0.420    0.522    0.600 

lowest : 0.00 0.01 0.02 0.03 0.04, highest: 0.75 0.76 0.78 0.79 1.00
---------------------------------------------------------------------------
residual.sugar 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0       91    0.996    2.539    1.078     1.59     1.70 
     .25      .50      .75      .90      .95 
    1.90     2.20     2.60     3.60     5.10 

lowest :  0.9  1.2  1.3  1.4  1.5, highest: 13.4 13.8 13.9 15.4 15.5
---------------------------------------------------------------------------
chlorides 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0      153        1  0.08747  0.03217   0.0540   0.0600 
     .25      .50      .75      .90      .95 
  0.0700   0.0790   0.0900   0.1090   0.1261 

lowest : 0.012 0.034 0.038 0.039 0.041, highest: 0.422 0.464 0.467 0.610 0.611
---------------------------------------------------------------------------
free.sulfur.dioxide 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0       60    0.998    15.87    11.24        4        5 
     .25      .50      .75      .90      .95 
       7       14       21       31       35 

lowest :  1  2  3  4  5, highest: 55 57 66 68 72
---------------------------------------------------------------------------
total.sulfur.dioxide 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0      144        1    46.47    34.63     11.0     14.0 
     .25      .50      .75      .90      .95 
    22.0     38.0     62.0     93.2    112.1 

lowest :   6   7   8   9  10, highest: 155 160 165 278 289
---------------------------------------------------------------------------
density 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0      436        1   0.9967 0.002081   0.9936   0.9946 
     .25      .50      .75      .90      .95 
  0.9956   0.9968   0.9978   0.9991   1.0000 

lowest : 0.99007 0.99020 0.99064 0.99080 0.99084
highest: 1.00260 1.00289 1.00315 1.00320 1.00369
---------------------------------------------------------------------------
pH 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0       89        1    3.311   0.1716     3.06     3.12 
     .25      .50      .75      .90      .95 
    3.21     3.31     3.40     3.51     3.57 

lowest : 2.74 2.86 2.87 2.88 2.89, highest: 3.75 3.78 3.85 3.90 4.01
---------------------------------------------------------------------------
sulphates 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0       96    0.999   0.6581   0.1679     0.47     0.50 
     .25      .50      .75      .90      .95 
    0.55     0.62     0.73     0.85     0.93 

lowest : 0.33 0.37 0.39 0.40 0.42, highest: 1.61 1.62 1.95 1.98 2.00
---------------------------------------------------------------------------
alcohol 
       n  missing distinct     Info     Mean      Gmd      .05      .10 
    1599        0       65    0.998    10.42    1.178      9.2      9.3 
     .25      .50      .75      .90      .95 
     9.5     10.2     11.1     12.0     12.5 

lowest :  8.40000  8.50000  8.70000  8.80000  9.00000
highest: 13.50000 13.56667 13.60000 14.00000 14.90000
---------------------------------------------------------------------------
quality 
       n  missing distinct     Info      Sum     Mean      Gmd 
    1599        0        2    0.352      217   0.1357   0.2347 

---------------------------------------------------------------------------
~~~

**Checking for NAs**

~~~
colSums(is.na(wine))
~~~
~~~
       fixed.acidity     volatile.acidity          citric.acid 
                   0                    0                    0 
      residual.sugar            chlorides  free.sulfur.dioxide 
                   0                    0                    0 
total.sulfur.dioxide              density                   pH 
                   0                    0                    0 
           sulphates              alcohol              quality 
                   0                    0                    0 
~~~

data is clean and ready for analys

We have to first divide the data into training and test sets before performing any analysis.

~~~
##make it reproducible
set.seed(113)
#use 80% of dataset as training set and 20% as test set
sample <- sample(c(TRUE, FALSE), nrow(wine), replace=TRUE, prob=c(0.8,0.2))
train.wine <- wine[sample, ]
test.wine <- wine[!sample, ]
~~~

**A First Look at the Data**

Distribution of wine quality

~~~
ggplot(wine,aes(x=as.factor(quality),fill=as.factor(quality)))+
  geom_bar(width = 0.5)+
  ggtitle("Distribution of wine quality")+
  theme(legend.position = "none")+
  xlab("quality")
~~~

![1](https://user-images.githubusercontent.com/41892582/228455768-3d856ca9-0c5a-4564-a71c-60d42c129f4d.jpg)

it's just about 13 percent of a wine being considered a good wine

**using variables to investigate trends**

the relationship between quality and fixed acidity
~~~
ggplot(wine,aes(y=quality,x=fixed.acidity))+
  geom_smooth(method="loess")
~~~

![2](https://user-images.githubusercontent.com/41892582/228460978-1f868ae2-ec1a-47dc-a8f0-af54683dae6a.jpg)

the relationship between quality and volatile acidity

~~~
ggplot(wine,aes(y=quality,x=volatile.acidity))+
  geom_smooth(method="loess")
~~~

![3](https://user-images.githubusercontent.com/41892582/228461029-a2725710-5d7f-4dac-8b01-6838130f633c.jpg)

the relationship between quality and citric acid
~~~
ggplot(wine,aes(y=quality,x=citric.acid))+
  geom_smooth(method="loess")
~~~

![4](https://user-images.githubusercontent.com/41892582/228461095-5176d731-55e3-41e0-945c-ae890a88a441.jpg)

the relationship between quality and residual.sugar
~~~
ggplot(wine,aes(y=quality,x=residual.sugar))+
  geom_smooth(method="loess")
~~~

![5](https://user-images.githubusercontent.com/41892582/228461159-0c2b8149-2839-4177-8564-f0aa134e82fc.jpg)

the relationship between quality and chlorides
~~~
ggplot(wine,aes(y=quality,x=chlorides))+
  geom_smooth(method="loess")
~~~

![6](https://user-images.githubusercontent.com/41892582/228461253-ff5dc6f7-71f1-40a9-9e8e-2e8ffebfbad8.jpg)

the relationship between quality and free sulfur dioxide
~~~
ggplot(wine,aes(y=quality,x=free.sulfur.dioxide))+
  geom_smooth(method="loess")
~~~

![7](https://user-images.githubusercontent.com/41892582/228461306-23c4ffba-6a0c-4dee-95b8-36ba7d0af704.jpg)

the relationship between quality and total sulfur dioxide 
~~~
ggplot(wine,aes(y=quality,x=total.sulfur.dioxide))+
  geom_smooth(method="loess")
~~~

![8](https://user-images.githubusercontent.com/41892582/228461388-6b55703a-7c25-42b0-930d-b697948e5a63.jpg)

the relationship between quality and density 
~~~
ggplot(wine,aes(y=quality,x=density))+
  geom_smooth(method="loess")
~~~

![9](https://user-images.githubusercontent.com/41892582/228461447-bcdaaa28-a221-4483-bb10-5d1c9f167497.jpg)

the relationship between quality and pH 
~~~
ggplot(wine,aes(y=quality,x=pH))+
  geom_smooth(method="loess")
~~~

![10](https://user-images.githubusercontent.com/41892582/228461515-3f51952b-61ef-4c72-a2f2-92ae84515712.jpg)

the relationship between quality and sulphates 
~~~
ggplot(wine,aes(y=quality,x=sulphates))+
  geom_smooth(method="loess")
~~~

![11](https://user-images.githubusercontent.com/41892582/228461575-65c6063d-3252-4ab5-8cba-1db077aa5a2f.jpg)

the relationship between quality and alcohol 
~~~
ggplot(wine,aes(y=quality,x=alcohol))+
  geom_smooth(method="loess")
~~~

![12](https://user-images.githubusercontent.com/41892582/228461647-97330bc4-c7ab-4b76-b6ab-3b5b004ee4fb.jpg)


**Exploration of the data**

Let's see the correlation matrix of the dataset

~~~
corrplot(cor(train.wine), method="number")
~~~

![13](https://user-images.githubusercontent.com/41892582/228462831-0e8fdbda-c2fe-4813-b843-a37d9f74c3f2.png)

As we can see:

1-citric.acid and fixed.acidity are highly correlated

2-density and fixed.acidity are highly correlated

3-total.sulfur.dioxide and free.sulfur.dioxide are highly correlated

4-fixed.acidity and pH are highly correlated

5-citric.acid and volatile.acidity are highly correlated

6-citric.acid and pH are highly correlated

Calculate the partial correlation coefficient, t-statistic, and corresponding p-value. 

~~~
pcor(wine,method="pearson")
~~~

~~~
$estimate
                     fixed.acidity volatile.acidity citric.acid residual.sugar    chlorides free.sulfur.dioxide total.sulfur.dioxide     density          pH
fixed.acidity           1.00000000       0.05511657  0.33000119   -0.434545002 -0.225921889          0.11082683          -0.22732687  0.78667787 -0.71702627
volatile.acidity        0.05511657       1.00000000 -0.53093044   -0.016600555  0.255475581         -0.16589998           0.20920963  0.12015808  0.02695299
citric.acid             0.33000119      -0.53093044  1.00000000    0.050257849  0.265340058         -0.16194853           0.26610089  0.01149543 -0.03437654
residual.sugar         -0.43454500      -0.01660056  0.05025785    1.000000000 -0.003069007          0.12721487           0.04061223  0.59857006 -0.32812856
chlorides              -0.22592189       0.25547558  0.26534006   -0.003069007  1.000000000          0.05713921          -0.14370145  0.08530702 -0.22980768
free.sulfur.dioxide     0.11082683      -0.16589998 -0.16194853    0.127214872  0.057139210          1.00000000           0.66136989 -0.08633545  0.13771027
total.sulfur.dioxide   -0.22732687       0.20920963  0.26610089    0.040612228 -0.143701448          0.66136989           1.00000000  0.07380598 -0.19341721
density                 0.78667787       0.12015808  0.01149543    0.598570064  0.085307023         -0.08633545           0.07380598  1.00000000  0.57026155
pH                     -0.71702627       0.02695299 -0.03437654   -0.328128559 -0.229807684          0.13771027          -0.19341721  0.57026155  1.00000000
sulphates              -0.16130569      -0.19277031  0.02089762   -0.212142296  0.358426751          0.05549460           0.04073907  0.25868532 -0.12764853
alcohol                 0.52457279       0.09002877  0.14265123    0.479969805 -0.077836198         -0.02369126          -0.07467660 -0.72943461  0.51304516
quality                 0.07102042      -0.07973432  0.03204009    0.090880269 -0.084655712         -0.01382610          -0.04957838 -0.09173106  0.00489172
                       sulphates     alcohol     quality
fixed.acidity        -0.16130569  0.52457279  0.07102042
volatile.acidity     -0.19277031  0.09002877 -0.07973432
citric.acid           0.02089762  0.14265123  0.03204009
residual.sugar       -0.21214230  0.47996980  0.09088027
chlorides             0.35842675 -0.07783620 -0.08465571
free.sulfur.dioxide   0.05549460 -0.02369126 -0.01382610
total.sulfur.dioxide  0.04073907 -0.07467660 -0.04957838
density               0.25868532 -0.72943461 -0.09173106
pH                   -0.12764853  0.51304516  0.00489172
sulphates             1.00000000  0.25598356  0.16477276
alcohol               0.25598356  1.00000000  0.15441256
quality               0.16477276  0.15441256  1.00000000

$p.value
                     fixed.acidity volatile.acidity   citric.acid residual.sugar    chlorides free.sulfur.dioxide total.sulfur.dioxide       density
fixed.acidity         0.000000e+00     2.801956e-02  1.114776e-41   3.490853e-74 7.738266e-20        9.512055e-06         4.515237e-20  0.000000e+00
volatile.acidity      2.801956e-02     0.000000e+00 2.915844e-116   5.084453e-01 4.272776e-25        2.850299e-11         3.564009e-17  1.559812e-06
citric.acid           1.114776e-41    2.915844e-116  0.000000e+00   4.516861e-02 5.190735e-27        8.388495e-11         3.665068e-27  6.470326e-01
residual.sugar        3.490853e-74     5.084453e-01  4.516861e-02   0.000000e+00 9.027077e-01        3.623065e-07         1.056003e-01 4.460735e-155
chlorides             7.738266e-20     4.272776e-25  5.190735e-27   9.027077e-01 0.000000e+00        2.274047e-02         8.734056e-09  6.639179e-04
free.sulfur.dioxide   9.512055e-06     2.850299e-11  8.388495e-11   3.623065e-07 2.274047e-02        0.000000e+00        1.811607e-200  5.705264e-04
total.sulfur.dioxide  4.515237e-20     3.564009e-17  3.665068e-27   1.056003e-01 8.734056e-09       1.811607e-200         0.000000e+00  3.242369e-03
density               0.000000e+00     1.559812e-06  6.470326e-01  4.460735e-155 6.639179e-04        5.705264e-04         3.242369e-03  0.000000e+00
pH                   5.033181e-251     2.829316e-01  1.707952e-01   3.356751e-41 1.728694e-20        3.559902e-08         7.395196e-15 9.914385e-138
sulphates             9.973965e-11     9.115619e-15  4.051484e-01   1.260208e-17 2.288180e-49        2.695947e-02         1.045151e-01  1.038520e-25
alcohol              4.734912e-113     3.266102e-04  1.122111e-08   2.419717e-92 1.902811e-03        3.452838e-01         2.895728e-03 5.288305e-264
quality               4.620233e-03     1.467906e-03  2.017742e-01   2.863443e-04 7.302097e-04        5.818172e-01         4.815860e-02  2.507899e-04
                                pH    sulphates       alcohol      quality
fixed.acidity        5.033181e-251 9.973965e-11 4.734912e-113 4.620233e-03
volatile.acidity      2.829316e-01 9.115619e-15  3.266102e-04 1.467906e-03
citric.acid           1.707952e-01 4.051484e-01  1.122111e-08 2.017742e-01
residual.sugar        3.356751e-41 1.260208e-17  2.419717e-92 2.863443e-04
chlorides             1.728694e-20 2.288180e-49  1.902811e-03 7.302097e-04
free.sulfur.dioxide   3.559902e-08 2.695947e-02  3.452838e-01 5.818172e-01
total.sulfur.dioxide  7.395196e-15 1.045151e-01  2.895728e-03 4.815860e-02
density              9.914385e-138 1.038520e-25 5.288305e-264 2.507899e-04
pH                    0.000000e+00 3.303537e-07 2.115516e-107 8.455161e-01
sulphates             3.303537e-07 0.000000e+00  3.420297e-25 3.888559e-11
alcohol              2.115516e-107 3.420297e-25  0.000000e+00 6.111237e-10
quality               8.455161e-01 3.888559e-11  6.111237e-10 0.000000e+00

$statistic
                     fixed.acidity volatile.acidity citric.acid residual.sugar  chlorides free.sulfur.dioxide total.sulfur.dioxide     density          pH
fixed.acidity             0.000000        2.1990308  13.9264667    -19.2206075 -9.2389577           4.4423934            -9.299535  50.7626527 -40.9790603
volatile.acidity          2.199031        0.0000000 -24.9591470     -0.6614103 10.5267486          -6.7018557             8.522925   4.8216920   1.0741212
citric.acid              13.926467      -24.9591470   0.0000000      2.0046637 10.9633802          -6.5378760            10.997208   0.4579757  -1.3702738
residual.sugar          -19.220608       -0.6614103   2.0046637      0.0000000 -0.1222611           5.1093932             1.619212  29.7668405 -13.8378732
chlorides                -9.238958       10.5267486  10.9633802     -0.1222611  0.0000000           2.2799893            -5.784698   3.4108237  -9.4066464
free.sulfur.dioxide       4.442393       -6.7018557  -6.5378760      5.1093932  2.2799893           0.0000000            35.126666  -3.4522500   5.5387575
total.sulfur.dioxide     -9.299535        8.5229246  10.9972081      1.6192121 -5.7846977          35.1266660             0.000000   2.9482624  -7.8534946
density                  50.762653        4.8216920   0.4579757     29.7668405  3.4108237          -3.4522500             2.948262   0.0000000  27.6550501
pH                      -40.979060        1.0741212  -1.3702738    -13.8378732 -9.4066464           5.5387575            -7.853495  27.6550501   0.0000000
sulphates                -6.511230       -7.8262124   0.8326838     -8.6479871 15.2949352           2.2141596             1.624278  10.6684266  -5.1270983
alcohol                  24.545852        3.6011150   5.7415396     21.7952408 -3.1102096          -0.9440577            -2.983234 -42.4802085  23.8107878
quality                   2.836415       -3.1865349   1.2770422      3.6354568 -3.3845937          -0.5508454            -1.977494  -3.6697783   0.1948746
                      sulphates     alcohol    quality
fixed.acidity        -6.5112299  24.5458524  2.8364150
volatile.acidity     -7.8262124   3.6011150 -3.1865349
citric.acid           0.8326838   5.7415396  1.2770422
residual.sugar       -8.6479871  21.7952408  3.6354568
chlorides            15.2949352  -3.1102096 -3.3845937
free.sulfur.dioxide   2.2141596  -0.9440577 -0.5508454
total.sulfur.dioxide  1.6242777  -2.9832343 -1.9774940
density              10.6684266 -42.4802085 -3.6697783
pH                   -5.1270983  23.8107878  0.1948746
sulphates             0.0000000  10.5491459  6.6550444
alcohol              10.5491459   0.0000000  6.2260314
quality               6.6550444   6.2260314  0.0000000

$n
[1] 1599

$gp
[1] 10

$method
[1] "pearson"
~~~

the correlation between citric.acid and fixed.acidity are highly significant, as is the correlation between density and fixed.acidity,
also between total.sulfur.dioxide and free.sulfur.dioxide ,
fixed.acidity and pH and citric.acid and volatile.acidity We can safely assume that the independent variable has a high degree of correlation.

and let's check the suitability for factor analysis.

~~~
##new dataset of all independent variable
wine2<-dplyr::select(wine,-quality)
data_matrix<-cor(wine2)
KMO(data_matrix)
~~~

~~~
── Kaiser-Meyer-Olkin criterion (KMO) ─────────────────────────────────────

✖ The overall KMO value for your data is unacceptable.
  These data are not suitable for factor analysis.

  Overall: 0.432

  For each variable:
       fixed.acidity     volatile.acidity          citric.acid 
               0.449                0.522                0.697 
      residual.sugar            chlorides  free.sulfur.dioxide 
               0.205                0.465                0.484 
total.sulfur.dioxide              density                   pH 
               0.452                0.366                0.449 
           sulphates              alcohol 
               0.509                0.229 
~~~

Since MSA < 0.5, we can't run factor analysis on this data.

to deal with the multicollinearity we will use the stepwise regression.

**Multiple logistic regression model using all the data**

~~~
fit1<-glm(quality~.,train.wine,family = "binomial")
summary(fit1)
~~~

~~~
Call:
glm(formula = quality ~ ., family = "binomial", data = train.wine)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.9297  -0.4059  -0.2084  -0.1190   2.8000  

Coefficients:
                       Estimate Std. Error z value Pr(>|z|)    
(Intercept)           3.484e+02  1.241e+02   2.807 0.005002 ** 
fixed.acidity         4.568e-01  1.433e-01   3.188 0.001433 ** 
volatile.acidity     -3.537e+00  9.180e-01  -3.853 0.000117 ***
citric.acid          -2.680e-01  9.646e-01  -0.278 0.781176    
residual.sugar        2.432e-01  8.350e-02   2.912 0.003591 ** 
chlorides            -9.063e+00  4.077e+00  -2.223 0.026220 *  
free.sulfur.dioxide  -8.886e-04  1.358e-02  -0.065 0.947836    
total.sulfur.dioxide -1.021e-02  4.914e-03  -2.077 0.037772 *  
density              -3.665e+02  1.267e+02  -2.892 0.003828 ** 
pH                    1.064e+00  1.151e+00   0.924 0.355539    
sulphates             3.796e+00  5.955e-01   6.374 1.84e-10 ***
alcohol               6.631e-01  1.500e-01   4.420 9.87e-06 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 971.65  on 1249  degrees of freedom
Residual deviance: 651.43  on 1238  degrees of freedom
AIC: 675.43

Number of Fisher Scoring iterations: 6
~~~

Only 8 of the 11 independent variables are significant at the 0.05 level of significance.

**stepwise regression**

~~~
fit.step<-step(fit1,direction="both")
~~~

~~~
Start:  AIC=675.43
quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
    chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
    density + pH + sulphates + alcohol

                       Df Deviance    AIC
- free.sulfur.dioxide   1   651.43 673.43
- citric.acid           1   651.50 673.50
- pH                    1   652.27 674.27
<none>                      651.43 675.43
- total.sulfur.dioxide  1   656.13 678.13
- chlorides             1   658.30 680.30
- residual.sugar        1   658.71 680.71
- density               1   659.91 681.91
- fixed.acidity         1   661.67 683.67
- volatile.acidity      1   667.90 689.90
- alcohol               1   671.27 693.27
- sulphates             1   689.46 711.46

Step:  AIC=673.43
quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
    chlorides + total.sulfur.dioxide + density + pH + sulphates + 
    alcohol

                       Df Deviance    AIC
- citric.acid           1   651.50 671.50
- pH                    1   652.28 672.28
<none>                      651.43 673.43
+ free.sulfur.dioxide   1   651.43 675.43
- chlorides             1   658.34 678.34
- residual.sugar        1   658.78 678.78
- density               1   659.92 679.92
- total.sulfur.dioxide  1   660.15 680.15
- fixed.acidity         1   661.75 681.75
- volatile.acidity      1   668.01 688.01
- alcohol               1   671.41 691.41
- sulphates             1   689.47 709.47

Step:  AIC=671.5
quality ~ fixed.acidity + volatile.acidity + residual.sugar + 
    chlorides + total.sulfur.dioxide + density + pH + sulphates + 
    alcohol

                       Df Deviance    AIC
- pH                    1   652.47 670.47
<none>                      651.50 671.50
+ citric.acid           1   651.43 673.43
+ free.sulfur.dioxide   1   651.50 673.50
- residual.sugar        1   658.79 676.79
- chlorides             1   659.31 677.31
- density               1   660.49 678.49
- total.sulfur.dioxide  1   660.58 678.58
- fixed.acidity         1   661.92 679.92
- alcohol               1   672.43 690.43
- volatile.acidity      1   673.46 691.46
- sulphates             1   689.73 707.73

Step:  AIC=670.47
quality ~ fixed.acidity + volatile.acidity + residual.sugar + 
    chlorides + total.sulfur.dioxide + density + sulphates + 
    alcohol

                       Df Deviance    AIC
<none>                      652.47 670.47
+ pH                    1   651.50 671.50
+ citric.acid           1   652.28 672.28
+ free.sulfur.dioxide   1   652.46 672.46
- residual.sugar        1   658.83 674.83
- density               1   660.91 676.91
- chlorides             1   661.88 677.88
- total.sulfur.dioxide  1   662.69 678.69
- fixed.acidity         1   666.50 682.50
- volatile.acidity      1   674.16 690.16
- alcohol               1   684.95 700.95
- sulphates             1   689.77 705.77
~~~

pH,citric.acid and free sulfur dioxide were omitted from the stepwise regression.

~~~
summary(fit.step)
~~~

~~~
Call:
glm(formula = quality ~ fixed.acidity + volatile.acidity + residual.sugar + 
    chlorides + total.sulfur.dioxide + density + sulphates + 
    alcohol, family = "binomial", data = train.wine)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.9606  -0.4115  -0.2112  -0.1209   2.8083  

Coefficients:
                       Estimate Std. Error z value Pr(>|z|)    
(Intercept)           2.944e+02  1.067e+02   2.759 0.005803 ** 
fixed.acidity         3.462e-01  9.324e-02   3.713 0.000205 ***
volatile.acidity     -3.388e+00  7.628e-01  -4.442 8.91e-06 ***
residual.sugar        2.156e-01  7.908e-02   2.726 0.006410 ** 
chlorides            -9.994e+00  4.022e+00  -2.485 0.012959 *  
total.sulfur.dioxide -1.101e-02  3.634e-03  -3.030 0.002447 ** 
density              -3.084e+02  1.072e+02  -2.878 0.004006 ** 
sulphates             3.674e+00  5.775e-01   6.361 2.01e-10 ***
alcohol               7.164e-01  1.271e-01   5.634 1.76e-08 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 971.65  on 1249  degrees of freedom
Residual deviance: 652.47  on 1241  degrees of freedom
AIC: 670.47

Number of Fisher Scoring iterations: 6
~~~

and all eight independent variables are significant in the model.

using the model to predict the quality from the testing data and compare it with the real data

~~~
##apply the model to the test data
test.step<-predict(fit.step,test.wine)   
##convert the probability value to good or bad.
test.step<-ifelse(test.step>0.5,1,0)          
##compare the model's output to the actual data
confusionMatrix(table(test.step,test.wine$quality))
~~~

~~~
Confusion Matrix and Statistics

         
test.step   0   1
        0 291  46
        1   5   7
                                          
               Accuracy : 0.8539          
                 95% CI : (0.8124, 0.8892)
    No Information Rate : 0.8481          
    P-Value [Acc > NIR] : 0.418           
                                          
                  Kappa : 0.1688          
                                          
 Mcnemar's Test P-Value : 2.13e-08        
                                          
            Sensitivity : 0.9831          
            Specificity : 0.1321          
         Pos Pred Value : 0.8635          
         Neg Pred Value : 0.5833          
             Prevalence : 0.8481          
         Detection Rate : 0.8338          
   Detection Prevalence : 0.9656          
      Balanced Accuracy : 0.5576          
                                          
       'Positive' Class : 0  
~~~



**polynomial regression model**

In order to develop a polynomial regression model, we will add higher dimensional variables using the plots from the "Using Variables to Investigate Trends" section.

~~~
fit.poly<-glm(quality~poly(fixed.acidity,3)+
                volatile.acidity+
                poly(citric.acid,3)+
                residual.sugar+
                chlorides+
                free.sulfur.dioxide+
                poly(pH,3)+
                poly(total.sulfur.dioxide,3)+
                poly(density,2)+
                poly(sulphates,3)+
                poly(alcohol,3),data = train.wine,family = "binomial")
~~~

~~~
summary(fit.poly)
~~~

~~~
Call:
glm(formula = quality ~ poly(fixed.acidity, 3) + volatile.acidity + 
    poly(citric.acid, 3) + residual.sugar + chlorides + free.sulfur.dioxide + 
    poly(pH, 3) + poly(total.sulfur.dioxide, 3) + poly(density, 
    2) + poly(sulphates, 3) + poly(alcohol, 3), family = "binomial", 
    data = train.wine)

Deviance Residuals: 
     Min        1Q    Median        3Q       Max  
-1.75913  -0.40260  -0.17227  -0.07277   3.01148  

Coefficients:
                                 Estimate Std. Error z value Pr(>|z|)    
(Intercept)                     -0.984906   0.714081  -1.379 0.167813    
poly(fixed.acidity, 3)1         22.426173  10.119011   2.216 0.026675 *  
poly(fixed.acidity, 3)2         -0.805775   4.953961  -0.163 0.870792    
poly(fixed.acidity, 3)3         -0.635228   3.473472  -0.183 0.854892    
volatile.acidity                -3.730205   1.027452  -3.631 0.000283 ***
poly(citric.acid, 3)1           -9.555508   7.105298  -1.345 0.178676    
poly(citric.acid, 3)2            6.003271   4.353363   1.379 0.167896    
poly(citric.acid, 3)3            0.338865   3.697571   0.092 0.926980    
residual.sugar                   0.202167   0.096359   2.098 0.035901 *  
chlorides                       -7.974685   4.176552  -1.909 0.056211 .  
free.sulfur.dioxide             -0.006355   0.017351  -0.366 0.714156    
poly(pH, 3)1                    -0.207497   7.622031  -0.027 0.978282    
poly(pH, 3)2                    -1.839951   4.512538  -0.408 0.683463    
poly(pH, 3)3                     0.245661   4.007058   0.061 0.951115    
poly(total.sulfur.dioxide, 3)1 -12.072052   7.935769  -1.521 0.128205    
poly(total.sulfur.dioxide, 3)2   8.417723   6.100056   1.380 0.167605    
poly(total.sulfur.dioxide, 3)3   3.477378   5.693253   0.611 0.541339    
poly(density, 2)1              -23.389961   9.630314  -2.429 0.015150 *  
poly(density, 2)2                4.190958   4.682066   0.895 0.370729    
poly(sulphates, 3)1             31.591067   5.809342   5.438 5.39e-08 ***
poly(sulphates, 3)2            -20.119310   6.788101  -2.964 0.003038 ** 
poly(sulphates, 3)3              9.613039   5.615525   1.712 0.086921 .  
poly(alcohol, 3)1               25.967615   6.858608   3.786 0.000153 ***
poly(alcohol, 3)2               -6.816693   4.418348  -1.543 0.122876    
poly(alcohol, 3)3               -1.478606   3.583864  -0.413 0.679919    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 971.65  on 1249  degrees of freedom
Residual deviance: 615.18  on 1225  degrees of freedom
AIC: 665.18

Number of Fisher Scoring iterations: 7
~~~

using the model to predict the quality from the testing data and compare it with the real data

~~~
##apply the model to the test data
test.poly<-predict(fit.poly,test.wine)   
##convert the probability value to good or bad.
test.poly<-ifelse(test.poly>0.5,1,0)          
##compare the model's output to the actual data
confusionMatrix(table(test.poly,test.wine$quality))
~~~

~~~
Confusion Matrix and Statistics

         
test.poly   0   1
        0 291  45
        1   5   8
                                          
               Accuracy : 0.8567          
                 95% CI : (0.8155, 0.8918)
    No Information Rate : 0.8481          
    P-Value [Acc > NIR] : 0.3603          
                                          
                  Kappa : 0.1942          
                                          
 Mcnemar's Test P-Value : 3.479e-08       
                                          
            Sensitivity : 0.9831          
            Specificity : 0.1509          
         Pos Pred Value : 0.8661          
         Neg Pred Value : 0.6154          
             Prevalence : 0.8481          
         Detection Rate : 0.8338          
   Detection Prevalence : 0.9628          
      Balanced Accuracy : 0.5670          
                                          
       'Positive' Class : 0  
~~~



**Spline Regression**

We'll examine whether or not adding splines to the polynomial model will make it better.

~~~

fit.spline<-glm(quality~ns(fixed.acidity,3)+
                  volatile.acidity+
                  ns(citric.acid,3)+
                  residual.sugar+
                  chlorides+
                  ns(total.sulfur.dioxide,3)+
                  ns(density,2)+
                  ns(sulphates,3)+
                  ns(alcohol,3),data = train.wine,family = "binomial")
~~~

~~~
summary(fit.spline)
~~~

~~~
Call:
glm(formula = quality ~ ns(fixed.acidity, 3) + volatile.acidity + 
    ns(citric.acid, 3) + residual.sugar + chlorides + ns(total.sulfur.dioxide, 
    3) + ns(density, 2) + ns(sulphates, 3) + ns(alcohol, 3), 
    family = "binomial", data = train.wine)

Deviance Residuals: 
     Min        1Q    Median        3Q       Max  
-1.96142  -0.40253  -0.17717  -0.06536   2.97719  

Coefficients:
                             Estimate Std. Error z value Pr(>|z|)    
(Intercept)                  -6.77904    2.83593  -2.390 0.016829 *  
ns(fixed.acidity, 3)1         2.05312    0.83900   2.447 0.014401 *  
ns(fixed.acidity, 3)2         4.96998    2.08856   2.380 0.017330 *  
ns(fixed.acidity, 3)3         3.76268    1.44143   2.610 0.009044 ** 
volatile.acidity             -3.62154    1.02785  -3.523 0.000426 ***
ns(citric.acid, 3)1          -0.62376    0.69043  -0.903 0.366299    
ns(citric.acid, 3)2          -1.45728    0.99963  -1.458 0.144891    
ns(citric.acid, 3)3          -0.22383    0.77224  -0.290 0.771939    
residual.sugar                0.23490    0.09443   2.488 0.012862 *  
chlorides                    -7.78475    4.17044  -1.867 0.061951 .  
ns(total.sulfur.dioxide, 3)1 -2.35261    0.86562  -2.718 0.006571 ** 
ns(total.sulfur.dioxide, 3)2 -0.80129    1.43064  -0.560 0.575416    
ns(total.sulfur.dioxide, 3)3  1.57690    2.55534   0.617 0.537169    
ns(density, 2)1              -6.72984    2.33636  -2.880 0.003971 ** 
ns(density, 2)2              -3.33041    1.48663  -2.240 0.025075 *  
ns(sulphates, 3)1             6.73720    1.14550   5.881 4.07e-09 ***
ns(sulphates, 3)2             4.88774    4.58330   1.066 0.286232    
ns(sulphates, 3)3            -2.99285    4.91425  -0.609 0.542515    
ns(alcohol, 3)1               3.49852    1.03007   3.396 0.000683 ***
ns(alcohol, 3)2              10.21961    3.85939   2.648 0.008097 ** 
ns(alcohol, 3)3               4.22299    1.21810   3.467 0.000527 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 971.65  on 1249  degrees of freedom
Residual deviance: 614.44  on 1229  degrees of freedom
AIC: 656.44

~~~

using the model to predict the quality from the testing data and compare it with the real data

~~~
##apply the model to the test data
test.spline<-predict(fit.spline,test.wine)   
##convert the probability value to good or bad.
test.spline<-ifelse(test.spline>0.5,1,0)          
##compare the model's output to the actual data
confusionMatrix(table(test.spline,test.wine$quality))
~~~

~~~
Confusion Matrix and Statistics

           
test.spline   0   1
          0 290  46
          1   6   7
                                          
               Accuracy : 0.851           
                 95% CI : (0.8092, 0.8867)
    No Information Rate : 0.8481          
    P-Value [Acc > NIR] : 0.4772          
                                          
                  Kappa : 0.162           
                                          
 Mcnemar's Test P-Value : 6.362e-08       
                                          
            Sensitivity : 0.9797          
            Specificity : 0.1321          
         Pos Pred Value : 0.8631          
         Neg Pred Value : 0.5385          
             Prevalence : 0.8481          
         Detection Rate : 0.8309          
   Detection Prevalence : 0.9628          
      Balanced Accuracy : 0.5559          
                                          
       'Positive' Class : 0        
~~~


**Regression tree**

~~~
##the Regression tree model
dt<-rpart(quality~.,data =train.wine)
print(dt)
~~~

~~~
n= 1250 

node), split, n, deviance, yval
      * denotes terminal node

 1) root 1250 142.4832000 0.13120000  
   2) alcohol< 11.45 1048  73.8931300 0.07633588  
     4) volatile.acidity>=0.385 848  28.9386800 0.03537736 *
     5) volatile.acidity< 0.385 200  37.5000000 0.25000000  
      10) sulphates< 0.635 60   0.9833333 0.01666667 *
      11) sulphates>=0.635 140  31.8500000 0.35000000  
        22) alcohol< 9.75 32   2.7187500 0.09375000 *
        23) alcohol>=9.75 108  26.4074100 0.42592590  
          46) pH>=3.265 57  10.5614000 0.24561400  
            92) density>=0.995425 45   5.9111110 0.15555560 *
            93) density< 0.995425 12   2.9166670 0.58333330 *
          47) pH< 3.265 51  11.9215700 0.62745100 *
   3) alcohol>=11.45 202  49.0693100 0.41584160  
     6) sulphates< 0.685 108  17.5185200 0.20370370  
      12) volatile.acidity>=0.385 70   5.4857140 0.08571429 *
      13) volatile.acidity< 0.385 38   9.2631580 0.42105260  
        26) residual.sugar< 4 31   6.7741940 0.32258060 *
        27) residual.sugar>=4 7   0.8571429 0.85714290 *
     7) sulphates>=0.685 94  21.1063800 0.65957450  
      14) free.sulfur.dioxide>=18.5 30   6.9666670 0.36666670 *
      15) free.sulfur.dioxide< 18.5 64  10.3593800 0.79687500  
        30) chlorides>=0.087 15   3.7333330 0.46666670 *
        31) chlorides< 0.087 49   4.4897960 0.89795920 *
~~~

~~~
##the plot of the decision tree model
rpart.plot(dt)
~~~

![rd](https://user-images.githubusercontent.com/41892582/228502782-d8ba99c1-e2c8-46d3-973d-c191d04cf16e.jpg)

using the model to predict the quality from the testing data and compare it with the real data

~~~
##apply the model to the test data
test.dt<-predict(dt,test.wine)   
##convert the probability value to good or bad.
test.dt<-ifelse(test.dt>0.5,1,0)          
##compare the model's output to the actual data
confusionMatrix(table(test.dt,test.wine$quality))
~~~

~~~
Confusion Matrix and Statistics

       
test.dt   0   1
      0 281  38
      1  15  15
                                          
               Accuracy : 0.8481          
                 95% CI : (0.8061, 0.8841)
    No Information Rate : 0.8481          
    P-Value [Acc > NIR] : 0.536568        
                                          
                  Kappa : 0.2827          
                                          
 Mcnemar's Test P-Value : 0.002512        
                                          
            Sensitivity : 0.9493          
            Specificity : 0.2830          
         Pos Pred Value : 0.8809          
         Neg Pred Value : 0.5000          
             Prevalence : 0.8481          
         Detection Rate : 0.8052          
   Detection Prevalence : 0.9140          
      Balanced Accuracy : 0.6162          
                                          
       'Positive' Class : 0  
~~~~




**Random forest**

~~~
rf<-randomForest(quality~.,ntree=500,data =train.wine)
plot(rf)
~~~

![rf](https://user-images.githubusercontent.com/41892582/228503482-bc40d272-8595-4d19-af4b-a462e243d4aa.jpg)

~~~
print(rf)
~~~

~~~
Call:
 randomForest(formula = quality ~ ., data = train.wine, ntree = 500) 
               Type of random forest: regression
                     Number of trees: 500
No. of variables tried at each split: 3

          Mean of squared residuals: 0.06626878
                    % Var explained: 41.86
~~~

Mean of  Square Residals: 0.066 We need to know if, if we add more trees, the Mean of squared residuals would change or not.

~~~
rf2<-randomForest(quality~.,ntree=1000,data =train.wine)
plot(rf2)
~~~

![rf2](https://user-images.githubusercontent.com/41892582/228504094-84c237d0-889a-4495-861c-458696523cac.jpg)

~~~
print(rf2)
~~~

~~~
Call:
 randomForest(formula = quality ~ ., data = train.wine, ntree = 1000) 
               Type of random forest: regression
                     Number of trees: 1000
No. of variables tried at each split: 3

          Mean of squared residuals: 0.0658856
                    % Var explained: 42.2
~~~

When we add more trees, the Mean of Square residuals doesn't change much, as well as the error rate and it seems to be constant after 200 trees.

using the model to predict the quality from the testing data and compare it with the real data

~~~
##apply the model to the test data
test.rf<-predict(rf2,test.wine)   
##convert the probability value to good or bad.
test.rf<-ifelse(test.rf>0.5,1,0)          
##compare the model's output to the actual data
confusionMatrix(table(test.rf,test.wine$quality))
~~~

~~~
Confusion Matrix and Statistics

       
test.rf   0   1
      0 287  32
      1   9  21
                                         
               Accuracy : 0.8825         
                 95% CI : (0.844, 0.9144)
    No Information Rate : 0.8481         
    P-Value [Acc > NIR] : 0.0397614      
                                         
                  Kappa : 0.4451         
                                         
 Mcnemar's Test P-Value : 0.0005908      
                                         
            Sensitivity : 0.9696         
            Specificity : 0.3962         
         Pos Pred Value : 0.8997         
         Neg Pred Value : 0.7000         
             Prevalence : 0.8481         
         Detection Rate : 0.8223         
   Detection Prevalence : 0.9140         
      Balanced Accuracy : 0.6829         
                                         
       'Positive' Class : 0  
~~~

# Conclusion

In stepwise and polynomial models, we are able to achieve 98% sensitivity with poor specificity in both of them (13% and 15%, respectively), and in random forest regression, we are able to achieve 97% sensitivity, 40% specificity, and overall accuracy of 88%.
