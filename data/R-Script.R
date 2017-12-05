#### load requiered libraries ####

#load plyr
library(plyr)
#need car package for recode function
library(car)
#load R.utils
library(R.utils)
#need gdata package for reorder function
library(gdata)
#package for plotting
library(ggplot2)
#needed for unit
library(gridExtra)
#load ez package
library(ez)
#load reshape
library(reshape)
#load reshape2 for dcast
library(reshape2)
#load (pastecs)
library(pastecs)
#load dplyr
library(dplyr)
#load QuantPsyc for standardized regression coefficients
library(QuantPsyc)
#load moments for skewness and kurtosis
library(moments)
#needed for bootstrapping
library(boot)
#needed for GLMM
library(glmm)
library(lme4)
##### DATA COLLECTION ####

#save path to main directory
path="C:/Users/Hanna/ambi_gain_loss_shock/data"
#change working directory to data directory
setwd(path)
#read in excel file and create data set
dataSet  <-  read.csv("triplet.csv", header=T, sep=",", dec=".")

dataSet <- dataSet[!dataSet$parameter == 'ambiguityLevel',]
dataSet <- dataSet[,-c(1)]
dataSet <- dataSet[,-c(3,5,7)]


dataSet <- melt.data.frame(dataSet, id=c("MID", "parameter", "ambiguity"), measured=c("gain", "loss", "shock"))


ezANOVA(data = dataSet, dv = value, wid = MID, within = .(variable, parameter, ambiguity), type=3)

ambiguous = dataSet[dataSet$ambiguity=="ambi",]
unambiguous = dataSet[dataSet$ambiguity=="unambi",]

ezPlot(data = ambiguous, dv = value, wid = MID, within = .(parameter, variable), x=parameter, split = variable)
ezPlot(data = unambiguous, dv = value, wid = MID, within = .(parameter, variable), x=parameter, split = variable)


temp = as.data.frame(table(dataSet$MID))


complete.cases(dataSet)
dataSet[!complete.cases(dataSet),]

ezDesign(dataSet$gain)
