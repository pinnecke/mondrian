# uncomment if you haven't ggplot2
#install.packages("ggplot2")
library(ggplot2)

results =read.csv("results_file6.csv",header = F)
names(results)= c("sf","time","type")
my_plot=ggplot(data=results, aes(x= sf  , y=time, group=type, colour=type)  )+geom_line() +geom_point()+
  xlab("selectivity factor")+ylab("time in seconds")+
  ggtitle("branch free vs micro optimized vs optimized branch free")
ggsave(file="results.png")