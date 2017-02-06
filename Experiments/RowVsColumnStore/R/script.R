#install.packages(c("psych"), dependencies = TRUE)
require(psych)
require(ggplot2)
library(grid)

toSec = function(x) {
  x / 1000000000
}

inMio = function(x) {
  x / 1000000
}

addM = function(x) {
  paste(x, "M", sep = "")
}

addGB = function(x) {
  paste(x, "GB", sep = "")
}

asMio = function(vec) {
  lapply(vec, addM)
}

asGB = function(vec) {
  lapply(vec, addGB)
}

data <- read.csv(file="/Users/marcus/git/pantheon/Experiments/RowVsColumnStore/R/data.csv",head=TRUE,sep=";")

dataHost = subset(data, data$Platform=="Host")
  dataHostColumnStore = subset(dataHost, dataHost$Type=="ColumnStore")
  dataHostRowStore    = subset(dataHost, dataHost$Type=="RowStore")

dataDevice = subset(data, data$Platform=="Device")
  dataDeviceColumnStore = subset(dataDevice, dataDevice$Type=="ColumnStore")
  
  numRec = dataHostColumnStore$Q1NumRecordsToProcess
  xval = inMio(dataHostColumnStore$Q3NumRecordsToProcess) # number of records in item table!
  
  p<-ggplot(dataHostColumnStore)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ1MultiThreaded))), colour="#f2d0c2",size=2.0, shape=17)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ1MultiThreaded))), colour="#cddcf4",size=2.0, shape=16)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ1SingleThreaded))), colour="#ceeacf",size=2.0, shape=17)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ1SingleThreaded))), colour="#f4c2c3",size=2.0, shape=16)+
    
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ1MultiThreaded))), fill="#b5b4b5",color="#4c79be",size=0.7) +
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ1MultiThreaded))), fill="#b5b4b5",color="#f26529",size=0.7) +
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ1SingleThreaded))), fill="#b5b4b5",color="#5d935f",size=0.7) +
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ1SingleThreaded))), fill="#b5b4b5",color="#f8393a",size=0.7) +
    
    theme_bw() +
    theme(panel.border = element_blank(), axis.line = element_line(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_line( size=0.5, color="#cfcfcf" ),
          panel.grid.minor.y = element_blank(), 
          axis.line.x = element_line( size=0.5, linetype = "solid", colour = "#5b5b5b"),
          axis.line.y = element_line( size=0.5, linetype = "solid", colour = "#5b5b5b")) +
    #+
    #theme_light() +
    scale_x_continuous(expand = c(0, 0), breaks=seq(from = 0, to = 70, by = 10), labels=asMio)   +
    scale_y_continuous(labels = asMio) + 
    xlab("#records in item table") +
    ylab("throughput [records/s]") +
    #scale_y_log10(expand = c(0, 0.1)) + #, breaks = c(0.001, 0.002, 0.004, 0.01, 0.02, 0.1, 0.25,0.5, 1, 3, 6)) +
    
    scale_colour_manual(name="Error Bars",values=c("XX","YY"), 
                        guide = guide_legend(override.aes=aes(fill=NA))) + 
    scale_fill_manual(name="Bar",values=c("XX","YY"), guide="none") +
    theme(axis.title.x = element_text(size = 10)) +
    theme(axis.title.y = element_text(size = 10)) +
    
    ggtitle("sum prices of 150 items") + 
    theme(plot.title = element_text(size = 10)) 
  p
  
  ggsave("hostQ1.pdf", width = 3.5, height = 2)
  
  ################################################################################################################################
  
  numRec = dataHostColumnStore$Q2NumRecordsToProcess
  xval = inMio(dataHostColumnStore$NumberOfCustomers) # number of records in item table!
  
  p<-ggplot(dataHostColumnStore)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ2MultiThreaded))), colour="#f2d0c2",size=2.0, shape=17)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ2MultiThreaded))), colour="#cddcf4",size=2.0, shape=16)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ2SingleThreaded))), colour="#ceeacf",size=2.0, shape=17)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ2SingleThreaded))), colour="#f4c2c3",size=2.0, shape=16)+
    
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ2MultiThreaded))), fill="#b5b4b5",color="#4c79be",size=0.7) +
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ2MultiThreaded))), fill="#b5b4b5",color="#f26529",size=0.7) +
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ2SingleThreaded))), fill="#b5b4b5",color="#5d935f",size=0.7) +
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ2SingleThreaded))), fill="#b5b4b5",color="#f8393a",size=0.7) +
    
    theme_bw() +
    theme(panel.border = element_blank(), axis.line = element_line(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_line( size=0.5, color="#cfcfcf" ),
          panel.grid.minor.y = element_blank(), 
          axis.line.x = element_line( size=0.5, linetype = "solid", colour = "#5b5b5b"),
          axis.line.y = element_line( size=0.5, linetype = "solid", colour = "#5b5b5b")) +
    #+
    #theme_light() +
    scale_x_continuous(expand = c(0, 0), breaks=seq(from = 5, to = 90, by = 20), labels=asMio)   +
    scale_y_continuous(labels = asMio) + 
    xlab("#records in customer table") +
    ylab("throughput [records/s]") +
    #scale_y_log10(expand = c(0, 0.1)) + #, breaks = c(0.001, 0.002, 0.004, 0.01, 0.02, 0.1, 0.25,0.5, 1, 3, 6)) +
    
    scale_colour_manual(name="Error Bars",values=c("XX","YY"), 
                        guide = guide_legend(override.aes=aes(fill=NA))) + 
    scale_fill_manual(name="Bar",values=c("XX","YY"), guide="none") +
    theme(axis.title.x = element_text(size = 10)) +
    theme(axis.title.y = element_text(size = 10)) +
    
    ggtitle("materialize 150 customers") + 
    theme(plot.title = element_text(size = 10)) 
  p
  
  ggsave("hostQ2.pdf", width = 3.5, height = 2)
  
  ################################################################################################################################
 
  
  numRec = dataHostColumnStore$Q3NumRecordsToProcess
  xval = inMio(dataHostColumnStore$Q3NumRecordsToProcess)
  
  p<-ggplot(dataHostColumnStore)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ3MultiThreaded))), colour="#f2d0c2",size=2.0, shape=17)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ3SingleThreaded))), colour="#ceeacf",size=2.0, shape=17)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ3MultiThreaded))), colour="#cddcf4",size=2.0, shape=16)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ3SingleThreaded))), colour="#f4c2c3",size=2.0, shape=16)+

    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ3MultiThreaded))), fill="#b5b4b5",color="#f26529",size=0.7) +    
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ3MultiThreaded))), fill="#b5b4b5",color="#4c79be",size=0.7) +
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ3SingleThreaded))), fill="#b5b4b5",color="#5d935f",size=0.7) +
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostRowStore$DurationQ3SingleThreaded))), fill="#b5b4b5",color="#f8393a",size=0.7) +
    
    theme_bw() +
    theme(panel.border = element_blank(), axis.line = element_line(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_line( size=0.5, color="#cfcfcf" ),
          panel.grid.minor.y = element_blank(), 
          axis.line.x = element_line( size=0.5, linetype = "solid", colour = "#5b5b5b"),
          axis.line.y = element_line( size=0.5, linetype = "solid", colour = "#5b5b5b")) +
     #+
    #theme_light() +
    scale_x_continuous(expand = c(0, 0), breaks=seq(from = 5, to = 65, by = 10), labels=asMio)   +
    scale_y_continuous(lim=c(300, 2000), labels = asMio) + 
    xlab("#records in item table") +
    ylab("throughput [records/s]") +
    #scale_y_log10(expand = c(0, 0.1)) + #, breaks = c(0.001, 0.002, 0.004, 0.01, 0.02, 0.1, 0.25,0.5, 1, 3, 6)) +
    
    scale_colour_manual(name="Error Bars",values=c("XX","YY"), 
                        guide = guide_legend(override.aes=aes(fill=NA))) + 
    scale_fill_manual(name="Bar",values=c("XX","YY"), guide="none") +
    theme(axis.title.x = element_text(size = 10)) +
    theme(axis.title.y = element_text(size = 10)) +
    
    ggtitle("sum all prices in items table") + 
    theme(plot.title = element_text(size = 10)) 
  p
  
  ggsave("hostQ3.pdf", width = 3.5, height = 2)
  
  ################################################################################################################################
  
  numRec = dataHostColumnStore$Q3NumRecordsToProcess
  xval = inMio(dataHostColumnStore$Q3NumRecordsToProcess)
  
  p<-ggplot(dataHostColumnStore)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ3MultiThreaded))), colour="#f2d0c2",size=2.0, shape=17)+
    geom_jitter(aes(x = xval, y = inMio(numRec/toSec(dataDeviceColumnStore$DurationQ3MultiThreaded))), colour="#e2cdf1",size=2.0, shape=16)+

    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataHostColumnStore$DurationQ3MultiThreaded))), fill="#b5b4b5",color="#f26529",size=0.7) +    
    geom_smooth(aes(x = xval, y = inMio(numRec/toSec(dataDeviceColumnStore$DurationQ3MultiThreaded))), fill="#b5b4b5",color="#a939f8",size=0.7) +
    
    theme_bw() +
    theme(panel.border = element_blank(), axis.line = element_line(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_line( size=0.5, color="#cfcfcf" ),
          panel.grid.minor.y = element_blank(), 
          axis.line.x = element_line( size=0.5, linetype = "solid", colour = "#5b5b5b"),
          axis.line.y = element_line( size=0.5, linetype = "solid", colour = "#5b5b5b")) +
    #+
    #theme_light() +
    scale_x_continuous(expand = c(0, 0), breaks=seq(from = 5, to = 65, by = 10), labels=asMio)   +
    scale_y_continuous(lim=c(300, 10000), labels = asMio) + 
    xlab("#records in item table") +
    ylab("throughput [records/s]") +
    #scale_y_log10(expand = c(0, 0.1)) + #, breaks = c(0.001, 0.002, 0.004, 0.01, 0.02, 0.1, 0.25,0.5, 1, 3, 6)) +
    
    scale_colour_manual(name="Error Bars",values=c("XX","YY"), 
                        guide = guide_legend(override.aes=aes(fill=NA))) + 
    scale_fill_manual(name="Bar",values=c("XX","YY"), guide="none") +
    theme(axis.title.x = element_text(size = 10)) +
    theme(axis.title.y = element_text(size = 10)) +
    
    ggtitle("sum all prices in items table\n[transfer costs to device excluded]") + 
    theme(plot.title = element_text(size = 10)) 
  p
  
  ggsave("hostQ3D.pdf", width = 3.5, height = 2)
  
  