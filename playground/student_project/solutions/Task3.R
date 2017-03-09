  require(ggplot2)
  library(scales)

  # store standard output of task3-a to a file data.csv
  data <- read.csv(file="data.csv",head=TRUE,sep=";")

  ggplot(data, aes(
    x=(vector_size_num_int_val*64/8/1024/1024),
    y=duration_ms/1000, group = type, linetype = type)) +
    scale_y_continuous(limits = c(1.6, 4.2), expand = c(0,0))+
    scale_x_continuous(limits = c(-0.1, 12.1), expand = c(0,0)) +
    xlab("vector size [in MiB]") +
    ylab("execution time [in sec]") +
    ggtitle("vector size impact [six filters on 450 MB column]") +
    geom_line() +theme_bw()  + theme(
      plot.title = element_text(size=10),
      axis.title.x = element_text(size=10),
      axis.title.y = element_text(size=10)
    )

