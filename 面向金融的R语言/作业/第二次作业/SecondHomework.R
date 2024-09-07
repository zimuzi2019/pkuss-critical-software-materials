#Let f(x) = (sinx)^2 for 0 < x < 2π.



# (a) Graph the function.
# 若未安装 ggplot2 安装包则把注释去掉
#install.packages("ggplot2")
library(ggplot2)

f <- function(x) {
  (sin(x))^2
}

x <- seq(0, 2*pi, length.out = 1000)
y <- f(x)
df <- data.frame(x, y)
ggplot(df, aes(x, y)) + geom_line() + xlab("x") + ylab("f(x)") + ggtitle("Graph of f(x)")



# 蒙特卡洛积分法
set.seed(123)  # 设置随机种子，保证结果可复现

N <- 100000  # 抽样次数
x_samples <- runif(N, min = 0, max = 2*pi)  # 在区间 [0, 2π] 中进行均匀抽样
f_samples <- f(x_samples)  # 计算样本的函数值

area <- mean(f_samples) * (2*pi)  # 估计的面积
error <- 1.96 * sd(f_samples) / sqrt(N)  # 误差

lower_bound <- area - error  # 置信区间下界
upper_bound <- area + error  # 置信区间上界


print(area)
print(lower_bound)
print(upper_bound)


exact_area <- integrate(f, 0, 2*pi)$value

print(exact_area)




rsin2 <- function(n) {
  k <- area  # 密度归一化常数
  
  samples <- c()
  i <- 1
  
  while (i <= n) {
    x <- runif(1, min = 0, max = 2*pi)
    y <- runif(1, min = 0, max = k)
    
    if (y <= f(x)) {
      samples[i] <- x
      i <- i + 1
    }
  }
  
  samples
}

set.seed(123)  # 设置随机种子，保证结果可复现

samples <- rsin2(1000)  # 生成1000个样本

# 绘制直方图
ggplot(data.frame(samples), aes(samples)) + geom_histogram(bins = 30, fill = "steelblue", color = "white") + xlab("x") + ylab("Frequency") + ggtitle("Histogram of 1000 Samples")





set.seed(123)  # 设置随机种子，保证结果可复现

n <- 1000000
samples <- rsin2(n)  # 生成1,000,000个样本
mean_value <- mean(samples)  # 计算均值

# 计算置信区间
error <- 1.96 * sd(samples) / sqrt(n)
lower_bound <- mean_value - error
upper_bound <- mean_value + error

print(mean_value)
print(lower_bound)
print(upper_bound)