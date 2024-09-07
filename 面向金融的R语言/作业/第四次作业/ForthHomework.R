# 安装和加载必要的包
# install.packages("quantmod")
library(quantmod)

# 使用优化算法（这里使用DEoptim包，也可以选择其他优化包）
# install.packages("DEoptim")
library(DEoptim)

# 设置标的资产
symbols <- c("AAPL", "GOOGL", "MSFT", "GLD", "TLT")  # 选取了苹果、谷歌、微软、黄金、长期国债ETF

# 下载资产数据
getSymbols(symbols, from = "2020-01-01", to = Sys.Date(), adjust = TRUE)

# 计算每日收益率
returns <- ROC(Cl(get(symbols[1])), type = "continuous")
for (symbol in symbols[-1]) {
  returns <- merge(returns, ROC(Cl(get(symbol)), type = "continuous"))
}

# 移除第一天的收益率数据
returns <- returns[-1, ]

# 计算每个资产的平均收益率和标准差
mean_returns <- colMeans(returns)
sd_returns <- apply(returns, 2, sd)

# 定义目标函数
sharpe_ratio <- function(weights, mean_returns, cov_matrix) {
  port_returns <- sum(weights * mean_returns)
  port_volatility <- sqrt(t(weights) %*% cov_matrix %*% weights)
  sharpe_ratio <- port_returns / port_volatility
  return (sharpe_ratio * -1.0)
}

# 设置优化问题
cov_matrix <- cov(returns)
num_assets <- length(symbols)
bounds <- list(rep(0, num_assets), rep(1, num_assets))  # 权重在0和1之间

# 定义优化问题
optimal_portfolio <- DEoptim(
  fn = sharpe_ratio,
  lower = bounds[[1]],
  upper = bounds[[2]],
  cov_matrix = cov_matrix,
  mean_returns = mean_returns,
  DEoptim.control(trace = TRUE, itermax = 1000)
)

# 输出优化结果
optimal_weights <- optimal_portfolio$optim$bestmem
optimal_weights