“为了解各位的学习状况，下节课（11月26日）我们进行一次期中测试。测试时长1个小时。请各位同学准备1-2张空白纸答题。”

“考学过的内容”

“考试不记入成绩。只做了解学习情况用。”

“体量不大，就是一个小时。写代码也不需要写太多。”

“闭卷测试。大家不要担心，只是做了解学习状况用。”

## 题解

1. 编程求解 x = sin(x) 的解。

   ``` R
   f <- function(x) sin(x)
   x <- 1 # 初值
   tol <- 1e-8 # 精度要求
   max_iter <- 1000 # 最大迭代次数
   for (i in 1:max_iter) {
     x_new <- f(x)
     if (abs(x_new - x) < tol) {
       break
     }
     x <- x_new
   }
   if (i == max_iter) {
     cat("未能收敛\n")
   } else {
     cat("解为：", x_new, "\n")
   }
   ```

2. 给定状态转移矩阵 A，编写代码模拟马尔可夫过程。

   ``` R
   A <- matrix(c(0.5, 0.2, 0.3,
                 0.4, 0.3, 0.3,
                 0.6, 0.1, 0.3), nrow = 3, byrow = TRUE) # 状态转移矩阵
   n <- 10000 # 模拟步数
   start_state <- 1 # 初始状态
   state_seq <- numeric(n) # 记录状态序列
   state_seq[1] <- start_state
   for (i in 2:n) {
     state_seq[i] <- sample(1:3, size = 1, prob = A[state_seq[i - 1], ])
   }
   table(state_seq) / n # 输出状态出现的频率，可以近似看作概率
   ```

3. 给出一种模拟保险公司利润随时间变化的过程。用泊松分布表示出险次数分布，出险金额可自定义。保费收入自定义。

   这题没咋看懂（

   ``` R
   set.seed(123) # 设置随机数种子，使结果可重复
   
   # 模拟参数设置
   T <- 365 # 模拟时间长度，单位为天
   lambda <- 100 # 泊松分布的参数，表示平均每天的事故次数
   mu_claims <- 1000 # 出险金额的期望值
   sigma_claims <- 500 # 出险金额的标准差
   premium_income <- 2000 # 保费收入，假设每天的保费收入是固定的
   
   # 生成随机数
   N <- rpois(T, lambda = lambda) # 生成每天的出险次数，服从泊松分布
   claims_amount <- rnorm(T, mean = mu_claims, sd = sigma_claims) # 生成每天的出险金额，服从正态分布
   
   # 计算每天的利润
   profit <- - claims_amount * N + premium_income
   
   # 绘制时间序列图
   plot(profit, type = "l", xlab = "Time", ylab = "Profit")
   ```

   