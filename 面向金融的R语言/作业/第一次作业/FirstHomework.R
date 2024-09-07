# 方法一：IRR_caculate 函数循环寻找 IRR
IRR_caculate1 <- function(C) {
  # C：初始投资成本 C_0 and 每期的净现金流 C_t
  C_0 <- -abs(C[1]) 
  C_t <- C[2: length(C)] 
  tolerance <- 0.0001
  
  for (IRR in seq(0, 1, 0.00000001)) {
    NPV <- 0
    for (i in 1:length(C_t)) { # length(C_t) 期数
      NPV <- NPV + C_t[i]/(1+IRR)^i
    }
    if (abs(NPV+C_0) < tolerance) break
  }
  
  return (IRR)
}

# 测试，可正确输出结果0.08363264
result1 <- IRR_caculate1(C = c(-10000,1000,1000,2000,3000,3000,4000))

print(result1)


# 方法二：对IRR计算公式进行推导变为多项式求解问题
IRR_caculate2 <- function(C){
  P_x <- round(Re(polyroot(C)),6)
  P_x <- P_x[P_x > 0]
  IRR <- 1/P_x - 1
  IRR_ff <- c()
  for(i in 1:length(IRR)){
    IRR_ff[i] <- all(!round(IRR[i],6) == round(IRR[-i],6))
  }
  if(length(which(IRR_ff))>0){
    IRR <- IRR[which(IRR_ff)]
    IRR <- IRR[which.max(IRR)]
  }else{
    IRR <- IRR[which.max(IRR)]
  }
  return(IRR)
}

# 测试，可正确输出结果0.0836326
result2 <- IRR_caculate2(C = c(-10000,1000,1000,2000,3000,3000,4000))

print(result2)