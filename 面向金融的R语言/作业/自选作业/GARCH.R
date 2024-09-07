# GARCH model
# (1) read data
library(quantmod)                                            # 加载包
library(fGarch)
getSymbols('^HSI', from='1989-12-01', to='2013-11-30')      # 从 Yahoo 网站下载恒生日价格指数
dim(HSI)                                                    # 查看数据规模
names(HSI)                                                  # 数据变量名称
chartSeries(HSI, theme='white')                             # 画出价格与交易的时序图

HSI <- read.table('HSI.txt')                                # 或者直接从硬盘中读取下载好的恒生指数日价格指数
HSI <- as.xts(HSI)                                          # 将数据格式转化为 xts 格式

# (2) compute return series
ptd.HSI <- HSI$HSI.Adjusted                                 # 提取日收盘价信息
rtd.HSI <- diff(log(ptd.HSI))*100                           # 计算日对数收益
rtd.HSI <- rtd.HSI[-1,]                                     # 删除一期缺失值
plot(rtd.HSI)                                               # 画出日收益序列的时序图

ptm.HSI <- to.monthly(HSI)$HSI.Adjusted                     # 提取月收盘价信息
rtm.HSI <- diff(log(ptm.HSI))*100                           # 计算月对数收益
rtm.HSI <- rtm.HSI[-1,]                                     # 删除一期缺失值
plot(rtm.HSI)                                               # 画出月收益序列的时序图
detach(package:quantmod)

# (3) ARCH 效应检验
# rtm.HSI <- as.numeric(rtm.HSI)
ind.outsample <- sub(' ','',substr(index(rtm.HSI), 4, 8)) %in% '2013'   # 设置样本外下标：2013年为样本外
ind.insample <- !ind.outsample                                          # 设置样本内下标，其余为样本内
rtm.insample <- rtm.HSI[ind.insample]
rtm.outsample <- rtm.HSI[ind.outsample]
Box.test(rtm.insample, lag=12, type='Ljung-Box')                        # 月收益不存在自相关
Box.test(rtm.insample^2, lag=12, type='Ljung-Box')                      # 平方月收益序列存在自相关

FinTS::ArchTest(x=rtm.insample, lags=12)                                # 存在显著的 ARCH 效应

# (4) 模型定阶
epst <- rtm.insample - mean(rtm.insample)                               # 均值调整对数收益
par(mfrow=c(1,2))
acf(as.numeric(epst)^2, lag.max=20, main='平方序列')
pacf(as.numeric(epst)^2, lag.max=20, main='平方序列')                               

# (5) 建立 GARCH 类
GARCH.model_1 <- garchFit(~garch(1,1), data=rtm.insample, trace=FALSE)   # GARCH(1,1)-N模型
GARCH.model_2 <- garchFit(~garch(2,1), data=rtm.insample, trace=FALSE)   # GARCH(1,2)-N模型
GARCH.model_3 <- garchFit(~garch(1,1), data=rtm.insample, cond.dist='std', trace=FALSE)  # GARCH(1,1)-t模型
GARCH.model_4 <- garchFit(~garch(1,1), data=rtm.insample, cond.dist='sstd', trace=FALSE) # GARCH(1,1)-st模型
GARCH.model_5 <- garchFit(~garch(1,1), data=rtm.insample, cond.dist='ged', trace=FALSE)  # GARCH(1,1)-GED模型
GARCH.model_6 <- garchFit(~garch(1,1), data=rtm.insample, cond.dist='sged', trace=FALSE) # GARCH(1,1)-SGED模型

summary(GARCH.model_1)
summary(GARCH.model_3)

plot(GARCH.model_1)     # 键入相应数字获取信息

# (6) 提取 GARCH 类模型信息
vol_1 <- fBasics::volatility(GARCH.model_1)                   # 提取GARCH(1,1)-N模型得到的波动率估计
sres_1 <- residuals(GARCH.model_1, standardize=TRUE)          # 提取GARCH(1,1)-N模型得到的标准化残差
vol_1.ts <- ts(vol_1, frequency=12, start=c(1990, 1))
sres_1.ts <- ts(sres_1, frequency=12, start=c(1990, 1))
par(mfcol=c(2,1))
plot(vol_1.ts, xlab='年', ylab='波动率')
plot(sres_1.ts, xlab='年', ylab='标准化残差')

# (7) 模型检验
par(mfrow=c(2,2))
acf(sres_1, lag=24)
pacf(sres_1, lag=24)
acf(sres_1^2, lag=24)
pacf(sres_1^2, lag=24)

par(mfrow=c(1,1))
qqnorm(sres_1)
qqline(sres_1)

# (8) 模型预测
pred.model_1 <- predict(GARCH.model_1, n.ahead = 11, trace = FALSE, mse = 'cond', plot=FALSE)
pred.model_2 <- predict(GARCH.model_2, n.ahead = 11, trace = FALSE, mse = 'cond', plot=FALSE)
pred.model_3 <- predict(GARCH.model_3, n.ahead = 11, trace = FALSE, mse = 'cond', plot=FALSE)
pred.model_4 <- predict(GARCH.model_4, n.ahead = 11, trace = FALSE, mse = 'cond', plot=FALSE)
pred.model_5 <- predict(GARCH.model_5, n.ahead = 11, trace = FALSE, mse = 'cond', plot=FALSE)
pred.model_6 <- predict(GARCH.model_6, n.ahead = 11, trace = FALSE, mse = 'cond', plot=FALSE)

predVol_1 <- pred.model_1$standardDeviation
predVol_2 <- pred.model_2$standardDeviation
predVol_3 <- pred.model_3$standardDeviation
predVol_4 <- pred.model_4$standardDeviation
predVol_5 <- pred.model_5$standardDeviation
predVol_6 <- pred.model_6$standardDeviation
et <- abs(rtm.outsample - mean(rtm.outsample))
rtd.HSI.2013 <- rtd.HSI['2013']
rv <- sqrt(aggregate(rtd.HSI.2013^2, by=substr(index(rtd.HSI.2013), 1, 7), sum))

predVol <- round(rbind(predVol_1,predVol_2,predVol_3,predVol_4,predVol_5,predVol_6, 
                       as.numeric(et), as.numeric(rv)), digits=3)
colnames(predVol) <- 1:11
rownames(predVol) <- c('GARCH(1,1)-N模型','GARCH(1,2)-N模型','GARCH(1,1)-t模型','GARCH(1,1)-st模型',
                       'GARCH(1,1)-GED模型','GARCH(1,1)-SGED模型','残差绝对值ֵ', '已实现波动')
print(predVol)

# (9) 模型选择
cor(t(predVol))

# 5. GARCH-M模型、TGARCH模型与 APARCH模型
library(rugarch)
# (1) GARCH-M模型
GARCHM.spec <- ugarchspec(variance.model=list(model='fGARCH', garchOrder=c(1,1), submodel='GARCH'), 
                          mean.model=list(armaOrder=c(0,0), include.mean=TRUE, archm=TRUE),
                          distribution.model='norm')
GARCHM.fit <- ugarchfit(GARCHM.spec, data=rtm.insample)

# (2) TGARCH模型
TGARCH.spec <- ugarchspec(variance.model=list(model='fGARCH', garchOrder=c(1,1), submodel='TGARCH'), 
                          mean.model=list(armaOrder=c(0,0), include.mean=TRUE, archm=FALSE),
                          distribution.model='norm')
TGARCH.fit <- ugarchfit(TGARCH.spec, data=rtm.insample)

# (3) APARCH模型
APARCH.model_1 <- garchFit(~1+aparch(1,1), data=rtm.insample, trace=FALSE)                    # GARCH(1,1)-N模型
summary(APARCH.model_1)
APARCH.model_2 <- garchFit(~1+aparch(1,1), data=rtm.insample, delta=2, trace=FALSE)                    # GARCH(1,1)-N模型
summary(APARCH.model_2)