# Seoul Bicycle

# library
library(spatstat); library(ROSE); library(glmnet); library(ncpen)
library(ggplot2); library(corrplot); library(lars)
library(glmnet);library(ncvreg); library(genlasso)
library(FGSG); library(grplasso)
library(xgboost); library(Matrix); 
library(dplyr); library(e1071); library(caret); library(randomForest)

# ready
rm(list=ls())
setwd("C:\\Users\\jinji\\Desktop\\2020년\\7. 데이터마이닝\\과제2_기말분석과제\\Seoul_Bicycle")
train = read.csv("train.csv", encoding = "UTF-8")
test = read.csv("test.csv", encoding = "UTF-8")
###############################################################################################
# train data : EDA 및 전처리
###############################################################################################

# EDA 및 전처리
dim(train)
dim(test)

head(train)
head(test)

summary(train)
summary(test)
str(train)
str(test)

# check na
sum(is.na(train))     # 300개
sum(is.na(test))      # 113개

sum(is.na(train[,1])) # id에는 결측치 없음 확인
sum(is.na(train[,2])) # hour 에도 결측치 없음 확인

sum(is.na(test[,1])) # id에는 결측치 없음 확인
sum(is.na(test[,2])) # id에는 결측치 없음 확인

sum(is.na(train[,11])) # count에는 결측치 없음 확인


# na 처리
# 그 시간대 평균값으로 대체 : hour 는 0~23 사이 값
# complete.cases(train)      # na 확인


# 변수별 na, 시간대별 mean으로 대체
names(train)
# 기온
train[is.na(train$hour_bef_temperature),]      # hour, count 빼고 모두 na인 행 -> 935,1036
train = train[-c(935,1036),]       # 행 제거
sum(is.na(train$hour_bef_temperature))

# 비
sum(is.na(train[,4]))

# 풍속
sum(is.na(train[,5]))      # 7
train[is.na(train[,5]),]     # 행확인 -> 시간대 hour : 13,1,3,0,20,12,2
for (i in 0:24){
  train[,5][train$hour==i][is.na(train[,5][train$hour==i])] = mean(train[,5][train$hour==i],na.rm = TRUE)
}

# 습도
sum(is.na(train[,6])) 

# 가시성
sum(is.na(train[,7]))      

# 오존
sum(is.na(train[,8]))      # 74
train[is.na(train[,8]),]     # 행확인 -> 시간대 hour : 1,17,16,10,18,12,7,11,15,6,13
train[,8][train$hour==1][is.na(train[,8][train$hour==1])] = mean(train[,8][train$hour==0|train$hour==2],na.rm = TRUE)     
# 1시 모두 nan -> 0시,2시의 평균값으로 대체
for (i in 2:24){
  train[,8][train$hour==i][is.na(train[,8][train$hour==i])] = mean(train[,8][train$hour==i],na.rm = TRUE)
}

# 미세먼지
sum(is.na(train[,9]))      # 88
train[is.na(train[,9]),]     # 행확인 -> 시간대 hour : 1,16,19,10,15,13,12,11,20,18,17,14
train[,9][train$hour==1][is.na(train[,9][train$hour==1])] = mean(train[,9][train$hour==0|train$hour==2],na.rm = TRUE)     
# 1시 모두 nan -> 0시,2시의 평균값으로 대체
for (i in 2:24){
  train[,9][train$hour==i][is.na(train[,9][train$hour==i])] = mean(train[,9][train$hour==i],na.rm = TRUE)
}    

# 초미세먼지지
sum(is.na(train[,10]))      # 115
train[is.na(train[,10]),]     # 행확인 -> 시간대 hour : 0, 1,21,16,19,10,17,15,13,12,7,11,14,6,5,18,8,23
train[,10][train$hour==1][is.na(train[,10][train$hour==1])] = mean(train[,10][train$hour==0|train$hour==2],na.rm = TRUE)     
# 1시 모두 nan -> 0시,2시의 평균값으로 대체
train[,10][train$hour==0][is.na(train[,10][train$hour==0])] = mean(train[,10][train$hour==0],na.rm = TRUE)     

for (i in 2:24){
  train[,10][train$hour==i][is.na(train[,10][train$hour==i])] = mean(train[,10][train$hour==i],na.rm = TRUE)
}


# 함수짰는데 안됨
na_mean = function(x,var){
  for (i in 0:24){
    na_mean = mean(x$var[x$hour==i],na.rm = TRUE)
    x$var[x$hour==i][is.na(x$var[x$hour==i])] = na_mean
  }
}

# feature engineering
# train['dust'] = train$hour_bef_pm10 + train$hour_bef_pm2.5     # 총 미세먼지

# 최종 차원 확인
dim(test) # 1459 -> 1467

###############################################################################################
# test data, na 처리
###############################################################################################
# 변수별 na, 시간대별 mean으로 대체
names(test)
# 기온
test[is.na(test[,3]),]    # hour, count 빼고 모두 na인 행 -> 해당 시간대 평균으로 대체
test[,3][test$hour==19][is.na(test[,3][test$hour==19])] = mean(test[,3][test$hour==19],na.rm = TRUE)     
sum(is.na(test$hour_bef_temperature))

# 비
sum(is.na(test[,4]))
test[is.na(test[,4]),]     
test[,4][test$hour==19][is.na(test[,4][test$hour==19])] = 0     # 안온척

# 풍속
sum(is.na(test[,5]))      
test[is.na(test[,5]),]     
test[,5][test$hour==19][is.na(test[,5][test$hour==19])] = mean(test[,5][test$hour==19],na.rm = TRUE)     

# 습도
sum(is.na(test[,6])) 
test[,6][test$hour==19][is.na(test[,6][test$hour==19])] = mean(test[,6][test$hour==19],na.rm = TRUE)     

# 가시성
sum(is.na(test[,7]))      
test[,7][test$hour==19][is.na(test[,7][test$hour==19])] = mean(test[,7][test$hour==19],na.rm = TRUE)     

# 오존
sum(is.na(test[,8]))      # 35
test[is.na(test[,8]),]     
test[,8][test$hour==1][is.na(test[,8][test$hour==1])] = mean(test[,8][test$hour==0|test$hour==2],na.rm = TRUE)     
# 1시 모두 nan -> 0시,2시의 평균값으로 대체
for (i in 2:24){
  test[,8][test$hour==i][is.na(test[,8][test$hour==i])] = mean(test[,8][test$hour==i],na.rm = TRUE)
}

# 미세먼지
sum(is.na(test[,9]))      # 88
test[is.na(test[,9]),]     # 행확인 -> 시간대 hour : 1,16,19,10,15,13,12,11,20,18,17,14
test[,9][test$hour==1][is.na(test[,9][test$hour==1])] = mean(test[,9][test$hour==0|test$hour==2],na.rm = TRUE)     
# 1시 모두 nan -> 2시의 평균값으로 대체
for (i in 2:24){
  test[,9][test$hour==i][is.na(test[,9][test$hour==i])] = mean(test[,9][test$hour==i],na.rm = TRUE)
}    

# 초미세먼지지
sum(is.na(test[,10]))      # 115
test[is.na(test[,10]),]     # 행확인 -> 시간대 hour : 0, 1,21,16,19,10,17,15,13,12,7,11,14,6,5,18,8,23
test[,10][test$hour==1][is.na(test[,10][test$hour==1])] = mean(test[,10][test$hour==0|test$hour==2],na.rm = TRUE)     
# 1시 모두 nan -> 2시의 평균값으로 대체
for (i in 2:24){
  test[,10][test$hour==i][is.na(test[,10][test$hour==i])] = mean(test[,10][test$hour==i],na.rm = TRUE)
}

sum(is.na(test))     # 확인
dim(test)      # 715,10

###############################################################################################
# train data : Visualization 
###############################################################################################
# train data , 각 변수별 plot
par(mfrow=c(2,2))
plot(train$hour_bef_temperature, train$count, xlab = '기온', ylab = 'count')      # 기온
plot(train$hour_bef_windspeed, train$count, xlab = '풍속', ylab = 'count')      # 풍속
plot(train$hour_bef_humidity, train$count, xlab = '습도', ylab = 'count')      # 습도
plot(train$hour_bef_visibility, train$count, xlab = '가시성', ylab = 'count')      # 가시성
plot(train$hour_bef_ozone, train$count, xlab = '오존', ylab = 'count')      # 오존
plot(train$hour_bef_pm10, train$count, xlab = '미세먼지', ylab = 'count')      # 미세먼지
plot(train$hour_bef_pm2.5, train$count, xlab = '초미세먼지', ylab = 'count')      # 초미세먼지

# 변수별 정규성 Noro QQ plot
par(mfrow=c(2,2))
qqnorm(train[,2], main='hour')
qqline(train[,2],col="red",lwd=2)
qqnorm(train[,3], main='temperature')
qqline(train[,3],col="red",lwd=2)
qqnorm(train[,5], main='windspeed')
qqline(train[,5],col="red",lwd=2)
qqnorm(train[,6], main='humidity')
qqline(train[,6],col="red",lwd=2)
qqnorm(train[,7], main='visibility')
qqline(train[,7],col="red",lwd=2)
qqnorm(train[,8], main='ozone')
qqline(train[,8],col="red",lwd=2)
qqnorm(train[,9], main='pm10')
qqline(train[,9],col="red",lwd=2)
qqnorm(train[,10], main='pn2.5')
qqline(train[,10],col="red",lwd=2)
qqnorm(train[,11], main='count')
qqline(train[,11],col="red",lwd=2)

# humidity, visibility, pm10, pm2.5, count 정규성 만족 x

# 상관분석 시각화 : train, test
#install.packages('corrplot')
par(mfrow=c(1,1))
plot(train[,-c(1,4)])
corrplot(cor(train[,-c(1,4)]))


# 시간별 시각화
par(mfrow=c(1,1))
#theme_set(theme_minimal()) #ggplot2 background annotation 최소화
ggplot(data = train, aes(x = hour, y = count))+
  geom_line(color = "#00AFBB", size = 1)
# 시간대 0~24시별 count라 많이 의미있어보이지는 않음

###############################################################################################
# Loss, RMSE 사용
###############################################################################################
loss = function(y_pred, y){
  return(sqrt(sum((y_pred - y)^2) / length(y)))
}

###############################################################################################
# train data : linear regression + variable selection
# ref : 권성훈교수님
###############################################################################################
# 변수 정의
X = as.matrix(train[,-c(1,11)])     # id, target 인 count 뺌뺌
y = as.matrix(train[,11])       # count
dim(X);length(y)
train_df = data.frame(X,y)
head(train_df)
names(X)

# multiple linear regression
fit = lm(y~., data = train_df)
summary(fit)
fit$coefficients      # full model

###############################################################################################
# train data : variable selection without tunning
###############################################################################################

# lasso regression
#install.packages("lars")
fit_lasso = lars(as.matrix(X),as.matrix(y),type='lasso',trace=TRUE)
summary(fit_lasso)
plot(fit_lasso, plottype="Cp")
getwd()
source("../../data.mining.functions.R")     # 교수님 function불러오기

b.mat =NULL
y_pred=NULL
# 1. AIC
fit = forward.fun(y,X,inf.wt = 2, trace=F)
opt = dim(fit$coef.mat)[2]
b.mat = cbind(b.mat,fit$coef.mat[,opt])
beta = fit$coef.mat[,opt]
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
y_pred = cbind(y_pred,y_fitted.values)
# 2. BIC
fit = forward.fun(y,X,inf.wt = log(nrow(X)), trace=F)
opt = dim(fit$coef.mat)[2]
b.mat = cbind(b.mat,fit$coef.mat[,opt])
beta = fit$coef.mat[,opt]
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
y_pred = cbind(y_pred,y_fitted.values)
# 3. GIC : LASSO + AIC
#?ncpen
fit = ncpen(y,X,penalty="lasso")
aic = gic.ncpen(fit,weight = 2)
opt = which.min(aic$gic) # 사실 aic
b.mat = cbind(b.mat, fit$beta[,opt])
beta = fit$beta[,opt]
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
y_pred = cbind(y_pred,y_fitted.values)
# 4. GIC : ridge + AIC
fit = ncpen(y,X,penalty="ridge")
aic = gic.ncpen(fit,weight = 2)
opt = which.min(aic$gic) # 사실 aic
b.mat = cbind(b.mat, fit$beta[,opt])
beta = fit$beta[,opt]
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
y_pred = cbind(y_pred,y_fitted.values)
# 5. GIC : ridge + BIC
fit = ncpen(y,X,penalty="ridge")
aic = gic.ncpen(fit,weight = log(nrow(X)))
opt = which.min(aic$gic) # 사실 Bic
b.mat = cbind(b.mat, fit$beta[,opt])
beta = aic$opt.beta
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
y_pred = cbind(y_pred,y_fitted.values)
# 6. GIC : SCAD + BIC
fit = ncpen(y,X,penalty="scad")
aic = gic.ncpen(fit,weight = log(nrow(X)))
opt = which.min(aic$gic) # 사실 Bic
b.mat = cbind(b.mat, fit$beta[,opt])
beta = aic$opt.beta
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
y_pred = cbind(y_pred,y_fitted.values)
# 7. Stepwise + AIC
reduced.model=step(fit,direction="both")
summary(reduced.model)
reduced.model$fitted.values
reduced.model$fitted.values[reduced.model$fitted.values<0]=0
r.y_pred = round(reduced.model$fitted.values)
y_pred = cbind(y_pred, r.y_pred)

# output
b.mat      # 6가지 모형 beta 저장됨
y_pred     # 7가지 모형의 예측값 저장됨

# 7가지 모델에 대한 평가 -> Loss 사용
loss.mat = NULL
loss.mat = cbind(loss.mat,loss(y_pred[,1],y))
loss.mat = cbind(loss.mat,loss(y_pred[,2],y))
loss.mat = cbind(loss.mat,loss(y_pred[,3],y))
loss.mat = cbind(loss.mat,loss(y_pred[,4],y))
loss.mat = cbind(loss.mat,loss(y_pred[,5],y))
loss.mat = cbind(loss.mat,loss(y_pred[,6],y))
loss.mat = cbind(loss.mat,loss(y_pred[,7],y))
min(loss.mat)     #51.97449 모델 1 = 모델 2


###############################################################################################
# train data : k-fold Cross Validation with tunning
###############################################################################################
# 10-fold cross validation 실시
# ready
source("../../data.mining.functions.2020.0602.R")     # 교수님 function불러오기
set.seed(1004)
X = as.matrix(train[,-c(1,11)])     
y = as.matrix(train[,11])       

cv.id = cv.index.fun(y, k.val=10); cv.id

m.vec = c("cv-ridge","cv-lasso","cv-scad","cv-mbridge")
cv.b.mat = matrix(0, nrow=ncol(X)+1, ncol=length(m.vec))     
colnames(cv.b.mat) = m.vec
cv.y_pred=NULL

### 원래 로스52.14543
# 수정 로스51.98341
# ridge
cv.fit = cv.glmnet(X, y, family='gaussian', foldid=cv.id, alpha=0, lambda.min.ratio = 1e-5)
names(cv.fit)
summary(cv.fit)
plot(cv.fit$cvm)        
opt = which.min(cv.fit$cvm)
abline(v=opt, col=2)
cv.b.mat[,'cv-ridge'] = coef(cv.fit$glmnet.fit)[,opt]
beta = coef(cv.fit$glmnet.fit)[,opt]
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
cv.y_pred = cbind(cv.y_pred,y_fitted.values)
# LASSO
cv.fit = cv.glmnet(X, y, family='gaussian', foldid=cv.id)
plot(cv.fit$cvm)        # what is cvm
opt = which.min(cv.fit$cvm)
abline(v=opt, col=2)
cv.b.mat[,'cv-lasso'] = coef(cv.fit$glmnet.fit)[,opt]
beta = coef(cv.fit$glmnet.fit)[,opt]
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
cv.y_pred = cbind(cv.y_pred,y_fitted.values)
# SCAD
cv.fit = cv.ncpen(y, X, family='gaussian', penalty= 'scad', fold.id = cv.id)
names(cv.fit)
plot(cv.fit$rmse)        
opt = which.min(cv.fit$rmse)
abline(v=opt, col=2)
cv.b.mat[,'cv-scad'] = coef(cv.fit$ncpen.fit)[,opt]
beta = coef(cv.fit$ncpen.fit)[,opt]
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
cv.y_pred = cbind(cv.y_pred,y_fitted.values)
# bridge
cv.fit = cv.ncpen(y, X, family='gaussian', penalty = 'mbridge', fold.id = cv.id)
names(cv.fit)
plot(cv.fit$rmse)
abline(v=opt, col=2)
cv.b.mat[,'cv-mbridge'] = coef(cv.fit$ncpen.fit)[,opt]
beta = coef(cv.fit$ncpen.fit)[,opt]
y_fitted.values = cbind(1,as.matrix(X))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
cv.y_pred = cbind(cv.y_pred,y_fitted.values)
# cv 한 4 가지 모델 loss로 평가
cv.loss.mat = NULL
cv.loss.mat = cbind(cv.loss.mat,loss(cv.y_pred[,1],y))
cv.loss.mat = cbind(cv.loss.mat,loss(cv.y_pred[,2],y))
cv.loss.mat = cbind(cv.loss.mat,loss(cv.y_pred[,3],y))
cv.loss.mat = cbind(cv.loss.mat,loss(cv.y_pred[,4],y))

min(cv.loss.mat)     # 2번째 모델 -> LASSO 51.98341

# total loss
total_loss = cbind(loss.mat, cv.loss.mat)
plot(col(total_loss),total_loss)

###############################################################################################
# tunning model 을 test data에 적용하기
###############################################################################################
# 2번째 모델 -> LASSO 사용
X_test = test[,-c(1)]
y_test = NULL
# LASSO
cv.fit = cv.glmnet(X, y, family='gaussian', foldid=cv.id)
opt = which.min(cv.fit$cvm)
beta = coef(cv.fit$glmnet.fit)[,opt]
y_fitted.values = cbind(1,as.matrix(X_test))%*%as.matrix(beta)
y_fitted.values[y_fitted.values<0]=0
y_fitted.values = round(y_fitted.values)
y_test_pred = y_fitted.values
dim(y_test_pred)

submission = read.csv("submission.csv")
submission['count'] = y_test_pred
dim(submission)
write.csv(submission,"submission02.csv",row.names = FALSE)

# 31/33위 ^^

###############################################################################################
# SVM regression
# ref : https://hospital82.tistory.com/136
###############################################################################################
# ready
#rm(list=ls())
#set.seed(Sys.Date())

## 위의 전처리 거친 train X,y data 사용
X = train[,-c(1,11)]
y = train[,11]
dim(X); length(y)
df = data.frame(X,y)
str(df)

# data split for Cross validation
par = createDataPartition(df$hour, p=0.7, list=FALSE)
train.data = df[par,]
test.data = df[-par,]

# SVM model fitting
# 1450/100 = 15번 -> 최대10번 f-fold tune 수행
num.data = nrow(train.data)
if ((num.data/100) <= 10){
  num.cross = round((num.data/100),0)
}else{
  num.cross = 10
}

x.train = subset(train.data,select = c(hour,hour_bef_temperature,hour_bef_precipitation,hour_bef_windspeed,
                                      hour_bef_humidity,hour_bef_visibility,hour_bef_ozone,
                                       hour_bef_pm10,hour_bef_pm2.5))
y.train = subset(train.data, select = c(y))
x.test = subset(test.data,select = c(hour,hour_bef_temperature,hour_bef_precipitation,hour_bef_windspeed,
                                      hour_bef_humidity,hour_bef_visibility,hour_bef_ozone,
                                      hour_bef_pm10,hour_bef_pm2.5))
y.test = subset(test.data, select = c(y))


# Final Model fit
?svm
fit.svm = svm(x = as.matrix(x.train), y = as.vector(y.train),
              type = 'eps-regression', kernerl = 'radial', epsilon = 0.01, cross=15)
svm.fore = predict(fit.svm, newdata = as.matrix(x.test))

# fitting model 확인
str(fit.svm)
fit.svm$fitted
# 시각화
output = list()
diff = y.test - svm.fore
diff_ratio = abs((y.test-svm.fore)/y.test)
output = cbind(y.test, svm.fore, diff, diff_ratio)
colnames(output) = c('count','fore','SE','APE')
rmse = sqrt(mean((output$SE)^2))
print(rmse)      # 54.74347

mape = mean(output$APE)
print(mape)      # 0.7507015

plot(output$count, type = 'l', col='gray')
lines(output$fore, ttype = 'l', col='blue', iwd='2')

# fitting model 실제 train 데이터에 적용
X_test = test[,-c(1)]
dim(X_test)
dim(x.test)
names(x.test)
names(X_test)
svm.final = predict(fit.svm, newdata = as.matrix(X_test))
# 저장
submission = read.csv("submission.csv")
submission['count'] = svm.final
write.csv(submission,"submission03.csv",row.names = FALSE)

## 30/33 linear regression 보다는 성능이 좋지만 개선 필요 

################################################################################################
# svm tunning : kernel (radial, sigmoid) 과 epsilon (0.01, 0.1) 조절
###############################################################################################
fit.svm.sig.eps01 = svm(x = as.matrix(x.train), y = as.vector(y.train),
              type = 'eps-regression', kernerl = 'sigmoid', epsilon = 0.1, cross=10)
svm.sig.eps01 = predict(fit.svm.sig.eps01, newdata = as.matrix(x.test))

# fitting model 확인
str(fit.svm.sig)
fit.svm.sig$fitted
# 시각화
output = list()
diff = y.test - svm.sig.eps01
diff_ratio = abs((y.test-svm.sig.eps01)/y.test)
output = cbind(y.test, svm.sig.eps01, diff, diff_ratio)
colnames(output) = c('count','fore','SE','APE')
rmse = sqrt(mean((output$SE)^2))
print(rmse)      # 54.74347 -> 46.08594

mape = mean(output$APE)
print(mape)      # 0.7507015 -> 0.866419

plot(output$count, type = 'l', col='gray')
lines(output$fore, type = 'l', col='blue')

# fitting model 실제 test 데이터에 적용
## 실제 training data 모두 사용해서 training시킴 
## tunning : sigmoid, eps = 0.1
full.svm = svm(x = as.matrix(X), y = as.vector(y),
                        type = 'eps-regression', kernerl = 'sigmoid', epsilon = 0.1, cross=10)
svm_pred = predict(full.svm, newdata = as.matrix(X_test))
y_pred = round(svm_pred)

# 저장
submission = read.csv("submission.csv")
submission['count'] = y_pred
write.csv(submission,"submission08_full_svm.csv",row.names = FALSE)

########### sigmoid, eps = 0.1 이 성능 가장 좋음, eps 낮을 때, overfitting 되는듯
## full data 사용한 svm data 오버피팅 되는 듯, rmse 떨어짐...

########################################################################
# randomForest without tunning
########################################################################
# 변수정의
X = as.matrix(train[,-c(1,11)]) 
y = as.matrix(train[,11])
X_test = test[,-c(1)]
dim(X);length(y)
train_df = data.frame(X,y)

# data split for Cross validation
par = createDataPartition(train_df$hour, p=0.7, list=FALSE)
train.data = train_df[par,]
test.data = train_df[-par,]

x.train = subset(train.data,select = c(hour,hour_bef_temperature,hour_bef_precipitation,hour_bef_windspeed,
                                       hour_bef_humidity,hour_bef_visibility,hour_bef_ozone,
                                       hour_bef_pm10,hour_bef_pm2.5))
y.train = subset(train.data, select = c(y))
x.test = subset(test.data,select = c(hour,hour_bef_temperature,hour_bef_precipitation,hour_bef_windspeed,
                                     hour_bef_humidity,hour_bef_visibility,hour_bef_ozone,
                                     hour_bef_pm10,hour_bef_pm2.5))
y.test = subset(test.data, select = c(y))

df_train = data.frame(x.train, y.train)
rf.bicycle = randomForest(y~., data = df_train, ntree = 100, mtry = 5, importance = T)
summary(rf.bicycle)                          
rf.bicycle.test = predict(rf.bicycle, newdata = as.matrix(x.test))
loss(rf.bicycle.test, y.test)     # 806.8469

# full data 사용
df_train = data.frame(X, y)
rf.bicycle = randomForest(y~., data = df_train, ntree = 100, mtry = 5, importance = T)
rf.bicycle.test = predict(rf.bicycle, newdata = as.matrix(X_test))
y_pred = round(rf.bicycle.test)

# 저장
submission = read.csv("submission.csv")
submission['count'] = y_pred
write.csv(submission,"submission08_full_rf.csv",row.names = FALSE)

########################################################################
# xgboost with grid search
# ref : https://www.kaggle.com/silverstone1903/xgboost-grid-search-r
# ref : https://apple-rbox.tistory.com/6
########################################################################
X = train[,-c(1,11)]
y = train[,11]
par = createDataPartition(train_df$hour, p=0.7, list=FALSE)
train.data = train_df[par,]
test.data = train_df[-par,]

x.train = subset(train.data,select = c(hour,hour_bef_temperature,hour_bef_precipitation,hour_bef_windspeed,
                                       hour_bef_humidity,hour_bef_visibility,hour_bef_ozone,
                                       hour_bef_pm10,hour_bef_pm2.5))
y.train = subset(train.data, select = c(y))
x.test = subset(test.data,select = c(hour,hour_bef_temperature,hour_bef_precipitation,hour_bef_windspeed,
                                     hour_bef_humidity,hour_bef_visibility,hour_bef_ozone,
                                     hour_bef_pm10,hour_bef_pm2.5))
y.test = subset(test.data, select = c(y))

# xgboost with cv
xg_model = xgb.cv(data = as.matrix(x.train), label = as.matrix(y.train),
                  nfold=10, nrounds=200,early_stopping_rounds = 150, 
                  objecttive = 'reg:linear', verbose=T, prediction = T)
cvplot(xg_model)

#visualizing model
cvplot = function(model){ #visualizing function
  eval.log = model$evaluation_log
  
  std = names(eval.log[,2]) %>% gsub('train_','',.) %>% gsub('_mean','',.)
  
  data.frame(error = c(unlist(eval.log[,2]),unlist(eval.log[,4])),
             class = c(rep('train',nrow(eval.log)),
                       rep('test',nrow(eval.log))),
             nround = rep(1:nrow(eval.log),2)
  ) %>%
    ggplot(aes(nround,error,col = class))+
    geom_point(alpha = 0.2)+
    geom_smooth(alpha = 0.4,se = F)+
    theme_bw()+
    ggtitle("XGBoost Cross-validation Visualization",
            subtitle = paste0('fold : ',length(model$folds),
                              '  iteration : ',model$niter
            )
    )+ylab(std)+theme(axis.title=element_text(size=11))
}

# full data xgboost with cv
xg_model_full = xgb.cv(data = as.matrix(X), label = as.matrix(y),
                  nfold=10, nrounds=200,early_stopping_rounds = 150, 
                  objecttive = 'reg:linear', verbose=T, prediction = T)
# 최종 모델
xg_model_full = xgboost(data = as.matrix(X), label = as.matrix(y),
                       nfold=10, nrounds=200,early_stopping_rounds = 150, 
                       objecttive = 'reg:linear', verbose=T, prediction = T)

loss(y,xg_model_full$pred)
y_pred = predict(xg_model_full, data.matrix(X_test))
y_pred = round(y_pred)
loss(y_test,y_pred)

# 저장
submission = read.csv("submission.csv")
submission['count'] = y_pred
write.csv(submission,"submission09_xgb_round.csv",row.names = FALSE)
