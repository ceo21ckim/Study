#4.17 Data
#3.29file / 4.15outlier
fit <- lm(mpg ~ wt, data=mtcars)
hatvalues(fit) #Hii
head(mtcars)
summary(mtcars)
nrow(mtcars) #32�� data
4/32 #p+1 / n (p=1,n=32)���� ū ���� outlier�� Ȯ���� Ŀ
which.max(hatvalues(fit)) #which.max�Ἥ ã��. 
which(hatvalues(fit)>0.125)
lm.influence(fit)$hat 
#lm.influence(fit)$sigma = sigma.hat(-i) ���̵�
rstandard(fit) #studentized residaul (ri)
residuals(fit)/(sqrt(1-hatvalues(fit))*sigma(fit)) #studentized residaul (ri)
rstudent(fit) #extrenally studentized residual (ti)
