library(mice)

# Load data in R

train_data <- read.csv("train.csv")
test_data <- read.csv("test.csv")

# Adding Sale Price column to test data
test_data$SalePrice <- rep(NA, 1459)


# Funtion to convert NA to None
na_to_none <- function (col) {
  col <- as.character(col)
  col[is.na(col)] <- 'None'
  col <- as.factor(col)
  return(col)
}

# Funtion to convert NA to 0
na_to_zero <- function (col) {
  col[is.na(col)] <- 0
  return(col)
}

# Replace NA's with 'None' wherever required in training and test datasets
train_data$Alley <- na_to_none(train_data$Alley)
test_data$Alley <- na_to_none(test_data$Alley)

test_data$BsmtFullBath <- na_to_zero(test_data$BsmtFullBath)

test_data$BsmtHalfBath <- na_to_zero(test_data$BsmtHalfBath)

train_data$FireplaceQu <- na_to_none(train_data$FireplaceQu)
test_data$FireplaceQu <- na_to_none(test_data$FireplaceQu)

train_data$Fence <- na_to_none(train_data$Fence)
test_data$Fence <- na_to_none(test_data$Fence)

train_data$PoolQC <- na_to_none(train_data$PoolQC)

test_data$PoolQC <- as.character(test_data$PoolQC)
test_data$PoolQC[(test_data$PoolArea == 0) & is.na(test_data$PoolQC)] <- 'None'
test_data$PoolQC <- as.factor(test_data$PoolQC)

train_data$MiscFeature <- na_to_none(train_data$MiscFeature)

test_data$MiscFeature <- as.character(test_data$MiscFeature)
test_data$MiscFeature[(test_data$MiscVal == 0) & is.na(test_data$MiscFeature)] <- 'None'
test_data$MiscFeature <- as.factor(test_data$MiscFeature)

train_data$MasVnrType <- na_to_none(train_data$MasVnrType)
# None is already a factor in test data for the below column
test_data$MasVnrType[is.na(test_data$MasVnrArea) & is.na(test_data$MasVnrType)] <- 'None'

train_data$MasVnrArea <- na_to_zero(train_data$MasVnrArea)
test_data$MasVnrArea <- na_to_zero(test_data$MasVnrArea)

# Basement related attributes for training data
train_data$BsmtQual <- na_to_none(train_data$BsmtQual)
train_data$BsmtCond <- na_to_none(train_data$BsmtCond)

train_data$BsmtExposure <- as.character(train_data$BsmtExposure)
train_data$BsmtExposure[(train_data$TotalBsmtSF == 0) & is.na(train_data$BsmtExposure)] <- 'None'
train_data$BsmtExposure <- as.factor(train_data$BsmtExposure)

train_data$BsmtFinType1 <- na_to_none(train_data$BsmtFinType1)

train_data$BsmtFinType2 <- as.character(train_data$BsmtFinType2)
train_data$BsmtFinType2[(train_data$TotalBsmtSF == 0) & is.na(train_data$BsmtFinType2)] <- 'None'
train_data$BsmtFinType2 <- as.factor(train_data$BsmtFinType2)

# Basement related attributes for test data
test_data$TotalBsmtSF <- na_to_zero(test_data$TotalBsmtSF)
test_data$BsmtFinSF1 <- na_to_zero(test_data$BsmtFinSF1)
test_data$BsmtFinSF2 <- na_to_zero(test_data$BsmtFinSF2)
test_data$BsmtUnfSF <- na_to_zero(test_data$BsmtUnfSF)


test_data$BsmtQual <- as.character(test_data$BsmtQual)
test_data$BsmtQual[(test_data$TotalBsmtSF == 0) & is.na(test_data$BsmtQual)] <- 'None'
test_data$BsmtQual <- as.factor(test_data$BsmtQual)

test_data$BsmtCond <- as.character(test_data$BsmtCond)
test_data$BsmtCond[(test_data$TotalBsmtSF == 0) & is.na(test_data$BsmtCond)] <- 'None'
test_data$BsmtCond <- as.factor(test_data$BsmtCond)

test_data$BsmtExposure <- as.character(test_data$BsmtExposure)
test_data$BsmtExposure[(test_data$TotalBsmtSF == 0) & is.na(test_data$BsmtExposure)] <- 'None'
test_data$BsmtExposure <- as.factor(test_data$BsmtExposure)

test_data$BsmtFinType1 <- na_to_none(test_data$BsmtFinType1)

test_data$BsmtFinType2 <- na_to_none(test_data$BsmtFinType2)

# Garage related attributes for training data
train_data$GarageType <- na_to_none(train_data$GarageType)
# Yet to decide whether to factor the year column or not
# train_data$GarageYrBlt <- na_to_zero(train_data$GarageYrBlt)
train_data$GarageYrBlt <- na_to_none(train_data$GarageYrBlt)
train_data$GarageFinish <- na_to_none(train_data$GarageFinish)
train_data$GarageQual <- na_to_none(train_data$GarageQual)
train_data$GarageCond <- na_to_none(train_data$GarageCond)

# Garage related attributes for test data

test_data$GarageYrBlt <- as.character(test_data$GarageYrBlt)
test_data$GarageYrBlt[is.na(test_data$GarageType) & is.na(test_data$GarageYrBlt)] <- 'None'
test_data$GarageYrBlt <- as.factor(test_data$GarageYrBlt)

test_data$GarageFinish <- as.character(test_data$GarageFinish)
test_data$GarageFinish[is.na(test_data$GarageType) & is.na(test_data$GarageFinish)] <- 'None'
test_data$GarageFinish <- as.factor(test_data$GarageFinish)

test_data$GarageQual <- as.character(test_data$GarageQual)
test_data$GarageQual[is.na(test_data$GarageType) & is.na(test_data$GarageQual)] <- 'None'
test_data$GarageQual <- as.factor(test_data$GarageQual)

test_data$GarageCond <- as.character(test_data$GarageCond)
test_data$GarageCond[is.na(test_data$GarageType) & is.na(test_data$GarageCond)] <- 'None'
test_data$GarageCond <- as.factor(test_data$GarageCond)

test_data$GarageType <- na_to_none(test_data$GarageType)

# Merge the train and test datasets after replacing NA's wherever possible
housing_data <- rbind(train_data, test_data)

# Get information regarding categorical and continuous variables and also factor levels
summary(housing_data)

# Impute the remaining missing values using mice
imp_housing_data <- mice(housing_data, m=1, method='cart', printFlag = FALSE)

# Above takes a very long time so instead perform the operation separately on training and test data

imp_train_data <- mice(train_data, m=1, method='cart', printFlag = FALSE)
imp_test_data <- mice(test_data, m=1, method='cart', printFlag = FALSE)

train_comp <- complete(imp_train_data)
test_comp <- complete(imp_test_data)

