library(mice)

# Load data in R

train_data <- read.csv("train.csv")
test_data <- read.csv("test.csv")

# Adding Sale Price column to test data
test_data$SalePrice <- rep(NA, 1459)


# Funtion to convert NA to None or 0
na_to_val <- function (col, value ) {
  col <- as.character(col)
  col[is.na(col)] <- value
  col <- as.factor(col)
  return(col)
}

# Replace NA's with 'None' wherever required in training and test datasets
train_data$Alley <- na_to_val(train_data$Alley, 'None')
test_data$Alley <- na_to_val(test_data$Alley, 'None')

test_data$BsmtFullBath <- na_to_val(test_data$BsmtFullBath, 0)

test_data$BsmtHalfBath <- na_to_val(test_data$BsmtHalfBath, 0)

train_data$FireplaceQu <- na_to_val(train_data$FireplaceQu, 'None')
test_data$FireplaceQu <- na_to_val(test_data$FireplaceQu, 'None')

train_data$Fence <- na_to_val(train_data$Fence, 'None')
test_data$Fence <- na_to_val(test_data$Fence, 'None')

# Merge the train and test datasets after replacing NA's wherever possible
housing_data <- rbind(train_data, test_data)

# Get information regarding categorical and continuous variables and also factor levels
str(housing_data)
