### evaluate KNN classifier which will 
#   decide what kind of cancer each patient has
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# Ten real-valued features are computed for each cell nucleus:
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three
#                                                  largest values) of these features were computed for each image,
# resulting in 30 features. For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.
# All feature values are recoded with four significant digits.
library(class)
library(caret)
library(dplyr)
library(verification)
library(ggplot2)
diagnoses <- read.csv("C:/Users/user/Downloads/diagnose.csv")
str(diagnoses)
summary(diagnoses)
ggplot(diagnoses, aes(x = diagnosis, y = radius_mean, fill = diagnosis)) +
  geom_boxplot() + 
  labs(title = "Radius distribution depending on diagnosis",
       x = "Diagnosis",
       y = "Radius Mean")
# so here I was analysing boxplots to identify only staticticaly significant variables.
# I left here only one piece of code for plot, you can just change a variable in "y" field to double check the significance
# for chosen variable.
# My primary criterion is the degree of overlap: if the distribution for benign  
# and malignant diagnoses overlaps significantly, I consider the variable 
# insufficient for clear classification.
# Thus, I've got 8 "clean" explanatory variables that I will use for my first model.
# Also, I've got 3 "questionable" variables - their boxes overlap a bit, but I can still see the difference.
# I am planning to include them to my second model.

# Also, I excluded 'Perimeter' and 'Area', and kept only radius, because these variables are very close to each other
# (I mean, you have to use radius to calculate both perimeter and area, so I thought they could cause multicollinearity)

diagnoses <- diagnoses %>% 
  dplyr::select(-id, -contains("X"), -contains("Unnamed")) %>% 
  mutate(diagnosis = as.factor(diagnosis))
# converting diagnosis from character class to a factor
# unselecting columns X, id and Unnamed to avoid noise in my data set

#selected features
model_one_features <- c("radius_mean", "radius_worst", "compactness_mean", 
                        "compactness_worst", "concavity_mean", "concavity_worst", 
                        "concave.points_mean", "concave.points_worst")

# shuffling the data and splitting it
set.seed(12)
train_part <- createDataPartition(diagnoses$diagnosis, p = 0.7, list = FALSE)

train_data_raw <- diagnoses[train_part, ] %>% dplyr::select(all_of(model_one_features))
test_data_raw  <- diagnoses[-train_part, ] %>% dplyr::select(all_of(model_one_features))

train_labels <- diagnoses$diagnosis[train_part]
test_labels  <- diagnoses$diagnosis[-train_part]

scaling_recipe <- preProcess(train_data_raw, method = c("range", "zv"))

train_data_scaled <- predict(scaling_recipe, train_data_raw)
test_data_scaled  <- predict(scaling_recipe, test_data_raw)

#print(paste("NAs in the dataset", sum(is.na(train_data_scaled))))
#print(paste("NAs in the dataset:", sum(is.na(test_data_scaled))))

# setted k for 5
predictions_m1 <- knn(train = train_data_scaled, 
                      test = test_data_scaled, 
                      cl = train_labels, 
                      k = 5)


confusionMatrix(predictions_m1, test_labels, positive = "M")

# The first model with k=5 and 8 selected features showed high overall accuracy of 95.88%,
# but there were 6 false negatives.This is the most critical number. 
# Six people had malignant tumors, but the model did not detect them and sent them home as healthy
# This is critical in medical diagnostics,
# so the next models will focus on improving sensitivity. 

# SECOND MODEL
model_two_features <- c("radius_mean", "radius_worst", "compactness_mean", 
                        "compactness_worst", "concavity_mean", "concavity_worst", 
                        "concave.points_mean", "concave.points_worst", "smoothness_worst",
                        "symmetry_worst", "fractal_dimension_worst")
# as I mentioned earlier, here I've added 3 questionable variables that i was doubting about in the beginning

# Also, i will now write comments only if something new will appear 

train_data_raw2 <- diagnoses[train_part, ] %>% dplyr::select(all_of(model_two_features))
test_data_raw2 <- diagnoses[-train_part, ] %>% dplyr::select(all_of(model_two_features))

train_labels2 <- diagnoses$diagnosis[train_part]
test_labels2 <- diagnoses$diagnosis[-train_part]

scaling2 <- preProcess(train_data_raw2, method = c("range", "zv"))

train_data_scaled2 <- predict(scaling2, train_data_raw2)
test_data_scaled2 <- predict(scaling2, test_data_raw2)

predictions_m2 <- knn(train = train_data_scaled2,
                      test = test_data_scaled2,
                      cl = train_labels2,
                      k = 3)
# I've setted k to 3, because as I said this time I want to make my model more suspicious when it comes to false negative results,
# as in a medical diagnosing they are more significant,
# that's why I am making number of "neighbors" smaller
# and, yes, I do understand, that it'll most likely increase the number of false positives results,
# but this is what I have to sacrifice in this case.

confusionMatrix(predictions_m2, test_labels2, positive = "M")

# The experiment with Model No2 
# showed that adding additional variables (smoothness_worst, symmetry_worst, fractal_dimension_worst) did not improve predictive ability((((
# which is very sad
# Thus, Sensitivity remained at the same level (90.48%), while Specificity decreased!
# This confirms my initial conclusion that these variables are redundant and introduce noise into the classification process
# could be that the first model's results would be the best

# MODEL 3
model_three_features <- c("radius_mean", "radius_worst", "compactness_mean", 
                          "compactness_worst", "concavity_mean", "concavity_worst", 
                          "concave.points_mean", "concave.points_worst")
train_data_raw3 <- diagnoses[train_part, ] %>% dplyr::select(all_of(model_three_features))
test_data_raw3 <- diagnoses[-train_part, ] %>% dplyr::select(all_of(model_three_features))

train_labels3 <- diagnoses$diagnosis[train_part]
test_labels3 <- diagnoses$diagnosis[-train_part]

scaling3 <- preProcess(train_data_raw3, method = c("range", "zv"))

train_data_scaled3 <- predict(scaling3, train_data_raw3)
test_data_scaled3 <- predict(scaling3, test_data_raw3)
# sqrt(nrow(diagnoses))
predictions_m3 <- knn(train = train_data_scaled3,
                      test = test_data_scaled3,
                      cl = train_labels3,
                      k = 23)
confusionMatrix(predictions_m3, test_labels3, positive = "M")
# The value k=23 allowed the algorithm to ignore minor noise and focus on general patterns. 
# One of the patients who was mistakenly considered healthy at k=5 was correctly classified at k=23. 
# Returning to 8 clean features (without unnecessary noise from Model 2) helped model 3 see the boundary between classes more clearly.
# We saved one more patient
# And I also was able to keep the level of overall accuracy at 95.88 
# 

# so Model 3 is officialy our winner
