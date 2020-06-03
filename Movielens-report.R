## Katie Sharp
## MovieLens Project 
## HarvardX:PH125.9x


#############################################################
# Data Import and Preprocessing 
#############################################################

# Load the required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

# Download the movielens dataset
  # https://grouplens.org/datasets/movielens/10m/ 
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip  

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies, stringsAsFactors = TRUE) %>%
  mutate(movieId = as.numeric(levels(movieId))[movieId],
         title = as.character(title), 
         genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Examine overall structure of movielens
str(movielens)

# Find number of distinct movies, users, genres
movielens %>% summarise(n_movies = n_distinct(movieId),
                        n_users = n_distinct(userId),
                        n_genres = n_distinct(genres))

# Important summary information for movielens
summary(movielens)

# Create validation and edx datasets
set.seed(1, sample.kind="Rounding")
# Validation set is 10% of movielens dataset
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
 edx <- movielens[-test_index,]
 temp <- movielens[test_index,]
# Make sure movieId and userId in validation set are also in edx set 
validation <- temp %>% 
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")
# Add rows removed from validation set back into edx subset
removed <- anti_join(temp, validation)
 edx <- rbind(edx, removed)
# Remove files not needed
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#############################################################
# Data Exploration and Visualization
#############################################################

# Examine the first few rows of edx
head(edx)

# Convert timestamp column to better date format, add two columns for year_rated and year_released
edx_new <- edx %>% mutate(timestamp = as_datetime(timestamp), year_rated = year(as_datetime(timestamp)), year_release = as.numeric(str_sub(title,-5,-2)))
head(edx_new)

# Structure of our new edx dataset
str(edx_new)

### Ratings ###
# Examine Ratings distribution table
edx_new %>% group_by(rating) %>%
  summarise(n = n())

# Calculate share of ratings that are above or below 3
mean(edx_new$rating > 3)
mean(edx_new$rating < 3)

# Calculate share of ratings that are either 3 or 4
mean(edx_new$rating ==3 | edx_new$rating ==4)

# Plot the number of movies in each rating value
ratings_plot <- as.vector(edx_new$rating)
ratings_plot <- ratings_plot[ratings_plot != 0]
ratings_plot <- factor(ratings_plot)
qplot(ratings_plot) +
  xlab("Rating") + ylab("Number of Movies") +
  ggtitle("Ratings Distribution")

### Movies ###
# Plot histogram of how often movies get rated 
edx_new %>% group_by(movieId) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(fill = "blue", color = "black") +
  scale_x_log10() +
  xlab("Number of Ratings") + ylab("Number of Movies") +
  ggtitle("Movies")

# Top 10 movies with the most ratings per year since its release
edx_new %>% group_by(movieId) %>%
  summarize(n = n(), title = title[1], years = 2018 - first(year_release), avg_rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  top_n(10, rate) %>%
  arrange(desc(rate))

### Users ###
# Plot histogram of how often users rate movies 
edx_new %>% group_by(userId) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(fill = "blue", color = "black") +
  scale_x_log10() +
  xlab("Number of Ratings") + ylab("Number of Users") +
  ggtitle("Users")

# Plot histogram of frequency of average ratings
edx_new %>%
  group_by(userId) %>%
  summarize(Avg_Rating = mean(rating)) %>%
  ggplot(aes(Avg_Rating)) +
  geom_histogram(color = "black", fill = "blue") +
  xlab("Avg Rating") + ylab("Number of Users") +
  ggtitle("User Avg Rating")

### Genres ###
# Top 10 genres ordered by average rating
edx_new %>% group_by(genres) %>%
  summarize(n = n(), avg_rating = mean(rating)) %>%
  filter(n >= 1000) %>%
  top_n(10, avg_rating)

# Bottom 10 genres ordered by average rating
edx_new %>% group_by(genres) %>%
  summarize(n = n(), avg_rating = mean(rating)) %>%
  filter(n >= 1000) %>%
  top_n(-10, avg_rating)

### Time ###
# Plot Average rating vs Year of when movie was released
edx_new %>% group_by(year_release) %>%
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(x=year_release, y = avg_rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Time: Age of Movie")

# Plot Average rating vs Year of when movie was rated
edx_new %>% filter(year_release >= 1995) %>%
  mutate(date_diff = year_rated - year_release) %>%
  group_by(date_diff) %>%
  summarise(avg_rating = mean(rating)) %>%
  filter(date_diff >= 0) %>%
  ggplot(aes(x=date_diff, y=avg_rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Time: From release to rate")

#############################################################
# Methods and Analysis
#############################################################
# Define loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Split edx into test and train sets
set.seed(1, sample.kind="Rounding")
# Test set is 20% of edx
test_index <- createDataPartition(y = edx_new$rating, times = 1, p = 0.2, list = FALSE)
train <- edx_new[-test_index,]
test <- edx_new[test_index,]
test <- test %>% 
      semi_join(train, by = "movieId") %>%
      semi_join(train, by = "userId")
rm(test_index)

### Simple Model ###
# Calculate average rating for all users across all movies
mu <- mean(train$rating)
mu

# Compute RMSE
simple_rmse <- RMSE(test$rating, mu)
simple_rmse

### Movie Model ###
# Calculate movie effect on average rating and plot the bias
# Movie bias is the only "bias" plot we will show in order to save space, but it can be done for others
movie_bias <- train %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))
movie_bias %>% ggplot(aes(x=b_i)) +
  geom_histogram(bins = 15, color = I("black")) +
  xlab("Movie Bias") + ylab("Count") +
  ggtitle("Movie Effect")

# Compute predictions on test set and report RMSE
predicted_ratings <- mu + test %>%
  left_join(movie_bias, by='movieId') %>%
  pull(b_i)
movie_rmse <- RMSE(test$rating, predicted_ratings)
movie_rmse

### Movie + User Model ###
# Calculate user effect
user_bias <- train %>% 
  left_join(movie_bias, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Compute predictions on test set and report RMSE
predicted_ratings <- test %>%
  left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
user_rmse <- RMSE(test$rating, predicted_ratings)
user_rmse

### Movie + User + Time Model ###
# Calculate time effect 
time_bias <- train %>% 
  left_join(movie_bias, by = 'movieId') %>%
  left_join(user_bias, by = "userId") %>%
  group_by(year_release) %>%
  summarize(b_y = mean(rating - b_i - b_u - mu))

# Compute predictions on test set and report RMSE
predicted_ratings <- test %>%
  left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  left_join(time_bias, by='year_release') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)
time_rmse <- RMSE(test$rating, predicted_ratings)
time_rmse

### Movie + User + Time + Genre Model ###
# Calculate genre effect
genre_bias <- train %>% 
  left_join(movie_bias, by = 'movieId') %>%
  left_join(user_bias, by = "userId") %>%
  left_join(time_bias, by = "year_release") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - b_i - b_u - b_y - mu))

# Compute predictions on test set and report RMSE
predicted_ratings <- test %>%
  left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  left_join(time_bias, by='year_release') %>%
  left_join(genre_bias, by= "genres") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
  pull(pred)
genre_rmse <- RMSE(test$rating, predicted_ratings)
genre_rmse

### Regularized Model ###
# Examine top 10 and bottom 10 movies according to our estimates
movie_titles <- train %>% 
  select(movieId, title) %>%
  distinct()
movie_bias %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%
  knitr::kable()
movie_bias %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%
  knitr::kable()

# Examine how often these top 10 and bottom 10 movies are rated
train %>% count(movieId) %>% 
  left_join(movie_bias) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>%
  knitr::kable()
train %>% count(movieId) %>% 
  left_join(movie_bias) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>%
  knitr::kable()

# Regularization: find optimal lambda using cross-validation
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_i <- train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_y <- train %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(year_release) %>%
    summarize(b_y = sum(rating - b_i - b_u - mu)/(n()+l)) 
  b_g <- train %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year_release") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - b_y - mu)/(n()+l)) 
  predicted_ratings <- 
    test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year_release") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_y +b_g) %>%
    .$pred
  return(RMSE(test$rating, predicted_ratings))
})
lambda <- lambdas[which.min(rmses)]
lambda

# Get RMSE for that optimal lambda in our regularized model
all_reg_rmse <- min(rmses)
all_reg_rmse

### Matrix Factorization ###
# Use recosystem package
# Reference Manual:
  # https://cran.r-project.org/web/packages/recosystem/recosystem.pdf
# Vignette:
  # https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html
set.seed(123, sample.kind = "Rounding")
# Convert train and test sets into recosystem input format
train_data <- with(train, data_memory(user_index = userId,
                                          item_index = movieId,
                                          rating = rating))
test_data <- with(test, data_memory(user_index = userId,
                                        item_index = movieId,
                                        rating = rating))
# Create recosystem object
r <- Reco()
# Define tuning parameters
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30),
                                       lrate = c(0.01, 0.1),
                                       costp_l1 = 0,
                                       costq_l1 = 0,
                                       costp_l2 = c(0.01, 0.1),
                                       costq_l2 = c(0.01, 0.1),
                                       nthread = 4, 
                                       niter = 10))
# Train the algorithm
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))
# Calculate predictions on test set
predictions_reco <- r$predict(test_data, out_memory())
# Compute RMSE
matrix_rmse <- RMSE(test$rating, predictions_reco)
matrix_rmse

#############################################################
# Results
#############################################################
# Create table for RMSE results of all our models
models <- c("Avg Rating",
            "Avg Rating + Movie",
            "Avg Rating + Movie + User",
            "Avg Rating + Movie + User + Year",
            "Avg Rating + Movie + User + Year + Genre",
            "All Effects Regularized",
            "Matrix Factorization")
results <- c(simple_rmse, movie_rmse, user_rmse, time_rmse, genre_rmse, all_reg_rmse, matrix_rmse)
results_table <- data.frame(Model = models, RMSE = results)
knitr::kable(results_table)

### Final Validation ###
# apply matrix factorization model using edx and validation sets
set.seed(123, sample.kind = "Rounding")
# Convert edx and validation sets to recosystem input format
edx_data <- with(edx_new, data_memory(user_index = userId,
                                          item_index = movieId,
                                          rating = rating))
validation_data <- with(validation, data_memory(user_index = userId,
                                        item_index = movieId,
                                        rating = rating))
# Create recosystem object
r <- Reco()
# Define tuning parameters
opts <- r$tune(edx_data, opts = list(dim = c(10, 20, 30),
                                       lrate = c(0.01, 0.1),
                                       costp_l1 = 0,
                                       costq_l1 = 0,
                                       costp_l2 = c(0.01, 0.1),
                                       costq_l2 = c(0.01, 0.1),
                                       nthread = 4, 
                                       niter = 10))
# Train the algorithm
r$train(edx_data, opts = c(opts$min, nthread = 4, niter = 20))
# Calculate predictions on validation set
predictions_final_reco <- r$predict(validation_data, out_memory())
# Compute RMSE with validation set
matrix_final_rmse <- RMSE(validation$rating, predictions_final_reco)
matrix_final_rmse
