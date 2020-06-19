########################################
## PH125.9x Data Science: Capstone
## Project: MovieLens
##
## Brendan Hawk
## https://github.com/mrbrendanjs/PH125.9x-movielens
########################################

########################################
## Required Packages
########################################
if(!require(data.table))
  install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr))
  install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(dtplyr))
  install.packages("dtplyr", repos = "http://cran.us.r-project.org")
if(!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(foreach))
  install.packages("foreach", repos = "http://cran.us.r-project.org")
if(!require(doSNOW))
  install.packages("doSNOW", repos = "http://cran.us.r-project.org")
if(!require(recosystem))
  install.packages("recosystem", repos = "http://cran.us.r-project.org")

########################################
## If dataset has not already been acquired, run the provided
## script to download and split dataset.
##
## That script is a direct copy of the code provided by the courseware
## to acquire and split the dataset. I have included it as a separate
## file and sourced it here to keep my work separate from the provided code.
########################################
trainingDatasetFilePath <- file.path(getwd(), "data", "training.csv")
validationDatasetFilePath <- file.path(getwd(), "data", "validation.csv")
if (
  !file.exists(trainingDatasetFilePath)
  || !file.exists(validationDatasetFilePath)
) {
  # Run the provided script
  # this creates two variables in the current window:
  #   edx
  #   validation
  source(file.path(getwd(), "get_data.R"))

  # save the generated datasets to file for future usage of this script
  dataDirectoryPath <- file.path(getwd(), "data")
  if (!dir.exists(dataDirectoryPath)) {
    dir.create(dataDirectoryPath)
  }

  # write training dataset
  write.table(
    edx,
    trainingDatasetFilePath,
    sep = ",",
    row.names = FALSE,
    na = "",
    qmethod = "double"
  )

  # write validation dataset - NB this will not be used until
  # all analysis is complete, to verify accuracy
  write.table(
    validation,
    validationDatasetFilePath,
    sep = ",",
    row.names = FALSE,
    na = "",
    qmethod = "double"
  )

  # clean up memory
  remove(validation, dataDirectoryPath)
} else {
  # else we will read data from existing files
  edx <- fread(trainingDatasetFilePath)
}

# clean up unneeded variables
remove(trainingDatasetFilePath, validationDatasetFilePath)

# Set the seed for all randomness used in this script, so that
# results are reproducible
set.seed(2020, sample.kind = "Rounding")

# Split a chunk of data off to use as test data to validate
# training as we go.
testing.index <- createDataPartition(
  edx$rating,
  times = 1,
  p = 0.1,
  list = FALSE
)
training <- edx[-testing.index,]
testing.temp <- edx[testing.index,]

# Make sure userId and movieId in testing set are also in training set
# using as_tibble here and below because some joins are
# not compatible with dtplyr backend and data.tables,
# so applying them to lazy_dt's fails
testing <- testing.temp %>%
  as_tibble() %>%
  semi_join(training, by = "movieId") %>%
  semi_join(training, by = "userId")

# Add rows removed from testing set back into training set
removed <- anti_join(
  as_tibble(testing.temp),
  testing,
  by = names(testing)
)
training <- rbind(training, removed)

# Clean up memory
remove(edx, testing.index, testing.temp, removed)

# We will be relying heavily on data.table being much
# more performant and easier to manipulate than data.frame
training <- as.data.table(training)
testing <- as.data.table(testing)

########################################
## Create a generic RMSE function to measure loss.
## This loss function will be used to measure the
## accuracy of any algorithms used in Analysis of this data.
########################################
loss <- function(observations, predictions) {
  sqrt(mean((observations - predictions)^2))
}

########################################
## In all uses of parallel computing in this script, we will use
## this number to configure how many threads to spin up. This is
## HEAVILY dependent on the hardware you're running this script on,
## so be VERY CAREFUL to select an appropriate number for your machine.
## You should never try to use more threads than your processor
## has to offer.
########################################
nThreads <- 16

########################################
## Data analysis - Naive Baseline (rating = mu)
########################################
mu <- mean(training$rating)

loss(mu, training$rating)
# 1.060272

loss(mu, testing$rating)
# 1.060272

########################################
## Data analysis - Preparing data to model Movie and User Effects
########################################
# first, sweep Mu out of data
training[, swept_rating := rating - mu]

# movie-mu, for modeling movie effect
mMu <- training[
  ,
  list(mMu = mean(swept_rating)),
  by = movieId
]

# sweep mMu out of data
training <- merge(training, mMu, by = "movieId") %>%
  lazy_dt() %>%
  mutate(swept_rating = swept_rating - mMu) %>%
  as.data.table()

# user-mu, for modeling user effect
uMu <- training[
  ,
  list(uMu = mean(swept_rating)),
  by = userId
]

# sweep uMu out of data
training <- merge(training, uMu, by = "userId") %>%
  lazy_dt() %>%
  mutate(swept_rating = swept_rating - uMu) %>%
  as.data.table()

# Make a prediction that Rating = Mu + mMu + uMu
loss(
  training[, .(prediction = uMu + mMu + mu)]$prediction,
  training$rating
)
# This looks promising! However, let's check loss againts testing dataset
# add columns for the calculated mu's
testing$mu <- mu

testing <- merge(
  testing,
  mMu,
  by = "movieId"
)

testing <- merge(
  testing,
  uMu,
  by = "userId"
)

# calculate loss
loss(
  testing[, .(prediction = uMu + mMu + mu)]$prediction,
  testing$rating
)
# 0.8657138

# there is noticeably more loss with the testing dataset
# and every likelyhood this would be the case with
# the validation set as well.

# That said, these swept ratings, Mu, and movie/user effects will become
# the baseline for the ML Algorithms we apply - ie we will be predicting
# the "swept" rating value, and re-adding the Mu's when checking accuracy
training <- training[, list(movieId, userId, rating = swept_rating)]
testing <- testing[, list(movieId, userId, rating = rating - mu - mMu - uMu)]

gc()
########################################
## Data analysis - recosystem::recommender
##   It seems only prudent to use a purpose-built algorithm
##   for this task. Recosystem uses a parallelized version of
##   Matrix Factorization, allowing it to apply this costly
##   approach to large datasets more efficiently
########################################

# Create the recommender object
recommender <- Reco()

# Create an in-memory data object with the training
# dataset. Recosystem can also use data residing in files.
trainingObject <- data_memory(
  user_index = training$userId,
  item_index = training$movieId,
  rating = training$rating,
  index1 = TRUE
)

# Set the number of threads at the top of this file!
# NB: With my own resources, I was able to run the following code on a
# relatively powerful consumer processor, using 16 threads running at ~4Ghz.
# Even with this setup, the following tuning step took 34 real hours.
timer <- proc.time()
recoTuningResults <- recommender$tune(
  trainingObject,
  opts = list(
    dim = seq(10, 30, 10),
    costp_l1 = 0,
    costp_l2 = seq(0, 1, 0.25),
    costq_l1 = 0,
    costq_l2 = seq(0, 1, 0.1),
    lrate = 0.1,
    niter = 100,
    nfold = 5,
    nbin = 25,
    nthread = nThreads
  )
)
message("Tuning process timing: ", (proc.time() - timer)[3], " seconds")
remove(timer)

# The best-tuned opts can be shown here
recoTuningResults$min

# Train the algorithm with the best-case tuning parameters
recommender$train(trainingObject, opts = recoTuningResults$min)

# clear up some memory
remove(recoTuningResults)

#############
# Check this model's loss against the testing dataset.
#############
# Create the testing dataset object in memory
testingObject <- data_memory(
  user_index = testing$userId,
  item_index = testing$movieId,
  rating = testing$rating,
  index1 = TRUE
)

# Generate our predictions
recosystem.predictions = recommender$predict(testingObject, out_memory())

loss(
  recosystem.predictions,
  testing$rating
)
# 0.8218481

# cleanup
remove(recosystem.predictions, testingObject, trainingObject)

########################################
## Data analysis - Slope One Algorithm
##   While the above algorithm is already accurate enough, I wanted to
##   explore alternatives as it was SUCH a costly and complicated process.
##
##   Implementing Slope One on a dataset this size
##   will take much less time than Recosystem, however
##   it will come at a similarly large cost of memory. In lieu
##   of using RAM, I've implemented Slope One to write temporary
##   data to disk in chunks before finally re-aggregating the data into
##   a final model. In practice, this has cost anywhere from 20GB to 100Gb
##   of space, depending on how many chunks you split the data into. NB that
##   this script will fail if it runs out of space on your drive.
########################################

# First, set up the function that will calculate our Slope One model.
slopeone <- function(ratings) {
  # Iterate over every user's data, calculating the difference in scores
  # between each pair of movies they've rated - Order matters in these
  # permutations, so we will get (movie1 = x, movie2 = y)
  # as well as (movie1 = y, movie2 = x) in the resulting models.
  rating.diffs <- ratings %>%
    as_tibble() %>%
    group_by(userId) %>%
    group_map(function(user.data, data.key) {
      gc()

      # First: generate every combination of movies for this user's data
      row.combinations <- as.data.table(expand.grid(
        row1 = 1:nrow(user.data),
        row2 = 1:nrow(user.data)
      ))

      # Filter out rows where the same movie is used twice
      row.combinations <- row.combinations[row1 != row2]

      # For every combination generated above, calculate a difference
      # between the two movie scores.
      data.table(
        movieId1 = user.data$movieId[row.combinations$row1],
        movieId2 = user.data$movieId[row.combinations$row2],
        diff = user.data$rating[row.combinations$row2] - user.data$rating[row.combinations$row1]
      )
    }) %>%
    rbindlist()

  # If this chunk of user data didn't end up providing us any combinations
  # of movies it means this set of users EACH only rated 1 movie.
  # We cannot leverage that data, so return an empty model.
  if (nrow(rating.diffs) == 0) {
    return(data.table(movieId1 = c(), movieId2 = c(), b = c(), support = c()))
  }

  # Finally, return the resulting model, complete with supporting data
  rating.diffs[
    ,
    list(b = mean(diff), support = .N),
    by = list(movieId1, movieId2)
  ]
}

# Next, set up the function that will calculate Slope One predictions
# for a given user and movie.
predict.slopeone <- function(model, target.movieId, user.ratings) {
  # Alter the input data so that we can merge with our model data
  setnames(user.ratings, 'movieId', 'movieId1')

  # Set the target movie id as movieId2
  user.ratings$movieId2 = target.movieId

  # set the data.table key for the merge
  setkey(user.ratings, movieId1, movieId2)

  # Join the model and user data
  joined <- model[user.ratings, ]

  # Filter any rows that are incomplete (any relevant column is NA)
  joined <- joined[complete.cases(joined), ]

  # If there are no cases where movieId2 = target & movieId1 = some movie this
  # target user has rated, then we cannot predict a value here. This is a limitation
  # of the Slope one algorithm.
  if (NROW(joined) == 0) {
    return(NA)
  }

  # Otherwise, compute and return a weighted average of the sums of differences
  # in the joined data.
  return(sum(joined[, (b + rating) * support]) / sum(joined[, sum(support)]))
}

# In order to run this alogirthm in parallel, we must "chunk" the data
# to be passed into doParallel::foreach. In an effort to do this evenly, we
# will be using a rough 'binning" approach to generate N bins of roughly the
# same size.
nBins <- 100

# We must keep all of a given user's data together, so we will be binning
# by the aggregate number of reviews a user has given.
binData <- training[, .N, by = userId]

# Cumulatively sum these counts, breaking roughly when the sum approaches
# (total_number_of_reviews/nBins)
binData$bin <- cumsum(binData$N) %/% (ceiling(sum(binData$N) / nBins)) + 1

# Merge these bin numbers in as a column
setkey(binData, userId)
training <- binData[training]

# remove an unnecessary column for memory's sake
training[, N := NULL]

# cleanup
remove(binData)

# Our processing will dump the output of each bin to its own temporary file.
# First lets ensure that directory exists.
if (!dir.exists("./temp")) {
  dir.create(("./temp"))
}

# And clear it out from any previous run of this script
file.remove(list.files("./temp", full.names = TRUE))

# Set up a progress bar so that this long-running process can give
# the user some feedback as to its status.
progressBar <- txtProgressBar(0, nBins, style = 3)

# Setting up this function here allows us to have "progressBar" be
# accessed in the correct scope from the parallel handler.
updateProgress <- function(status) {
  setTxtProgressBar(progressBar, status)
}

# Garbage Collection - an attempt to make this process as painless for
# the computer's memory management as possible.
gc()

# Set the number of threads at the top of this file!
# Create nThreads SOCK clusters to run in parallel.
cl <- makeSOCKcluster(nThreads)
registerDoSNOW(cl)

# A noop (no-operation) function is a programming tool used when a higher-order
# function needs a callback, but we don't want that callback to do anything or
# waste any resources. This will be set as the "combine" handler for doParallel::foreach
noop <- function(...) {}

# The heart of our parallelization of Slope One.
# This will spin up a new thread for each bin, sending the
# appropriate data to the slopeone method, and writing the results
# directly to a csv file in the /temp/ directory we created. Upon
# completion, each thread will call the noop function (doing nothing).
# Taking advantage of .multicombine = TRUE here means that if more than one
# thread finishes at the same time, they are all passed to the same call of
# the noop function, further saving resources. the SNOW progress option
# given here will update the progress bar for each thread that finishes.
timer <- proc.time()
foreach(
  binIndex = 1:nBins,
  .packages = c("data.table", "dplyr", "dtplyr"),
  .combine = noop,
  .multicombine = TRUE,
  .options.snow = list(progress = updateProgress)
) %dopar% {
  # create the file path for this bin's data.
  filePath <- sprintf("./temp/model_%d.csv", binIndex)

  # calculate and write this data to the file
  write.table(
    slopeone(training[bin == binIndex, .(userId, movieId, rating)]),
    filePath,
    sep = ",",
    row.names = FALSE,
    na = ""
  )
}

# Stop the SOCK threads and close the stream to the progress bar
stopCluster(cl)
close(progressBar)

message("processing slopeone: ", (proc.time() - timer)[3], " seconds")
remove(timer)

# cleanup and garbage collection.
remove(progressBar, updateProgress, cl, noop, slopeone)
gc()

# Now we have to aggregate all the model data into one final model.
# This is done by reading the data into our model, and then "re-averaging"
# the values of b. Since an average is dependent on it's divisor
# (in this case, the support value) we can mathematically aggregate these means by
# multiplying them by their support, and taking a new mean with the sum of
# all the supports for that average.

# This process is very RAM heavy and due to the size of the vecotrs involved it takes
# even longer than calculating the models in the first place.
# In practice this took about an hour.

# This function will do the math of re-averaging for us as the files are read.
# It will be used in the %dopar% process below as the .combine function.
reaggregate.slopeone <- function(...) {
  gc()

  rbindlist(list(...))[
    ,
    .(b = sum(b * support)/sum(support), support = sum(support)),
    by = .(movieId1, movieId2)
  ]
}

# Begin a progress bar, with a function to keep it updated.
progressBar <- txtProgressBar(0, nBins, style = 3)
updateProgress <- function(status) {
  setTxtProgressBar(progressBar, status)
}

gc()

# This value is used to limit how many threads try to combine using the
# reaggregate.slopeone function at the same time. The more that gets passed
# to that function, the more RAM is needed to do the calculation, so this should
# be tested and limited to prevent the script from halting due to memory management.
maxcombine <- 6
cl <- makeSOCKcluster(maxcombine)
registerDoSNOW(cl)

timer <- proc.time()
slopeone.model <- foreach(
  binIndex = 1:nBins,
  .packages = c("data.table", "dplyr", "dtplyr"),
  .combine = reaggregate.slopeone,
  .multicombine = TRUE,
  .maxcombine = maxcombine,
  .options.snow = list(progress = updateProgress)
) %dopar% {
  fread(sprintf("./temp/model_%d.csv", binIndex))
}

close(progressBar)
message("Aggregate slopeone model: ", (proc.time() - timer)[3], " seconds")

remove(binIndex, timer, progressBar, updateProgress)
gc()

# for sanity, in case this file ever halts for some reason, we can
# cache this to file for safe-keeping
write.table(
  slopeone.model,
  './temp/finalModel.csv',
  row.names = FALSE,
  na = "",
  sep = ","
)

# Now that we have our final model, we can remove the rest of the temp data
file.remove(list.files("./temp", pattern='model_[0-9]+.csv', full.names = TRUE))

############
## Generate the predictions on the testing dataset
############
# Set keys, so that joins are considerably faster.
setkey(training, userId)
setkey(slopeone.model, movieId1, movieId2)

# Set up a progress bar for user feedback
progressBar <- txtProgressBar(0, nrow(testing), style = 3)
updateProgress <- function(status) {
  setTxtProgressBar(progressBar, status)
}

# create our parallel threads.
cl <- makeSOCKcluster(nThreads)
registerDoSNOW(cl)

# This prediction step also takes a while, even in parallel. This
# is because the calculation has to be run separately for every combination
# of movie/user, which means we are iterating over every row in
# the testing dataset.
timer <- proc.time()
predictions <- foreach(
  rowIndex = 1:nrow(testing),
  .packages = c("data.table", "dplyr", "dtplyr"),
  .combine = rbind,
  .multicombine = TRUE,
  .options.snow = list(progress = updateProgress)
) %dopar% {
  data.table(
    testing[rowIndex, ],
    prediction = predict.slopeone(
      slopeone.model,
      testing$movieId[rowIndex],
      training[J(testing$userId[rowIndex]), ]
    )
  )
}
# Stop the threads, cleanup, garbage collect
stopCluster(cl)
close(progressBar)

message("Predict slopeone testing: ", (proc.time() - timer)[3], " seconds")
remove(timer)

remove(cl, progressBar, updateProgress, rowIndex)
gc()

# "unsweep" the averages and movie/user effects into the predicted values
# we will need these Mu's to deal with NAs below.
setkey(predictions, movieId)
setkey(mMu, movieId)
predictions <- mMu[predictions]

setkey(predictions, userId)
setkey(uMu, userId)
predictions <- uMu[predictions]

# It is entirely possible that the Slope One model has generated NAs as
# predictions. As explained in the model function, this comes from an instance
# where there is no combination of data to line up for a given user/movie and
# therefor no inference can be performed. In this case, we will have to fall
# back on a different method. Here I have chosen to use the Naive model + Movie
# and User effects to keep this simple.
predictions[
  ,
  predicted_rating := ifelse(is.na(prediction), 0, prediction) + mu + mMu + uMu
][
  ,
  unswept_rating := rating + mu + mMu + uMu
]

loss(predictions$predicted_rating, predictions$unswept_rating)
# 0.8571771


# Final cleanup before Validation
remove(predictions, testing)

########################################
## Accuracy measurement with Validation data
########################################

# Read the validation dataset from file
validation <- fread("./data/validation.csv")

# Only these three columns are needed for the algorithms I'm applying
validation <- validation[
  ,
  .(userId, movieId, rating)
]

# Set keys, join movie and user Mu's
setkey(validation, movieId)
setkey(mMu, movieId)
validation <- mMu[validation]

setkey(validation, userId)
setkey(uMu, userId)
validation <- uMu[validation]

# Sweep Mu and Movie/User effects from ratings
validation[
  ,
  rating := rating - mu - mMu - uMu
]

# create the Recosystem data object for the validation dataset
validationObject <- data_memory(
  user_index = validation$userId,
  item_index = validation$movieId,
  rating = validation$rating,
  index1 = TRUE
)

# Create the predictions as a new column on this data.table
validation <- cbind(
  validation,
  prediction = recommender$predict(validationObject, out_memory())
)

# Take a final measurement of loss
loss(
  validation[, list(prediction = prediction + mu + mMu + uMu)]$prediction,
  validation[, list(rating = rating + mu + mMu + uMu)]$rating
)
# 0.8221265!

# For the purposes of this course and this project, this is WELL beyond the
# measurement of "good enough". However, my interest in the Slope One algorithm
# remains, as it is so much less complicated it amazes me that it can be as
# accurate as it is. As such, I also wanted to take a final measurement of
# the accuracy of Slope One against the validation dataset

# Drop the recosystem predictions
validation[, prediction := NULL ]

# Set keys on the data.table
setkey(training, userId)
setkey(slopeone.model, movieId1, movieId2)

# Create user feedback progress bar
progressBar <- txtProgressBar(0, nrow(validation), style = 3)
updateProgress <- function(status) {
  setTxtProgressBar(progressBar, status)
}

# Begin parallel threads
cl <- makeSOCKcluster(nThreads)
registerDoSNOW(cl)

timer <- proc.time()
# Make the predictions
predictions <- foreach(
  rowIndex = 1:nrow(validation),
  .packages = c("data.table", "dplyr", "dtplyr"),
  .combine = rbind,
  .multicombine = TRUE,
  .options.snow = list(progress = updateProgress)
) %dopar% {
  data.table(
    validation[rowIndex, ],
    prediction = predict.slopeone(
      slopeone.model,
      validation$movieId[rowIndex],
      training[J(validation$userId[rowIndex]), .(userId, movieId, rating)]
    )
  )
}
# Stop clusters, clean up, garbage collect
stopCluster(cl)
close(progressBar)

message("Predict slopeone validation: ", (proc.time() - timer)[3], " seconds")
remove(timer)

remove(cl, progressBar, updateProgress)

# A final measurement of loss for the Slope One model
loss(
  predictions[, list(prediction = prediction + mu + mMu + uMu)]$prediction,
  predictions[, list(rating = rating + mu + mMu + uMu)]$rating
)
# 0.8568992
