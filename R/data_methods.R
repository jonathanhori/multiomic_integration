
# colMeans(X_l[[1]])
# colVars(X_l[[1]], std = TRUE)


standardize_X <- function(data, training_data = NA) {
  if (is.na(training_data)) {
    col_means <- colMeans(data)
    col_sds <- colVars(data, std = TRUE)
  }
  
  scale(data, center = col_means, scale = col_sds)
}

standardize_views <- function(view_list, 
                              training_list = NA) {
  if (is.na(training_list)){
    lapply(1:length(view_list),
           function(l) standardize_X(view_list[[l]]))
  } else {
    print(training_list)
    print("training list passed, function not written")
    return(NA)
  }
}

standardize_outcome <- function(data, training_data = NA) {
  if (is.na(training_data)) {
    scale(data)
    # mean <- mean(y)
    # col_means <- colMeans(data)
    # col_sds <- colVars(data, std = TRUE)
  }
  
  # scale(data, center = col_means, scale = col_sds)
}

# mean(y)
# 
# scale(y)
