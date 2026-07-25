# Define an R function to Splatter-based simulate data
library(splatter)
library(Matrix)

# Renamed the function to reflect we are returning data, not writing a file
simulate_data <- function(nGroups=10,
                          nGenes=3000,
                          batchCells=1000,
                          dropout=0,
                          de.scale=0.4,
                          seed=100,
                          group.prob=NULL) {
  
  if (nGroups > 1) method <- 'groups' else method <- 'single'
  
  if (is.null(group.prob)) {
    group.prob <- rep(1, nGroups) / nGroups
  }
  
  sim <- splatSimulate(group.prob=group.prob,
                       nGenes=nGenes,
                       batchCells=batchCells,
                       dropout.type="experiment",
                       method=method,
                       dropout.shape=-1, dropout.mid=dropout,
                       de.facScale=de.scale, seed=seed)
  
  # Extract and transpose
  X <- as.matrix(t(counts(sim)))
  true_X <- as.matrix(t(assays(sim)$TrueCounts))
  Y <- as.integer(substring(colData(sim)$Group, 6)) - 1
  var_names <- as.character(rownames(sim))
  obs_names <- as.character(colnames(sim))
  
  # Return all components as a named R list
  return(list(X=X, true_X=true_X, Y=Y, var_names=var_names, obs_names=obs_names))
}