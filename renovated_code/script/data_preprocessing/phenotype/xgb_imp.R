load_xgboost_imputation <- function() {
  suppressPackageStartupMessages({
    library(xgboost)
    library(doParallel)
    library(doRNG)
    library(itertools)
  })

  numCores <- parallel::detectCores()
  registerDoParallel(max(numCores - 1, 1))

  xgboost_imputation <- function(data, maxiter = 10, verbose = TRUE, max.depth = 2,
                                 nrounds = 50, decreasing = FALSE, parallelize = "variables") {
    xmis <- data
    n <- nrow(xmis)
    p <- ncol(xmis)

    if (any(apply(is.na(xmis), 2, sum) == n)) {
      indCmis <- which(apply(is.na(xmis), 2, sum) == n)
      xmis <- xmis[, -indCmis]
      p <- ncol(xmis)
      cat("removed variable(s)", indCmis, "due to the missingness of all entries\n")
    }

    ximp <- xmis
    varType <- character(p)
    for (t.co in seq_len(p)) {
      ximp[is.na(xmis[, t.co]), t.co] <- colMeans(xmis, na.rm = TRUE)[t.co]
      varType[t.co] <- "numeric"
    }

    NAloc <- is.na(xmis)
    noNAvar <- apply(NAloc, 2, sum)
    sort.j <- order(noNAvar)
    if (decreasing) {
      sort.j <- rev(sort.j)
    }
    sort.noNAvar <- noNAvar[sort.j]
    nzsort.j <- sort.j[sort.noNAvar > 0]

    if (parallelize == "variables") {
      "%cols%" <- get("%dorng%")
      idxList <- as.list(isplitVector(nzsort.j, chunkSize = getDoParWorkers()))
    } else {
      stop("Only parallelize='variables' is supported")
    }

    iter <- 0
    k <- length(unique(varType))
    convNew <- rep(0, k)
    names(convNew) <- "numeric"
    convOld <- rep(Inf, k)
    Ximp <- vector("list", maxiter)

    stopCriterion <- function(varType, convNew, convOld, iter, maxiter) {
      k <- length(unique(varType))
      if (k == 1) {
        (convNew < convOld) & (iter < maxiter)
      } else {
        ((convNew[1] < convOld[1]) | (convNew[2] < convOld[2])) & (iter < maxiter)
      }
    }

    while (stopCriterion(varType, convNew, convOld, iter, maxiter)) {
      if (iter != 0) {
        convOld <- convNew
      }
      if (verbose) {
        cat("  XGBoost iteration", iter + 1, "in progress...")
      }
      ximp.old <- ximp

      for (idx in idxList) {
        results <- foreach(varInd = idx, .packages = "xgboost") %cols% {
          obsi <- !NAloc[, varInd]
          misi <- NAloc[, varInd]
          obsY <- ximp[obsi, varInd, drop = FALSE]
          misY <- ximp[misi, varInd, drop = FALSE]
          obsX <- ximp[obsi, seq_len(p)[-varInd], drop = FALSE]
          misX <- ximp[misi, seq_len(p)[-varInd], drop = FALSE]

          xgb_obsX <- xgb.DMatrix(data = as.matrix(obsX), label = as.matrix(obsY))
          xgb_misX <- xgb.DMatrix(data = as.matrix(misX), label = as.matrix(misY))
          xgb <- xgboost(data = xgb_obsX, max.depth = max.depth, nrounds = nrounds, verbose = 0)
          misY <- predict(xgb, xgb_misX)
          list(varInd = varInd, misY = misY)
        }

        for (res in results) {
          misi <- NAloc[, res$varInd]
          ximp[misi, res$varInd] <- res$misY
        }
      }

      if (verbose) {
        cat("done!\n")
      }

      iter <- iter + 1
      Ximp[[iter]] <- ximp

      t.co2 <- 1
      for (t.type in names(convNew)) {
        t.ind <- which(varType == t.type)
        convNew[t.co2] <- sum((ximp[, t.ind] - ximp.old[, t.ind])^2) / sum(ximp[, t.ind]^2)
        t.co2 <- t.co2 + 1
      }
    }

    if (iter == maxiter) Ximp[[iter]] else Ximp[[iter - 1]]
  }

  xgboost_imputation
}
