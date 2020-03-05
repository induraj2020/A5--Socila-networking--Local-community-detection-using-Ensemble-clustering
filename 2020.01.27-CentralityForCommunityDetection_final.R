# EISTI - ADEO2 - 2019/2020
# Social Network Analysis
# Professor: Kanawati
# Students: 
#   Gustavo Fleury Soares
#   Induraj P.
#   Quoc Viet Pham

#-----------
# Libraries and Install
#-----------
#install.packages("caret",depenencies=c("Depends","suggests"))
#install.packages("CINNA")

library(igraph)
library(CINNA)
library(randomForest)
library(caret)
require(neuralnet)
require(nnet)

#Codes for calculate Local Modularity "LocalModularity.R"
source("H:\\Google Drive\\ADEO2-Projects\\Social Networks\\LocalModularity.R")

#-----------
# READ DATASET
#-----------
# Select what graph will work with:

g <- read.graph("karate.gml",format="gml")   #ok
#g <- read.graph("dolphins.gml",format="gml") #Need update $id #OK
#g <- read.graph("football.gml",format="gml") #ok
#g <- read.graph("polbooks.gml",format="gml") #Need update $id and $value

#update $value
V(g)$value
for (v in V(g)){
  if ( V(g)$value[v] == "n" ) {V(g)$value[v] = 0} 
  if ( V(g)$value[v] == "c" ) {V(g)$value[v] = 1} 
  if ( V(g)$value[v] == "l" ) {V(g)$value[v] = 2} 
}
V(g)$value

#update the ID of the Vertices - To not start from 0.
V(g)$id
for (v in V(g)){
  V(g)$id[v] = v
}
V(g)$id

plot(g)

#-----------
# BASIC STATs
#-----------

basicTopFeaturesGraph(g)

#-----------
# LOCAL COMMUNITY - MODULRATIES
#-----------


loc_com<- function(g)
{
  nodes <- V(g)$id
  dfi <- data.frame(nodes)
  
  lmodR <- c()
  lmodM <- c()
  lmodL <- c()
  lbest_Mod <- c()
  for (v in V(g)){
    rmodR <- localcomquality(v,g,mod_R,"nmi")
    rmodM <- localcomquality(v,g,mod_M,"nmi")
    rmodL <- localcomquality(v,g,mod_L,"nmi")
    lmodR <- c(c(lmodR), c( rmodR ) ) # 1
    lmodM <- c(c(lmodM), c( rmodM ) ) # 2 
    lmodL <- c(c(lmodL), c( rmodL ) ) # 3
    if ( rmodR >= rmodM & rmodR >= rmodL  )  { best_Mod <- 1 }
    if ( rmodM >= rmodR & rmodM >= rmodL  )  { best_Mod <- 2 }
    if ( rmodL >= rmodR & rmodL >= rmodM  )  { best_Mod <- 3 }
    lbest_Mod <- c(c(lbest_Mod), c( best_Mod ) )
  }
  dfi$mod_R <- lmodR
  dfi$mod_M <- lmodM
  dfi$mod_L <- lmodL
  dfi$best_mod <- lbest_Mod
  return(dfi)
}

df<-loc_com(g)
View(df)

#-----------
# CENTRALITIES
#-----------
#Source: https://www.datacamp.com/community/tutorials/centrality-network-analysis-R

  #What Centralities Use?

# Compare:
pr.cent<-proper_centralities(g)
pr.cent
calculate_centralities(g, include = pr.cent[1:49])%>%
  pca_centralities(scale.unit = TRUE)

# DF of Centralities:
dfC <- data.frame(lbest_Mod)

# We will use:
dfC$Stress <- calculate_centralities(g, include = "Stress Centrality" )$`Stress Centrality`
dfC$Group  <- calculate_centralities(g, include = "Group Centrality" )$`Group Centrality`
dfC$Eccentricity <- calculate_centralities(g, include = "Eccentricity Centrality" )$`Eccentricity Centrality`
dfC$Harary <- calculate_centralities(g, include = "Harary Centrality" )$`Harary Centrality`
dfC$Geodesic <- calculate_centralities(g, include = "Geodesic K-Path Centrality" )$`Geodesic K-Path Centrality`
dfC$ShortestPath <- calculate_centralities(g, include = "Shortest-Paths Betweenness Centrality" )$`Shortest-Paths Betweenness Centrality`
dfC$Entropy <- calculate_centralities(g, include = "Entropy Centrality" )$`Entropy Centrality`
dfC$FlowBetweenness <- calculate_centralities(g, include = "Flow Betweenness Centrality" )$`Flow Betweenness Centrality`
dfC$MNC <- calculate_centralities(g, include = "MNC - Maximum Neighborhood Component" )$`MNC - Maximum Neighborhood Component`
dfC$Degree <- calculate_centralities(g, include = "Degree Centrality" )$`Degree Centrality`
dfC$Laplacian <- calculate_centralities(g, include = "Laplacian Centrality" )$`Laplacian Centrality`
dfC$Closeness <- calculate_centralities(g, include = "Closeness Centrality (Freeman)" )$`Closeness Centrality (Freeman)`

View(dfC)

#-----------
# TRAIN a MODEL:
#-----------

level_con<- function(df){
  df$lbest_Mod<- as.factor(df$lbest_Mod)
  if (length(levels(df$lbest_Mod))==2){
    levels(df$lbest_Mod)<-c("mod_m","mod_l")
  }
  else
  {
    levels(df$lbest_Mod)<-c("mod_r","mod_m","mod_l")
  }
  return(df)
}

dfc_g_new<- level_con(dfC)


#-------------------------------------
# to create training and test partition 
#-------------------------------------
set.seed(123)

partition_func<- function(df){
  return(createDataPartition(df$lbest_Mod,p=0.7,list = F))
}

partitionrule_g = partition_func(dfc_g_new)
trainingset_g = dfc_g_new[partitionrule_g, ]
testingset_g = dfc_g_new[-partitionrule_g, ]

#-------------------------------------------------
# Random Forest with kfold cross validation
#-------------------------------------------------

splitrule <- trainControl(method="cv", number=4, savePredictions = T, classProbs = T)
rf_model_g<- train(lbest_Mod~., data=trainingset_g, trControl = splitrule, method="rf", metric = "Accuracy")
rf_model_g
rf_test_g <- predict(rf_model_g, newdata=testingset_g)
confusionMatrix(data=rf_test_g,testingset_g$lbest_Mod)

#------------- cross fold with repeated cv -----------------------------------
splitrule_1 <- trainControl(method="repeatedcv", number=4, repeats = 3, savePredictions = T, classProbs = T)
rf_model_g_<- train(lbest_Mod~., data=trainingset_g, trControl = splitrule_1, method="rf", metric = "Accuracy")
rf_model_g_
rf_test_g_ <- predict(rf_model_g_, newdata=testingset_g)
confusionMatrix(data=rf_test_g_,testingset_g$lbest_Mod)


#-------------------------------------------------
# ANN - Artificial Neural Network
#-------------------------------------------------
#https://www.r-bloggers.com/multilabel-classification-with-neuralnet-package/
#https://datascienceplus.com/neuralnet-train-and-test-neural-networks-using-r/


normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# for karate graph
dfC_g <- dfC
maxmindf_g <- as.data.frame(lapply(dfC_g[,1:13], normalize))          # normalizing the entire dataframe including the label, In the label 3 becomes "1" and 2 becomes "0"
maxmindf_g [is.na(maxmindf_g)] <- 0

partitionrule_g_new = createDataPartition(dfC_g$lbest_Mod,p=0.7,list = F)
trainingset_g_new = maxmindf_g[partitionrule_g_new, ]
testingset_g_new = maxmindf_g[-partitionrule_g_new, ]
n_trainset_g<- cbind(trainingset_g_new[,-c(1)],class.ind(as.factor(trainingset_g_new$lbest_Mod)))
names(n_trainset_g)<- c(names(dfc_g_new)[-c(1)],"mod_m","mod_l")

nn_model_g<- neuralnet(mod_m+mod_l~.,data=n_trainset_g, hidden=3, act.fct="logistic",linear.output = F, lifesign = "minimal")
plot(nn_model_g)
nn_model_g$result.matrix
prediciton_nn_g <- compute(nn_model_g, trainingset_g_new[, 1:12])

pred_result_g<- prediciton_nn_g$net.result

# Accuracy (training set)
original_values_g <- max.col(n_trainset_g[,13:14])
prediciton_nn_acc <- max.col(pred_result_g)
mean(prediciton_nn_acc == original_values_g)

# for dolphin
dfC_g2 <- dfC
dfC_g2[is.na(dfC_g2)] <- 0    # to remove the nan's
maxmindf_g2 <- as.data.frame(lapply(dfC_g2[,1:13], normalize))          # normalizing the entire dataframe including the label, In the label 3 becomes "1" and 2 becomes "0"
partitionrule_g2_new = createDataPartition(dfC_g2$lbest_Mod,p=0.7,list = F)
trainingset_g2_new = maxmindf_g2[partitionrule_g2_new, ]
testingset_g2_new = maxmindf_g2[-partitionrule_g2_new, ]
n_trainset_g2<- cbind(trainingset_g_new[,-c(1)],class.ind(as.factor(trainingset_g_new$lbest_Mod)))
names(n_trainset_g2)<- c(names(dfc_g_new)[-c(1)],"mod_r","mod_m","mod_l")

nn_model_g2<- neuralnet(mod_r+mod_m+mod_l~.,data=n_trainset_g2, hidden=3, act.fct="logistic",linear.output = F, lifesign = "minimal")
plot(nn_model_g2)
nn_model_g2$result.matrix
prediciton_nn_g2 <- compute(nn_model_g2, trainingset_g2_new[, 1:12])

pred_result_g2<- prediciton_nn_g2$net.result

# Accuracy (training set)
original_values_g2 <- max.col(n_trainset_g2[,13:15])
prediciton_nn_acc <- max.col(pred_result_g2)
mean(prediciton_nn_acc == original_values_g2)


#---------------------------------------------------
# GENERATED ARTIFICIAL NETWORK - testing LRF model 
#---------------------------------------------------

# Read LRF generate with Python Networkx.
g1 <- read.graph("LRF-34v.gml",format="gml")  #Need update $id and $value
#g1 <- read.graph("LRF-100v.gml",format="gml")  #Need update $id and $value
#g1 <- read.graph("LRF2-250v.gml",format="gml")  #Need update $id and $value
plot(g1)


# Calculate Centralities:
Stress <- calculate_centralities(g1, include = "Stress Centrality" )$`Stress Centrality`
dfC_g1 <- data.frame(Stress)  #Create DF

dfC_g1$Group  <- calculate_centralities(g1, include = "Group Centrality" )$`Group Centrality`
dfC_g1$Eccentricity <- calculate_centralities(g1, include = "Eccentricity Centrality" )$`Eccentricity Centrality`
dfC_g1$Harary <- calculate_centralities(g1, include = "Harary Centrality" )$`Harary Centrality`
dfC_g1$Geodesic <- calculate_centralities(g1, include = "Geodesic K-Path Centrality" )$`Geodesic K-Path Centrality`
dfC_g1$ShortestPath <- calculate_centralities(g1, include = "Shortest-Paths Betweenness Centrality" )$`Shortest-Paths Betweenness Centrality`
dfC_g1$Entropy <- calculate_centralities(g1, include = "Entropy Centrality" )$`Entropy Centrality`
dfC_g1$FlowBetweenness <- calculate_centralities(g1, include = "Flow Betweenness Centrality" )$`Flow Betweenness Centrality`
dfC_g1$MNC <- calculate_centralities(g1, include = "MNC - Maximum Neighborhood Component" )$`MNC - Maximum Neighborhood Component`
dfC_g1$Degree <- calculate_centralities(g1, include = "Degree Centrality" )$`Degree Centrality`
dfC_g1$Laplacian <- calculate_centralities(g1, include = "Laplacian Centrality" )$`Laplacian Centrality`
dfC_g1$Closeness <- calculate_centralities(g1, include = "Closeness Centrality (Freeman)" )$`Closeness Centrality (Freeman)`


# For TEsting the LRF with models that we created

## RANDOM FOREST - K_Fold
rf_test_g1 <- predict(rf_model_g, newdata=dfC_g1)    # testing using Kfold rf model that was developed before
rf_test_g1

## RANDOM FOREST - K_Fold Repeated
rf_test_g2 <- predict(rf_model_g_, newdata=dfC_g1)
rf_test_g2

## ANN
ann_test_g <- compute(nn_model_g, dfC_g1)
pred_result_g2<- ann_test_g$net.result
prediciton_nn_acc <- max.col(pred_result_g2)


#Create the V(g1)$value based on $community
for (v in V(g1)){
  if ( V(g1)$community[v] == "0,3,6,7,9,10,11,14,18,23,25,26" ) {V(g1)$value[v] = 0} 
  if ( V(g1)$community[v] == "33,2,4,5,13,16,17,22,28,31" ) {V(g1)$value[v] = 1} 
  if ( V(g1)$community[v] == "32,1,8,12,15,19,20,21,24,27,29,30" ) {V(g1)$value[v] = 2} 
}


#Update INDEX values
V(g1)$id
for (v in V(g1)){
  V(g1)$id[v] = v
}

#Calculate the Modularities for each Node
dfLRF <- loc_com(g1)

dfLRF$lbest_Mod <- dfLRF$best_mod
dfLRF <- level_con(dfLRF)


dfLRF$predict_RFkfold  <- rf_test_g1
dfLRF$predict_RFkfold_True  <- 0
for (i in 1:nrow(dfLRF)){
  if (dfLRF$predict_RFkfold[i] == dfLRF$lbest_Mod[i] ) { dfLRF$predict_RFkfold_True[i] <- 1 } else { dfLRF$predict_RFkfold_True[i] <- 0 }
}

dfLRF$predict_RFkfoldR <- rf_test_g2
dfLRF$predict_RFkfoldR_True <- 0
for (i in 1:nrow(dfLRF)){
  if (dfLRF$predict_RFkfoldR[i] == dfLRF$lbest_Mod[i] ) { dfLRF$predict_RFkfoldR_True[i] <- 1 } else { dfLRF$predict_RFkfoldR_True[i] <- 0 }
}

dfLRF$predict_ANN      <- prediciton_nn_acc
dfLRF$predict_ANN_True      <- 0
for (i in 1:nrow(dfLRF)){
  if (dfLRF$predict_ANN[i] == 1 ) { dfLRF$predict_ANN[i] <- "mod_l" } else { dfLRF$predict_ANN[i] <- "mod_l" }
}
for (i in 1:nrow(dfLRF)){
  if (dfLRF$predict_ANN[i] == dfLRF$lbest_Mod[i] ) { dfLRF$predict_ANN_True[i] <- 1 } else { dfLRF$predict_ANN_True[i] <- 0 }
}

#Verify the Accuracy
View(dfLRF)
sum(dfLRF$predict_RFkfold_True)/nrow(dfLRF)
sum(dfLRF$predict_RFkfoldR_True)/nrow(dfLRF)
sum(dfLRF$predict_ANN_True)/nrow(dfLRF)


#############################################################################################################################

loc_com<- function(g)
{
  nodes <- V(g)$id
  df <- data.frame(nodes)
  lmodR <- c()
  lmodM <- c()
  lmodL <- c()
  lbest_Mod <- c()
  mean_mod<-c()
  for (v in V(g)){
    rmodR <- localcomquality(v,g,mod_R,"nmi")
    rmodM <- localcomquality(v,g,mod_M,"nmi")
    rmodL <- localcomquality(v,g,mod_L,"nmi")
    lmodR <- c(c(lmodR), c( rmodR ) ) # 1
    lmodM <- c(c(lmodM), c( rmodM ) ) # 2 
    lmodL <- c(c(lmodL), c( rmodL ) ) # 3
    if ( rmodR >= rmodM & rmodR >= rmodL  )  { best_Mod <- 1 }
    if ( rmodM >= rmodR & rmodM >= rmodL  )  { best_Mod <- 2 }
    if ( rmodL >= rmodR & rmodL >= rmodM  )  { best_Mod <- 3 }
    lbest_Mod <- c(c(lbest_Mod), c( best_Mod ) )
    mean_mod <- c(c(mean_mod),c((rmodR+rmodM+rmodL)/3))
  }
  df$mod_R <- lmodR
  df$mod_M <- lmodM
  df$mod_L <- lmodL
  df$best_mod <- lbest_Mod
  return(list(df,mean_mod))
}

returned_list  = loc_com(g)
df_g = as.data.frame(returned_list[1])
g_mean=as.vector(unlist(returned_list[2]))

#localcom(2,g1,mod_R)
#localcomquality(2,g1,mod_R,"nmi")   # checking on ARF, but not working

# returned_list_1 = loc_com(g1)
# df_g1 = as.data.frame(returned_list_1[1])
# g_mean1=as.vector(unlist(returned_list_1[2]))

returned_list_2 = loc_com(g2)
df_g2 = as.data.frame(returned_list_2[1])
g_mean2=as.vector(unlist(returned_list_2[2]))


#################################### Ensemble clustering ##############################
#--to check CSPA, generating consensus matrix


consensus_func<- function(g,df){
  cdf <- list()
  for (element in V(g)$id)
  {
    bind_vec =c()
    if(df$best_mod==3){result = localcom(element,g,mod_L)}
    if(df$best_mod==2){result = localcom(element,g,mod_M)}
    if(df$best_mod==1){result = localcom(element,g,mod_R)}
    for(loc_ele in df$nodes){
      if(loc_ele %in% result){
        bind_vec=append(bind_vec,1)
      }
      else{
        bind_vec=append(bind_vec,0)
      }
    }
    cdf[[element]]=bind_vec
  }
  new_cdf <- as.data.frame(do.call("rbind", cdf))
  return(new_cdf)
}

consensus_g<- consensus_func(g,df_g)
consensus_g2 <- consensus_func(g2, df_g2)

#------------------------
#--performing the cspa 
#------------------------

#install.packages("diceR")
library(diceR)
#https://cran.r-project.org/web/packages/diceR/vignettes/overview.html

consensus_cluster_func<- function(consensus_matri,clu){
  x <- consensus_cluster(consensus_matri, nk = clu, reps = 10, algorithms = c("hc", "diana"),progress = FALSE)
  cspa_cluster_result=CSPA(x,k=clu)
  return (cspa_cluster_result)
}

g_cluster_result = consensus_cluster_func(consensus_g,2)
g2_cluster_result = consensus_cluster_func(consensus_g2,2)


#--------------------------
#- measuring cluster similarity (NMI,ARI,MI,VI,NVI..)
#--------------------------
#i am just comparing the ensemble clustering (cspa) with the groundtruth cluster
#https://cran.r-project.org/web/packages/aricode/aricode.pdf
#install.packages("aricode")

library(aricode)

comp_cluster<- function(cluster_result,ground_cluster){
  return(clustComp(cspa_cluster_result,gro_cluster))
}

cspa_vs_g_ground= clustComp(g_cluster_result,V(g)$value)
cspa_vs_g2_ground= clustComp(g2_cluster_result,V(g2)$value)

## the above two lines results needs to be plotted; i.e, the cspa_vs_g_ground$NMI and cspa_vs_g2_ground$NMI has to plotted as bar#
#################################################################################################################################
################################################################################################################################

### the MCLA clustering is not working
# library(IntClust)
# EnsembleClustering(List=x,type="data",distmeasure=c("tanimoto","tanimoto"),normalize=c(FALSE,FALSE),method=c(NULL,NULL),StopRange=FALSE,
#                    clust="agnes",linkage=c("flexible","flexible"),nrclusters=c(7,7),gap=FALSE,
#                    maxK=15,ensembleMethod="MCLA",executable=FALSE)
# 

#------------------------------------------------------------------------
#------------------------------ CENTRALITIES
#--------------------------------------------------------------------
#Source: https://www.datacamp.com/community/tutorials/centrality-network-analysis-R

#What Centralities Use?

# for karate
# Compare:
pr.cent<-proper_centralities(g)
calculate_centralities(g, include = pr.cent[1:49])%>%pca_centralities(scale.unit = TRUE)
lbest_Mod<- df_g$best_mod
# DF of Centralities:
dfC_g <- data.frame(lbest_Mod)
# We will use:
dfC_g$Stress <- calculate_centralities(g, include = "Stress Centrality" )$`Stress Centrality`
dfC_g$Group  <- calculate_centralities(g, include = "Group Centrality" )$`Group Centrality`
dfC_g$Eccentricity <- calculate_centralities(g, include = "Eccentricity Centrality" )$`Eccentricity Centrality`
dfC_g$Harary <- calculate_centralities(g, include = "Harary Centrality" )$`Harary Centrality`
dfC_g$Geodesic <- calculate_centralities(g, include = "Geodesic K-Path Centrality" )$`Geodesic K-Path Centrality`
dfC_g$ShortestPath <- calculate_centralities(g, include = "Shortest-Paths Betweenness Centrality" )$`Shortest-Paths Betweenness Centrality`
dfC_g$Entropy <- calculate_centralities(g, include = "Entropy Centrality" )$`Entropy Centrality`
dfC_g$FlowBetweenness <- calculate_centralities(g, include = "Flow Betweenness Centrality" )$`Flow Betweenness Centrality`
dfC_g$MNC <- calculate_centralities(g, include = "MNC - Maximum Neighborhood Component" )$`MNC - Maximum Neighborhood Component`
dfC_g$Degree <- calculate_centralities(g, include = "Degree Centrality" )$`Degree Centrality`
dfC_g$Laplacian <- calculate_centralities(g, include = "Laplacian Centrality" )$`Laplacian Centrality`
dfC_g$Closeness <- calculate_centralities(g, include = "Closeness Centrality (Freeman)" )$`Closeness Centrality (Freeman)`
View(dfC_g)

# for dolphin
pr.cent<-proper_centralities(g2)
pr.cent
calculate_centralities(g2, include = pr.cent[1:49])%>%pca_centralities(scale.unit = TRUE)
# DF of Centralities:
df_g2
lbest_Mod<- df_g2$best_mod
# DF of Centralities:
dfC_g2 <- data.frame(lbest_Mod)
# We will use:
dfC_g2$Eccentricity <- calculate_centralities(g2, include = "Eccentricity Centrality" )$`Eccentricity Centrality`
dfC_g2$clustering <- calculate_centralities(g2, include = "clustering coefficient" )$`clustering coefficient`
dfC_g2$Harary <- calculate_centralities(g2, include = "Harary Centrality" )$`Harary Centrality`
dfC_g2$load <- calculate_centralities(g2, include = "Load Centrality" )$`Load Centrality`
dfC_g2$Stress <- calculate_centralities(g2, include = "Stress Centrality" )$`Stress Centrality`
dfC_g2$kleinauth <- calculate_centralities(g2, include = "Kleinberg's authority centrality scores" )$`Kleinberg's authority centrality scores`
dfC_g2$kleinhub <- calculate_centralities(g2, include = "Kleinberg's hub centrality scores" )$`Kleinberg's hub centrality scores`
dfC_g2$eig_cen <- calculate_centralities(g2, include = "eigenvector centralities" )$`eigenvector centralities`
dfC_g2$loc_brid <- calculate_centralities(g2, include = "Local Bridging Centrality" )$`Local Bridging Centrality`
dfC_g2$k_core <- calculate_centralities(g2, include = "K-core Decomposition" )$`K-core Decomposition`
dfC_g2$leve_cent <- calculate_centralities(g2, include = "Leverage Centrality" )$`Leverage Centrality`
dfC_g2$cluster_rank <- calculate_centralities(g2, include = "ClusterRank" )$`ClusterRank`
View(dfC_g2)
dfC_g2[is.na(dfC_g2)] <- 0    

#------------------------------------------------------------------------------------------------
# ML modelling
#---------------------------------------------------------------------------------------------------

library(caret)
dfc_g_new<- dfC_g
dfc_g2_new<- dfC_g2

level_con<- function(df){
  df$lbest_Mod<- as.factor(df$lbest_Mod)
  if (length(levels(df$lbest_Mod))==2){
    levels(df$lbest_Mod)<-c("mod_m","mod_l")
  }
  else
  {
    levels(df$lbest_Mod)<-c("mod_r","mod_m","mod_l")
  }
  return(df)
}

dfc_g_new<- level_con(dfc_g_new)
dfc_g2_new<- level_con(dfc_g2_new)


set.seed(123)

#-------------------------------------
# to create training and test partition 
#-------------------------------------

partition_func<- function(df){
  return(createDataPartition(df$lbest_Mod,p=0.7,list = F))
}

partitionrule_g = partition_func(dfc_g_new)
trainingset_g = dfc_g_new[partitionrule_g, ]
nrow(trainingset_g)
length(trainingset_g$lbest_Mod)
testingset_g = dfc_g_new[-partitionrule_g, ]

partitionrule_g2 = partition_func(dfc_g2_new)
trainingset_g2 = dfc_g2_new[partitionrule_g2, ]
testingset_g2 = dfc_g2_new[-partitionrule_g2, ]

#-------------------------------------------------
# to create kfold cross validation
#-------------------------------------------------

splitrule <- trainControl(method="cv", number=4, savePredictions = T, classProbs = T)

rf_model_g<- train(lbest_Mod~., data=trainingset_g, trControl = splitrule, method="rf", metric = "Accuracy")
rf_test_g <- predict(rf_model_g, newdata=testingset_g)
confusionMatrix(data=rf_test_g,testingset_g$lbest_Mod)

rf_model_g2<- train(lbest_Mod~., data=trainingset_g2, trControl = splitrule, method="rf", metric = "Accuracy",na.action=na.exclude)
rf_test_g2 <- predict(rf_model_g2, newdata=testingset_g2,na.action = na.exclude)
confusionMatrix(data=rf_test_g2,testingset_g2$lbest_Mod)


#------------- cross fold with repeated cv -----------------------------------
splitrule_1 <- trainControl(method="repeatedcv", number=4,repeats = 3, savePredictions = T, classProbs = T)
rf_model_g_<- train(lbest_Mod~., data=trainingset_g, trControl = splitrule_1, method="rf", metric = "Accuracy")
rf_test_g_ <- predict(rf_model_g_, newdata=testingset_g)
confusionMatrix(data=rf_test_g_,testingset_g$lbest_Mod)


rf_model_g2_<- train(lbest_Mod~., data=trainingset_g2, trControl = splitrule_1, method="rf", metric = "Accuracy",na.action=na.exclude)
rf_test_g2_ <- predict(rf_model_g2_, newdata=testingset_g2,na.action = na.exclude)
confusionMatrix(data=rf_test_g2_,testingset_g2$lbest_Mod)


#-------------------ANN ---------------------------------------
#https://www.r-bloggers.com/multilabel-classification-with-neuralnet-package/
# https://datascienceplus.com/neuralnet-train-and-test-neural-networks-using-r/

require(neuralnet)
require(nnet)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
# for karate graph
maxmindf_g <- as.data.frame(lapply(dfC_g[,1:13], normalize))          # normalizing the entire dataframe including the label, In the label 3 becomes "1" and 2 becomes "0"
partitionrule_g_new = createDataPartition(dfC_g$lbest_Mod,p=0.7,list = F)
trainingset_g_new = maxmindf_g[partitionrule_g_new, ]
testingset_g_new = maxmindf_g[-partitionrule_g_new, ]
n_trainset_g<- cbind(trainingset_g_new[,-c(1)],class.ind(as.factor(trainingset_g_new$lbest_Mod)))
names(n_trainset_g)<- c(names(dfc_g_new)[-c(1)],"mod_m","mod_l")

nn_model_g<- neuralnet(mod_m+mod_l~.,data=n_trainset_g, hidden=3, act.fct="logistic",linear.output = F, lifesign = "minimal")
plot(nn_model_g)
nn_model_g$result.matrix
prediciton_nn_g <- compute(nn_model_g, trainingset_g_new[, 1:12])

pred_result_g<- prediciton_nn_g$net.result

# Accuracy (training set)
original_values_g <- max.col(n_trainset_g[,13:14])
prediciton_nn_acc <- max.col(pred_result_g)
mean(prediciton_nn_acc == original_values_g)


# for dolphin
dfC_g2[is.na(dfC_g2)] <- 0    # to remove the nan's
maxmindf_g2 <- as.data.frame(lapply(dfC_g2[,1:13], normalize))          # normalizing the entire dataframe including the label, In the label 3 becomes "1" and 2 becomes "0"
partitionrule_g2_new = createDataPartition(dfC_g2$lbest_Mod,p=0.7,list = F)
trainingset_g2_new = maxmindf_g2[partitionrule_g2_new, ]
testingset_g2_new = maxmindf_g2[-partitionrule_g2_new, ]
n_trainset_g2<- cbind(trainingset_g2_new[,-c(1)],class.ind(as.factor(trainingset_g2_new$lbest_Mod)))
names(n_trainset_g2)<- c(names(dfc_g2_new)[-c(1)],"mod_r","mod_m","mod_l")

nn_model_g2<- neuralnet(mod_r+mod_m+mod_l~.,data=n_trainset_g2, hidden=3, act.fct="logistic",linear.output = F, lifesign = "minimal")
plot(nn_model_g2)
nn_model_g2$result.matrix
prediciton_nn_g2 <- compute(nn_model_g2, trainingset_g2_new[, 1:12])

pred_result_g2<- prediciton_nn_g2$net.result

# Accuracy (training set)
original_values_g2 <- max.col(n_trainset_g2[,13:15])
prediciton_nn_acc <- max.col(pred_result_g2)
mean(prediciton_nn_acc == original_values_g2)

#-------------------------------------------------------------------------------------------------------------
### testing LRF model 

pr.cent<-proper_centralities(g1)
calculate_centralities(g1, include = pr.cent[1:49])%>%pca_centralities(scale.unit = TRUE)
lbest_Mod<- V(g1)$id
# DF of Centralities:
dfC_g1 <- data.frame(lbest_Mod)
# We will use:
dfC_g1$Stress <- calculate_centralities(g1, include = "Stress Centrality" )$`Stress Centrality`
dfC_g1$Group  <- calculate_centralities(g1, include = "Group Centrality" )$`Group Centrality`
dfC_g1$Eccentricity <- calculate_centralities(g1, include = "Eccentricity Centrality" )$`Eccentricity Centrality`
dfC_g1$Harary <- calculate_centralities(g1, include = "Harary Centrality" )$`Harary Centrality`
dfC_g1$Geodesic <- calculate_centralities(g1, include = "Geodesic K-Path Centrality" )$`Geodesic K-Path Centrality`
dfC_g1$ShortestPath <- calculate_centralities(g1, include = "Shortest-Paths Betweenness Centrality" )$`Shortest-Paths Betweenness Centrality`
dfC_g1$Entropy <- calculate_centralities(g1, include = "Entropy Centrality" )$`Entropy Centrality`
dfC_g1$FlowBetweenness <- calculate_centralities(g1, include = "Flow Betweenness Centrality" )$`Flow Betweenness Centrality`
dfC_g1$MNC <- calculate_centralities(g1, include = "MNC - Maximum Neighborhood Component" )$`MNC - Maximum Neighborhood Component`
dfC_g1$Degree <- calculate_centralities(g1, include = "Degree Centrality" )$`Degree Centrality`
dfC_g1$Laplacian <- calculate_centralities(g1, include = "Laplacian Centrality" )$`Laplacian Centrality`
dfC_g1$Closeness <- calculate_centralities(g1, include = "Closeness Centrality (Freeman)" )$`Closeness Centrality (Freeman)`

########################################### For TEsting the LRF with models that we created #########################################

rf_test_g1 <- predict(rf_model_g, newdata=dfC_g1[2:13])    # testing using Kfold rf model that was developed before
rf_test_g1


## LRF data testing using CSPA
dfC_lrf <- data.frame()
dfC_lrf_1 <- data.frame()
for (nodess in V(g1)){
  q= unlist(strsplit(V(g1)$community[nodess],",", fixed=FALSE))
  che = c()
  for (ele in q){
    che=append(che, as.numeric(ele))
  }
  dfC_lrf=rbind(dfC_lrf, c(che))
  che1 =c()
  for (num in seq(1:34)){
    if(num%in%che){
      che1= append(che1,1)
    }
    else{
      che1=append(che1,0)
    }
  }
  dfC_lrf_1<- rbind(dfC_lrf_1,che1)
}

lrf_groundtruth_cspa_community<- consensus_cluster_func(dfC_lrf_1,3)   # original consensus cluster for the ground truth 

# taking the mod results given in above result
level_con_1<- function(df){
  for(nodes in 1:nrow(df)){
    if(df$best_mod[nodes]=="mod_r"){df$lbest_mod[nodes]<-1}
    if(df$best_mod[nodes]=="mod_m"){df$lbest_mod[nodes]<-2}
    if(df$best_mod[nodes]=="mod_l"){df$lbest_mod[nodes]<-3}
  }
  return (df)
}

# for updating the id of g1
for (v in V(g1)$id){
  V(g1)$id[v]<-v
}

# To  formulate consensus matrix on the graph LRF

dfC_lrf_12<- level_con_1(lrf_test_df)
dfC_lrf_12 <- as.data.frame(dfC_lrf_12[,-c(1)])
names(dfC_lrf_12)<- c("best_mod")
dfC_lrf_12$nodes<- c(seq(1:34))

dfc_lrf_l23<- dfC_lrf_12[, c(2,1)]
g1_consensus_mat = consensus_func(g1,dfc_lrf_l23)


g_cluster_result_final_LRF = consensus_cluster_func(g1_consensus_mat,3)

#final comparison of LRF Ground truth vs the predicted
cspa_vs_g_ground_LRF_FINAL= clustComp(g_cluster_result_final_LRF,lrf_groundtruth_cspa_community[-c(1)])






