# EISTI - ADEO2 - 2019/2020
# Social Network Analysis
# Professor: Kanawati
# Students: 
#   Gustavo Fleury Soares
#   Induraj P.
#   Quoc Viet Pham

library(igraph)
library(igraphdata)
library(CINNA)
g<-read.graph("karate.gml",format="gml")
g1<- read.graph("LRF3.gml",format="gml")
g2 <- read.graph("dolphins.gml",format="gml")  #Need update $id #OK
#g3 <- read.graph("polbooks.gml",format="gml")   #Need update $id and $value

# ----- for fixing the dolphin id's---------

for (v in V(g2)){
  V(g2)$id[v] = v
}

plot(g1, main='LFRGraph')
plot(g, main='KarateGraph')
plot(g2, main='dolphin')

basicTopFeaturesGraph <- function(g){
  cat('N Vertices: ', vcount(g))
  cat('\nN Edges   : ', ecount(g))
  cat('\nG Density : ', graph.density(g))
  cat('\nMax Degree: ', max(degree(g)))
  cat('\nIs Connect: ', is.connected(g))
  cat('\nL S Path  : ', max(shortest.paths(g)))
  cat('\nE Density : ', edge_density(g))
  cat('\nG Diameter: ', diameter(g))
  cat('\nClusterCoe: ', transitivity(g))
  cat(' (Transitivity!)')
  par(mfrow=c(2,2)) 
  plot(g, vertex.label=NA, main='Graph')
  plot(degree_distribution(g), main='DegreeDistribution')
  boxplot(degree_distribution(g), main='DegreeDistribution_BoxPlot')
}

# Exercise 1. Local Modularity R
mod_R <- function(g,C,B,S) {
  Bin  = length( E(g)[B%--%B] ) 
  Bout = length( E(g)[B%--%S] )
  return( Bin/(Bin+Bout) )
}

# Exercise 2. Local Modularity M
mod_M <- function(g,C,B,S) {
  D=c(C,B)
  Din  = length( E(g)[D%--%D] ) 
  Dout = length( E(g)[D%--%S] )
  return( Din/Dout )
}

# Exercise 3. Local Modularity L
neighborsin<-function(n,g,E){
  #returnsthenumberofneighborsofnodeninsetEingraphg
  return(length(intersect(neighbors(g,n),E)))
}
mod_L<-function(g,C,B,S){
  D<-union(C,B)
  lin<-sum(sapply(D,neighborsin,g,D))/length(D)
  lout<-sum(sapply(B,neighborsin,g,S))/length(B)
  return(lin/lout)
}


#Exercise 4. Local_com #Page:21
#TEACHER SOLUTION
update <- function(n,g,C,B,S){
  #moveninStoD
  S <- S[S!=n]
  D <- union(C,B)
  #cat('',n)
  if(all(neighbors(g,n)%in%D)){
    #addntoC
    C <- union(C,n)
  } else {
    #addntoB
    B <- union(B,n)
    news=setdiff(neighbors(g,n),union(D,S))
    if(length(news)>0){
      S <- union(S,news)
    }
    for (b in B){
      if(all(neighbors(g,b)%in%D)){
        B <- B[B!=b]
        C <- union(C,b)
      }
    }
  }
  return(list(C=C,B=B,S=S))
}

computequality <- function(n,g,C,B,S,mod){
  #computesthequalityofacommunityifnodenjoins
  #nisanodeinS
  res <- update(n,g,C,B,S)
  C <- res$C
  B <- res$B
  S <- res$S
  return(mod(g,C,B,S))
}

localcom <- function(target,g,mod){
  #initilizations
  if( is.igraph(g) && target %in% V(g)){
    C <- c()
    B <- c(target)
    S <- c(V(g)[neighbors(g,target)]$id)
    Q <- 0
    newQ <- 0
    while((length(S)>0)&&(newQ>=Q)){
      QS <- sapply(S,computequality,g,C,B,S,mod)
      newQ <- max(QS)
      if(newQ>=Q){
        snode <- S[which.max(QS)]
        res <- update(snode,g,C,B,S)
        C <- res$C
        B <- res$B
        S <- res$S
        Q <- newQ
      }
    }
    return(union(C,B))
  }else{
    stop("invalid arguments")
  }
}

computeegopartition <- function(target,g,mod){
  res <- vector(mode="list", length=2)
  names(res) <- c("com", "notcom")
  res$com <- localcom(target,g,mod)
  res$notcom <- V(g)[!(id %in% res$com)]$id
  #res
  return(res)
}

groundtruthlocalcomquality <- function(g,bipartition,method="nmi"){
  V(g)[id%in%bipartition$com]$egocom <- 0
  V(g)[id%in%bipartition$notcom]$egocom <- 1
  return(compare(V(g)$value,V(g)$egocom,method))
}

localcomquality <- function(target,g,mod,method){
  bipartition <- computeegopartition(target,g,mod)
  return(groundtruthlocalcomquality(g,bipartition,method))
}

labelLocalCom <- function(g,bipartition){
  V(g)[id%in%bipartition$com]$egocom <- 0
  V(g)[id%in%bipartition$notcom]$egocom <- 1
  return( V(g)$egocom )
}

displayCommunityQuality <- function(g,name){
  if(is.igraph(g)){
    mc <- multilevel.community(g)
    cat('\n multilevel.community      : ', compare(mc, V(g)$value, method="nmi") )
    bc <- edge.betweenness.community(g)
    cat('\n edge.betweenness.community: ', compare(bc, V(g)$value, method="nmi"))
    wc <- walktrap.community(g)
    cat('\n walktrap.community        : ', compare(wc, V(g)$value, method="nmi"))
    ic <- infomap.community(g)
    cat('\n infomap.community         : ', compare(ic, V(g)$value, method="nmi"))
    ic <- infomap.community(g)
    cat('\n infomap.community         : ', compare(ic, V(g)$value, method="nmi"))
    
    
    par(mfrow=c(2,2))    
    plot(mc, g, vertex.label=NA, main='Multilevel')
    plot(bc, g, vertex.label=NA, main='Betweenness')
    plot(wc, g, vertex.label=NA, main='Walktrap')
    plot(ic, g, vertex.label=NA, main='Infomap')
    par(mfrow=c(1,1))
  }
}


#-----------
# LOCAL COMMUNITY - MODULRATIES
#-----------
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
  View(df)
  return(df)
  }

df_g = loc_com(g)
df_g1 = loc_com(g1)
df_g2 = loc_com(g2)
#---------------cominne and rank apprioach

