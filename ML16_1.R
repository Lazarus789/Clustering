dataset <- read.csv('Mall_Customers.csv')
x <- dataset[, 4:5]

dendrogram = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
plot(dendrogram, main = paste('Dendrogram'), xlab = 'Customers', ylab = 'Euclidean distances')

hc = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

library(cluster)
clusplot(x, y_hc, lines = 0, shade = TRUE, 
         color = TRUE, labels = 2, plotchar = FALSE, 
         span = TRUE, main = paste('cluster of Clients'), 
         xlab = "Annual Income", ylab = "Spending Score")



