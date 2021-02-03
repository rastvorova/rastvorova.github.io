##2

#transcript heatmap
transcript = read.csv("trans_counts.csv")
rownames(transcript) = transcript$transcript_id
transcript$transcript_id = NULL
transcript = transcript[apply(transcript, 1, sum) > 0,]
spearman = cor(transcript , method = 'spearman')
heatmap(spearman, symm = TRUE, main = 'Transcript')

#genes heatmap
genes = read.table('gene_counts.csv', sep = ',', header = T)
rownames(genes) = genes$gene_id
genes$gene_id = NULL
genes = genes[apply(genes, 1, sum) > 0,]
spearman = cor(genes, method = 'spearman')
heatmap(spearman, symm = TRUE, main = 'Genes')

#PCA
summary(prcomp(transcript, center = TRUE, scale = TRUE))
summary(prcomp(genes, center = TRUE, scale = TRUE))


##3

library(edgeR)
# Find genes with interstitial and/or age-related changes in expression
# Corrected p-value <0.05
edger = DGEList(counts = genes, 
                group = substr(colnames(genes), 1, 1),
                genes = rownames(genes))
tissue = substr(colnames(genes), 1, 1)
age = substr(colnames(genes), 2, 5)
design = model.matrix(~ tissue + age)
edger = calcNormFactors(edger, method = 'RLE')
edger = estimateDisp(edger, design)
plotBCV(edger)

glm = glmFit(edger, design)
glm_pv_2 = glmLRT(glm, 2)
tissue_adj = p.adjust(glm_pv_2$table$PValue, method = 'BH')
table(tissue_adj < 0.05)

glm_pv_3 = glmLRT(glm, 3)
age_adj = p.adjust(glm_pv_3$table$PValue, method = "BH")
table(age_adj < 0.05)

#Scalsterize significant genes into 6 clusters.
tissue_filter = glm_pv_2$table[tissue_adj < 0.05 & 
                                 (glm_pv_2$table$logFC >= 1 | glm_pv_2$table$logFC <= -1),]

age_filter = glm_pv_3$table[age_adj < 0.05,]
sign_genes = genes[rownames(genes) %in% unique(c(rownames(tissue_filter), rownames(age_filter))),]

spearman = cor(t(sign_genes), method = 'spearman')
distance = dist(1 - spearman)
clusters = cutree(hclust(distance, method='average'), k = 6)
table(clusters)

# Scale the expression of each gene to mean zero and variance one (z-score)
# Draw mean z-score versus age for both tissues for each cluster
scale = t(scale(t(sign_genes), center = TRUE, scale = TRUE))

par(mfrow = c(3, 2), 
    mar = c(3, 3, 2, 1), 
    mgp = c(1.3, 0.2, 0))

for(i in 1:6){
  plot(1, 
       type = "n", 
       xlab = "age", 
       ylab = "mean z-score", 
       xlim = c(0, 4), 
       ylim = c(-1, 3),
       xaxt = 'n', 
       main = paste('Claster ', i))
  legend('topright', c('B', 'C'), fill = c('orange', 'green'))
  lines(seq(0,4), apply(scale[clusters == i, 1:5], 2, mean), col = 'orange')
  lines(seq(0,4), apply(scale[clusters == i, 6:10], 2, mean), col = 'green')
  axis(side = 1, at = seq(0, 4), labels = unique(age))
}
