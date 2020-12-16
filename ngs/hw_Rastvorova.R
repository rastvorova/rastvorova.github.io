library(edgeR)
data = read.csv("01. RiboSeq_RNASeq_HCC_counts.tsv", header=TRUE, sep="\t")
rownames(data) = data$geneID
data[is.na(data)] = 0

RNAnames =  colnames(data)[grep('RNA', colnames(data))]
Ribonames = colnames(data)[grep('RPF', colnames(data))]

RNAdge = DGEList(counts = data[,RNAnames], 
                 group = substr(RNAnames, 7, 7), 
                 genes = data$geneSymbol)

Ribodge = DGEList(counts = data[,Ribonames], 
                  group = substr(Ribonames, 7, 7), 
                  genes = data$geneSymbol)

dge = calcNormFactors(RNAdge)
dge = estimateCommonDisp(dge)
dge = estimateTagwiseDisp(dge)
dgeTest = exactTest(dge)
resdge = topTags(dgeTest, n=nrow(dgeTest$table))
volcanoDataRNA = cbind(resdge$table$logFC, -log10(resdge$table$FDR))
colnames(volcanoDataRNA) = c("log FC", "neg Log P-value")
plot(volcanoDataRNA, 
     xlim=c(-7, 8), 
     ylim=c(0, 30), 
     main='RNASeq',
     col = ifelse(volcanoDataRNA[,"neg Log P-value"] < 1.3, 'red', 'darkgreen'))

dge = calcNormFactors(Ribodge)
dge = estimateCommonDisp(dge)
dge = estimateTagwiseDisp(dge)
dgeTest = exactTest(dge)
resdge = topTags(dgeTest, n=nrow(dgeTest$table))
volcanoDataRibo = cbind(resdge$table$logFC, -log10(resdge$table$FDR))
colnames(volcanoDataRibo) = c("log FC", "neg Log P-value")
plot(volcanoDataRibo, 
     xlim=c(-7, 8), 
     ylim=c(0, 30), 
     main='RiboSeq', 
     col = ifelse(volcanoDataRibo[,"neg Log P-value"] < 1.3, 'red', 'darkgreen'))
