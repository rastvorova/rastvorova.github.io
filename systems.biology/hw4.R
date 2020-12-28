library('EnrichmentBrowser')
library("fgsea")
library('org.Hs.eg.db')

genesets = getGenesets(org = 'hsa', db = 'kegg', cache = F)
genesets = genesets[c(grepl('cancer', names(genesets)),
                      grepl('carcinoma', names(genesets)))]
genesets = genesets[!grepl('_in_cancer', names(genesets))]

analysis = function(num){
  expr = read.csv(paste0('data4/expr', num, '.tsv'), sep = '\t', header = TRUE)
  anno = read.csv(paste0('data4/anno', num, '.tsv'), sep = '\t', header = TRUE)
  genes = read.csv(paste0('data4/genes', num, '.tsv'), sep = '\t', header = TRUE)
  colnames(expr) = anno$Group
  rownames(expr) = genes$rownames.eset.
  expr = data.frame(expr)
  print('data ready')
  
  pv = apply(expr, 1, function (x) {
    t.test(expr[x, grepl('norm', names(expr))], 
           expr[x, grepl('cancer', names(expr))])$p.value})
  pv = p.adjust(pv, method = 'fdr')
  pv = data.frame(pv, entrez=mapIds(org.Hs.eg.db, names(pv), 'ENTREZID', 'SYMBOL'))
  print('pv ready')
  
  pv = pv[pv$pv < 0.01, ]
  pv = pv[pv$entrez %in% unique(unlist(genesets)), ]
  print(nrow(pv[pv$pv < 0.01,]))
  
#  writeLines(pv$entrez, paste0('webgestaldt', num, '.txt'))
#  print('wrote')
#  additional check
  
  ranks = apply(expr, 1, function (x) {
    log2(mean(x[grepl('cancer', names(expr))]) /
           mean(x[grepl('norm', names(expr))]))
  })
  names(ranks) = mapIds(org.Hs.eg.db, names(ranks), 'ENTREZID', 'SYMBOL')
  
  fgseaRes = fgsea(pathways = genesets, stats = sort(ranks), eps = 0.0)
  print(fgseaRes$pathway[fgseaRes$pval == min(fgseaRes$pval)])
  print(min(fgseaRes$pval))
  return(fgseaRes$pathway[fgseaRes$pval == min(fgseaRes$pval)])
}

pathways = c()
for(i in 1:13){
  pathways = c(pathways, analysis(i))
}

writeLines(pathways, 'hw4_Rastvorova.txt')
