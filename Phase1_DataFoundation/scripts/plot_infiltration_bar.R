# scripts/plot_infiltration_bar.R
# Creates stacked barplot of immune fractions (EPIC)

library(ggplot2)
library(reshape2)

# 1) Read the table produced by EPIC
df <- read.table("results/immune_infiltration_scores.tsv",
                 header = TRUE, sep = "\t",
                 check.names = FALSE, stringsAsFactors = FALSE)

# 2) Long format for ggplot
df_m <- melt(df, id.vars = colnames(df)[1])   # first column = sample id
colnames(df_m) <- c("Sample", "CellType", "Fraction")

# 3) IMPORTANT: make Sample categorical (fixes blank plot)
df_m$Sample <- factor(df_m$Sample, levels = unique(df_m$Sample))

# 4) Plot
p <- ggplot(df_m, aes(x = Sample, y = Fraction, fill = CellType)) +
  geom_bar(stat = "identity", width = 0.8) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
        panel.grid.minor = element_blank()) +
  ylab("Fraction") + xlab("Samples") +
  ggtitle("Immune Infiltration (EPIC)")

# 5) Save
dir.create("qc", showWarnings = FALSE)
ggsave("qc/immune_infiltration_barplot.png", p, width = 12, height = 6, dpi = 150)
cat("Saved plot -> qc/immune_infiltration_barplot.png\n")