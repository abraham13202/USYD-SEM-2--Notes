# Script to compile RMarkdown to PDF
# Run this to create your final submission PDF

# Install required packages if needed
packages <- c("rmarkdown", "knitr")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.rstudio.com/")
    library(pkg, character.only = TRUE)
  }
}

# Compile the RMarkdown file
rmarkdown::render("STAT5002_Assignment_Complete.Rmd",
                  output_format = "pdf_document")

cat("\n\n========================================\n")
cat("PDF compilation complete!\n")
cat("Your submission file is: STAT5002_Assignment_Complete.pdf\n")
cat("========================================\n\n")
