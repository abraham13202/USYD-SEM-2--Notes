# Script to compile RMarkdown to HTML
# You can then print this HTML to PDF using your browser

# Install required packages if needed
packages <- c("rmarkdown", "knitr")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.rstudio.com/")
    library(pkg, character.only = TRUE)
  }
}

# Compile to HTML
rmarkdown::render("STAT5002_Assignment_Complete.Rmd",
                  output_format = "html_document")

cat("\n\n========================================\n")
cat("HTML compilation complete!\n")
cat("Your file is: STAT5002_Assignment_Complete.html\n")
cat("\nTo convert to PDF:\n")
cat("1. Open the HTML file in your web browser\n")
cat("2. Press Cmd+P (Mac) or Ctrl+P (Windows)\n")
cat("3. Select 'Save as PDF'\n")
cat("4. Save with your SID in the filename\n")
cat("========================================\n\n")
