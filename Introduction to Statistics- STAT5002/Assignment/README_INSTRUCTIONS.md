# STAT5002 Assignment - Complete Solution Package

## Files Included

### Main Solution Files
1. **STAT5002_Assignment_Complete.Rmd** - R Markdown document with all solutions (RECOMMENDED)
2. **STAT5002_Assignment_Written_Solutions.md** - Markdown with detailed written solutions
3. **STAT5002_Assignment_Solutions.R** - Complete R script with all code

### Generated Output Files
- **Q2_assumption_plots.pdf** - Normality check plots for Question 2
- **Q2_bootstrap_histogram.pdf** - Bootstrap distribution for Question 2
- **Q4_diagnostic_plots.pdf** - Regression diagnostic plots for Question 4
- **Q4_scatterplot.pdf** - Scatterplot with regression line for Question 4
- **output.txt** - Console output from running the R script
- **STAT5002_Assignment_Workspace.RData** - Saved R workspace with all variables

---

## HOW TO CREATE YOUR FINAL PDF SUBMISSION

### OPTION 1: Using R Markdown (RECOMMENDED - EASIEST)

This is the **best option** as it will automatically compile everything into a beautiful PDF with proper formatting, all code, and all outputs.

#### Step 1: Install Required Packages (if not already installed)

Open R or RStudio and run:

```r
install.packages(c("rmarkdown", "knitr", "tinytex"))
tinytex::install_tinytex()  # This installs LaTeX if you don't have it
```

#### Step 2: Edit the Student ID

1. Open `STAT5002_Assignment_Complete.Rmd` in RStudio (or any text editor)
2. Find the line near the top that says:
   ```
   author: "Student ID: [INSERT YOUR SID HERE]"
   ```
3. Replace `[INSERT YOUR SID HERE]` with your actual Student ID
4. Save the file

#### Step 3: Compile to PDF

**In RStudio:**
- Open the .Rmd file
- Click the "Knit" button at the top
- Wait for it to compile (may take 1-2 minutes)
- A PDF will be generated automatically

**From R Console:**

```r
rmarkdown::render("STAT5002_Assignment_Complete.Rmd")
```

**From Terminal/Command Line:**

```bash
cd "/Users/ABRAHAM/Documents/USYD/Sem 2/Introduction to Statistics- STAT5002/Assignment"
Rscript -e "rmarkdown::render('STAT5002_Assignment_Complete.Rmd')"
```

#### Result:
You'll get a file called `STAT5002_Assignment_Complete.pdf` which contains:
- All written solutions with proper mathematical notation
- All R code
- All outputs and results
- All plots and graphs
- Properly formatted and numbered sections

**THIS IS YOUR SUBMISSION FILE!**

---

### OPTION 2: Manual Compilation (Alternative Method)

If you prefer to create your own document or can't use R Markdown:

#### Step 1: Use the Markdown File as Reference

Open `STAT5002_Assignment_Written_Solutions.md` - this contains all the written solutions with proper mathematical notation.

#### Step 2: Create Your Document

You can:
- Copy content into Microsoft Word
- Use Google Docs
- Use LaTeX directly
- Use any other PDF-creation tool

#### Step 3: Add Code Screenshots

1. Open `STAT5002_Assignment_Solutions.R`
2. Take screenshots of relevant code sections for each question
3. Insert them into your document

#### Step 4: Add Plot Images

Insert the generated PDF plots:
- Q2_assumption_plots.pdf (for Q2 part c)
- Q2_bootstrap_histogram.pdf (for Q2 part f)
- Q4_diagnostic_plots.pdf (for Q4 part b)
- Q4_scatterplot.pdf (for Q4 part b)

#### Step 5: Add Outputs

Copy relevant numerical results from `output.txt` into your document.

---

### OPTION 3: Using Pandoc (Advanced)

If you have Pandoc installed:

```bash
cd "/Users/ABRAHAM/Documents/USYD/Sem 2/Introduction to Statistics- STAT5002/Assignment"
pandoc STAT5002_Assignment_Written_Solutions.md -o MyAssignment.pdf --pdf-engine=xelatex
```

---

## Verification Checklist

Before submitting, make sure your PDF includes:

### Question 1 (25 marks)
- [✓] Part (a): Expected value and SE calculations with formulas
- [✓] Part (b): 97% PI calculation, interpretation, simulation code and results
- [✓] Part (c): Smallest p calculation with R code
- [✓] Part (d): Complete HATPC framework for chi-square test with table

### Question 2 (30 marks)
- [✓] Part (a): Hypotheses with proper parameter definition
- [✓] Part (b): Test selection and justification
- [✓] Part (c): Normality assumption check with plots and Shapiro-Wilk test
- [✓] Part (d): Test statistic, p-value, distribution, rejection region
- [✓] Part (e): Conclusion based on p-value
- [✓] Part (f): Bootstrap code and histogram comparing with theoretical
- [✓] Part (g): Bootstrap p-value and conclusion

### Question 3 (10 marks)
- [✓] Complete HATPC framework
- [✓] Hypotheses with parameter definitions
- [✓] Assumptions stated
- [✓] Test statistic calculations (pooled t-test)
- [✓] P-value calculation
- [✓] Conclusion with interpretation

### Question 4 (15 marks)
- [✓] Part (a): Regression output and interpretation of both coefficients
- [✓] Part (b): All three assumptions checked (normality, linearity, homoscedasticity)
- [✓] Part (b): Diagnostic plots included
- [✓] Part (c): Two other variables with justifications

### General Requirements
- [✓] Student ID included (NOT your name!)
- [✓] All answers clearly labeled
- [✓] Proper mathematical notation
- [✓] All code included (either in text or as clear screenshots)
- [✓] All numerical answers rounded to 2 decimal places
- [✓] Single combined PDF file

---

## Quick Start Commands

If you just want to get the PDF quickly:

```bash
# Navigate to the directory
cd "/Users/ABRAHAM/Documents/USYD/Sem 2/Introduction to Statistics- STAT5002/Assignment"

# Option 1: Use R to compile RMarkdown
Rscript -e "rmarkdown::render('STAT5002_Assignment_Complete.Rmd')"

# Option 2: Just run the R script to regenerate all outputs
Rscript STAT5002_Assignment_Solutions.R
```

---

## File Sizes and Preview

All files have been generated and tested:
- R Markdown: ~27 KB
- R Script: ~19 KB
- Plots: ~6-10 KB each (PDF format, high quality)
- Expected final PDF: ~300-500 KB

---

## Troubleshooting

### "Error: LaTeX not found"
- Install tinytex: `tinytex::install_tinytex()`
- Or install MiKTeX/MacTeX separately

### "Error: Package 'rmarkdown' not found"
- Run: `install.packages("rmarkdown")`

### Plots not showing
- Make sure all PDF plot files are in the same directory
- Re-run the R script to regenerate: `Rscript STAT5002_Assignment_Solutions.R`

### Math notation not rendering
- Make sure you're using PDF output (not HTML)
- LaTeX must be installed for proper math rendering

---

## Support

If you encounter issues:
1. Check that all required R packages are installed
2. Verify LaTeX is installed (for PDF generation)
3. Make sure all files are in the same directory
4. Try running the R script first to regenerate all outputs

---

## Final Notes

- **DO NOT include your name** - only your Student ID
- **Submit only ONE PDF file**
- **Due: 11:59 pm Sunday 02 Nov 2025**
- Review your submission carefully before uploading
- You can revise and resubmit until the due date

Good luck with your assignment! 🎯
