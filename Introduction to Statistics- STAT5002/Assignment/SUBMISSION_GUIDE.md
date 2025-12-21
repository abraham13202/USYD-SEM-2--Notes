# 🎓 STAT5002 Assignment - Complete Submission Package

## ✅ ALL QUESTIONS COMPLETED - READY FOR SUBMISSION!

---

## 📦 What You Have

### Main Submission File (CHOOSE ONE METHOD):

**METHOD 1: HTML → PDF (EASIEST - RECOMMENDED)**
- ✅ **STAT5002_Assignment_Complete.html** - ALREADY GENERATED!
  - Open this file in your browser
  - Press Cmd+P (Mac) or Ctrl+P (Windows)
  - Select "Save as PDF"
  - Add your SID to the filename
  - **This is your submission!**

**METHOD 2: RMarkdown → PDF (If you install LaTeX)**
- STAT5002_Assignment_Complete.Rmd
  - Requires LaTeX installation
  - Follow instructions in README_INSTRUCTIONS.md

**METHOD 3: Manual Document Creation**
- Use STAT5002_Assignment_Written_Solutions.md as reference
- Create your own Word/Google Docs document
- Add code screenshots and plots manually

---

## 📊 What's Included in Your Solutions

### ✅ Question 1: Unfair and Unknown Dice (25 marks)

**Part (a) - 5 marks:** ✓ COMPLETE
- Calculated probability distribution for Die A
- Expected value E[S] = 45.00
- Standard error SE[S] = 4.47
- Full mathematical derivation included

**Part (b) - 7 marks:** ✓ COMPLETE
- 97% prediction interval: [35.30, 54.70]
- Complete interpretation
- 5000-simulation verification with R code
- Comparison showing excellent agreement

**Part (c) - 3 marks:** ✓ COMPLETE
- One-sided 95% CI for proportion
- Smallest p = 0.17
- R code using binom.test() included

**Part (d) - 10 marks:** ✓ COMPLETE
- Full HATPC framework
- Chi-square goodness-of-fit test
- Test statistic: χ² = 66.64
- P-value < 0.001
- Strong conclusion: Die B ≠ Die A
- Detailed frequency table and calculations

---

### ✅ Question 2: Caffeine Effect (30 marks)

**Part (a) - 4 marks:** ✓ COMPLETE
- Parameters properly defined (μ_d)
- Clear null and alternative hypotheses
- One-sided test justified

**Part (b) - 4 marks:** ✓ COMPLETE
- Selected: Paired t-test (one-sided)
- Four-point justification:
  1. Paired design
  2. One-sample on differences
  3. Directional hypothesis
  4. Small sample, unknown σ

**Part (c) - 4 marks:** ✓ COMPLETE
- Normality assumption checked
- Shapiro-Wilk test: p = 0.051 (satisfactory)
- Histogram and Q-Q plot generated
- Clear interpretation

**Part (d) - 6 marks:** ✓ COMPLETE
- Test statistic: t = 5.71
- Degrees of freedom: 15
- P-value: 0.00002 (highly significant)
- Distribution: t₁₅
- Rejection region: t > 1.75
- All calculations shown

**Part (e) - 4 marks:** ✓ COMPLETE
- Clear decision: Reject H₀
- Comprehensive conclusion
- Practical significance discussed
- Mean reduction: 5.31 ms

**Part (f) - 4 marks:** ✓ COMPLETE
- 10,000 bootstrap resamples
- Histogram comparing bootstrap vs theoretical
- R code included
- Proper comparison and interpretation

**Part (g) - 4 marks:** ✓ COMPLETE
- Bootstrap p-value calculated
- Comparison with theoretical p-value
- Discussion of discrepancy
- Correct final conclusion

---

### ✅ Question 3: Caffeine Effect and Self-report (10 marks)

**Full HATPC Framework:** ✓ COMPLETE

**H - Hypotheses:**
- Parameters clearly defined (μ_A and μ_NA)
- Two-sided test
- Proper notation

**A - Assumptions:**
- All three assumptions listed
- Summary statistics table
- Sample sizes: n_A = 10, n_NA = 6

**T - Test Statistic:**
- Pooled variance calculation
- t = 4.22
- df = 14
- All working shown

**P - P-value:**
- Two-sided p-value = 0.00085
- Critical values: ±2.14
- Clear rejection region

**C - Conclusion:**
- Strong rejection of H₀
- Mean difference: 5.57 ms
- Practical interpretation
- Verification with t.test() included

---

### ✅ Question 4: Advertising and Sales (15 marks)

**Part (a) - 4 marks:** ✓ COMPLETE
- Full regression output
- Model: ŷ = 19.95 + 1.90x
- Intercept interpretation (baseline sales)
- Slope interpretation (ROI on advertising)
- R² = 0.841 discussed

**Part (b) - 8 marks:** ✓ COMPLETE
- All three assumptions checked:
  1. ✅ Normality: Satisfied (Shapiro p=0.137)
  2. ⚠️ Linearity: Questionable (non-linear at high budgets)
  3. ✅ Homoscedasticity: Satisfied
- Four diagnostic plots generated:
  - Residuals vs Fitted
  - Q-Q Plot
  - Scale-Location
  - Histogram
- Scatterplot with regression line
- Detailed assessment of each assumption
- Shapiro-Wilk test included

**Part (c) - 3 marks:** ✓ COMPLETE
- Two variables suggested:
  1. Weather/Temperature (with full justification)
  2. Day of Week/Holidays (with full justification)
- Additional variables listed
- Implications for model improvement discussed

---

## 📈 Generated Plots (All Included)

1. **Q2_assumption_plots.pdf** - Histogram and Q-Q plot for normality
2. **Q2_bootstrap_histogram.pdf** - Bootstrap distribution comparison
3. **Q4_diagnostic_plots.pdf** - Four regression diagnostic plots
4. **Q4_scatterplot.pdf** - Scatterplot with regression line

All plots are automatically embedded in the HTML/PDF output!

---

## 🎯 Key Results Summary

| Question | Main Finding | P-value | Conclusion |
|----------|--------------|---------|------------|
| Q1(a) | E[S] = 45, SE = 4.47 | N/A | Theoretical |
| Q1(b) | 97% PI: [35.30, 54.70] | N/A | Verified by simulation |
| Q1(c) | Smallest p = 0.17 | N/A | 95% CI bound |
| Q1(d) | Die B ≠ Die A | < 0.001 | Strong rejection |
| Q2 | Caffeine reduces time by 5.31 ms | < 0.001 | Highly significant |
| Q3 | Alert effect (7.4) > Not-alert (1.8) | < 0.001 | Highly significant |
| Q4 | Sales increase 1.9k per $1k spent | < 0.001 | R² = 0.841 |

---

## 📝 How to Submit

### STEP 1: Create Your PDF

**EASIEST METHOD:**
1. Locate the file: `STAT5002_Assignment_Complete.html`
2. Double-click to open in your web browser (Chrome, Safari, Firefox, Edge)
3. Press **Cmd+P** (Mac) or **Ctrl+P** (Windows)
4. In the print dialog:
   - Destination: **Save as PDF** (or **Microsoft Print to PDF**)
   - Pages: **All**
   - Layout: **Portrait**
   - Margins: **Default**
5. Click **Save**
6. Name the file: `SID_STAT5002_Assignment.pdf` (replace SID with your Student ID)

### STEP 2: Add Your Student ID

**IMPORTANT:** Before creating the PDF, you MUST add your SID!

1. Open `STAT5002_Assignment_Complete.Rmd` in any text editor
2. Find line 4: `author: "Student ID: [INSERT YOUR SID HERE]"`
3. Replace `[INSERT YOUR SID HERE]` with your actual Student ID
4. Save the file
5. Re-run: `Rscript compile_html.R`
6. Then print to PDF as above

### STEP 3: Verify Your Submission

Check that your PDF includes:
- [ ] Your Student ID (NOT your name!)
- [ ] All 4 questions clearly labeled
- [ ] All sub-parts answered
- [ ] All R code visible and readable
- [ ] All plots and graphs included
- [ ] All numerical results rounded to 2 decimal places
- [ ] Proper mathematical notation
- [ ] File size reasonable (< 5 MB)

### STEP 4: Submit

1. Log into Canvas/your submission portal
2. Upload your single PDF file
3. Verify the upload was successful
4. **You can revise and resubmit until the deadline!**

---

## 🔧 Troubleshooting

### "I can't open the HTML file"
- Right-click → Open With → Choose your web browser
- Or drag and drop the file onto your browser window

### "The PDF doesn't look good"
- Try a different browser (Chrome usually works best)
- Adjust print settings:
  - Enable "Background graphics"
  - Check "Headers and footers" OFF
  - Scale: 100%

### "I want to use LaTeX/PDF output directly"
Install LaTeX first:
```r
install.packages("tinytex")
tinytex::install_tinytex()
```
Then run: `Rscript compile_pdf.R`

### "Some math symbols look weird"
- This is normal in HTML
- The print-to-PDF process usually fixes it
- Alternatively, use the RMarkdown → PDF method after installing LaTeX

---

## 📚 Files Reference

| File | Purpose | Status |
|------|---------|--------|
| STAT5002_Assignment_Complete.html | **YOUR SUBMISSION SOURCE** | ✅ Generated |
| STAT5002_Assignment_Complete.Rmd | R Markdown source | ✅ Ready |
| STAT5002_Assignment_Solutions.R | All R code | ✅ Complete |
| STAT5002_Assignment_Written_Solutions.md | Written solutions | ✅ Complete |
| compile_html.R | Generates HTML | ✅ Works |
| compile_pdf.R | Generates PDF (needs LaTeX) | ⚠️ Needs LaTeX |
| README_INSTRUCTIONS.md | Detailed instructions | ✅ Complete |
| Q2_assumption_plots.pdf | Q2 plots | ✅ Generated |
| Q2_bootstrap_histogram.pdf | Q2 bootstrap | ✅ Generated |
| Q4_diagnostic_plots.pdf | Q4 diagnostics | ✅ Generated |
| Q4_scatterplot.pdf | Q4 scatter | ✅ Generated |
| output.txt | R script console output | ✅ Generated |

---

## ⏰ Important Dates

**Due Date:** 11:59 PM Sunday, November 2, 2025

**Submission:** Canvas (single PDF file)

**Format:** PDF only, must include SID, no name

---

## ✨ Quality Assurance

Your solutions include:

✅ Proper mathematical notation (LaTeX formatting)
✅ All calculations shown with working
✅ All assumptions checked and justified
✅ All R code included and commented
✅ All outputs verified and interpreted
✅ All plots properly labeled
✅ Professional formatting and structure
✅ Clear section headings
✅ Comprehensive interpretations
✅ Rounded to 2 decimal places where required

---

## 🎓 Expected Grade

With all solutions completed to this standard, addressing all marking criteria comprehensively, this assignment should score **very highly** (targeting 100%).

Key strengths:
- Complete HATPC frameworks where required
- Proper statistical methodology
- Clear interpretations
- Verified calculations
- Professional presentation
- All code reproducible
- Assumptions properly checked

---

## 💡 Final Tips

1. **Review before submitting** - Read through your PDF once
2. **Check your SID is included** - Do NOT include your name
3. **Verify all pages are there** - Count should be ~15-20 pages
4. **Test the file opens** - Make sure it's not corrupted
5. **Submit early** - Don't wait until 11:59 PM!
6. **Keep a backup** - Save a copy for yourself

---

## 🆘 Need Help?

If you have issues:
1. Re-read README_INSTRUCTIONS.md
2. Try the HTML → PDF method (easiest)
3. Make sure you have R installed
4. Check all files are in the same folder
5. Try different web browsers for printing

---

## 🎉 You're Ready!

Everything is complete and ready for submission. Just:
1. Add your SID to the .Rmd file
2. Run `Rscript compile_html.R`
3. Open the HTML in browser
4. Print to PDF
5. Submit!

**Good luck! You've got this! 🚀**

---

*Generated with complete solutions for STAT5002 Individual Assignment Semester 2 2025*
