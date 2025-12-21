# Sydney Airbnb Price Classification - Final Presentation

## Overview
This presentation summarizes the STAT5003 group project on predicting Sydney Airbnb price categories using machine learning classification models.

## Files in this Folder
- `Final_Presentation_Clean.Rmd` - **RECOMMENDED: Clean presentation file (Slidy format, NO blank pages)**
- `Final_Presentation.Rmd` - Original ioslides version (may have spacing issues)
- `Presentation_Script.md` - Complete speaking script with timing
- `styles.css` - Custom CSS styling for the presentation
- `README.md` - This file with instructions

## IMPORTANT: Use Final_Presentation_Clean.Rmd

The clean version uses Slidy format which provides:
- NO blank pages between slides
- Exactly 17 content slides
- Better navigation and flow
- Cleaner rendering

## How to Generate the Presentation

### Prerequisites
Make sure you have the following R packages installed:
```r
install.packages(c("tidyverse", "ggplot2", "gridExtra", "scales", "knitr", "caret", "rmarkdown"))
```

### Steps to Knit the Presentation

1. **Open RStudio**

2. **Open the Presentation File**
   - Navigate to this folder
   - Open `Final_Presentation_Clean.Rmd`

3. **Verify Data Path**
   - The presentation loads data from: `../Assignment2/listings_cleaned_with_features.csv`
   - Ensure this file exists in the Assignment2 folder

4. **Knit the Presentation**
   - Click the "Knit" button in RStudio
   - Or run: `rmarkdown::render("Final_Presentation_Clean.Rmd")`

5. **Output**
   - An HTML file will be generated: `Final_Presentation_Clean.html`
   - Open it in any web browser to view the presentation

## Presentation Format

The presentation uses **Slidy** format which provides:
- Clean, professional slides
- NO blank pages
- Easy navigation with arrow keys or Page Up/Down
- Click on slide headings at top for quick navigation
- Smooth scrolling between slides

## Presentation Structure (7 minutes)

The presentation covers:

1. **Introduction** (1 min)
   - Research question
   - Why it matters
   - Dataset overview

2. **Data Processing** (1.5 min)
   - Cleaning approach
   - Key visualizations of the data
   - Geographic patterns

3. **Methods** (1 min)
   - Five ML models tested
   - Training strategy

4. **Results** (2 min)
   - Model performance comparison
   - Feature importance
   - Category-specific results

5. **Insights & Limitations** (1 min)
   - Key findings
   - Challenges faced
   - Study limitations

6. **Conclusion** (0.5 min)
   - Link back to research question
   - Final takeaways

## Tips for Presenting

- **Practice timing:** Aim for 7 minutes total
- **Use visualizations:** Let the graphs tell the story
- **Keep it simple:** Explain to a non-technical audience
- **Engage the audience:** Ask rhetorical questions
- **Highlight insights:** Focus on "what we learned" not "what we did"

## Customization

### To change presentation theme:
Edit the YAML header in `Final_Presentation.Rmd`:
```yaml
output:
  ioslides_presentation:
    widescreen: true
    smaller: false
```

### To use different output format:
You can also generate Slidy or Beamer presentations:
```yaml
output:
  slidy_presentation: default
  # or
  beamer_presentation: default
```

## Q&A Preparation (3 minutes)

Be prepared to answer questions about:
- Why you chose these specific models
- How you handled missing data
- What the business implications are
- How the model could be improved
- What features matter most and why

## Contact

For questions about this presentation, contact Group W14_G03.

---

**STAT5003 - Computational Statistical Methods**
**University of Sydney | 2025**
