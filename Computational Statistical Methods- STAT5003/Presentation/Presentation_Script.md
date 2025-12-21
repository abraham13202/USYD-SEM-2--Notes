# Sydney Airbnb Price Classification - Presentation Script
## 7-Minute Presentation Guide

**Total Time: 7 minutes**
**Group: W14_G03**

---

## SLIDE 1: Title Slide (10 seconds)
**[Display title slide]**

"Good [morning/afternoon] everyone. I'm [Your Name] from Group W14_G03, and today we're going to tell you a data science story about Sydney's Airbnb market."

---

## SLIDE 2: Our Data Science Journey (30 seconds)

"Our project asks a simple question: Can we predict whether a Sydney Airbnb will be Budget, Mid-Market, or Premium based on its features?

Now, you might wonder - why does this matter?

Well, this helps property investors make smart decisions about where to buy and what to renovate. It guides Airbnb hosts in setting competitive prices. And it provides valuable insights for Sydney's tourism planning and housing affordability discussions.

Think of it this way - if you're thinking about buying an investment property in Sydney, wouldn't you want to know what makes some Airbnbs charge $300 a night while others charge $80?"

---

## SLIDE 3: Research Question & Impact (30 seconds)

"So our big question is: Can we predict Airbnb price categories using property characteristics, location, and host factors?

This matters to four main groups:

For travelers - it helps you find the right accommodation in your budget.
For property owners - you'll understand what drives premium pricing.
For Sydney - we get insights into the short-term rental market dynamics.
And for policymakers - this provides evidence for housing policy decisions.

Essentially, we're turning raw data into actionable insights."

---

## SLIDE 4: The Data Story (40 seconds)

"Let's talk about our data. We analyzed the Inside Airbnb Sydney dataset from 2025, which contains information on over 15,000 listings with dozens of features about each property.

**[Point to the bar chart]**

We created three price categories: Budget properties under $100 per night, Mid-Market between $100-$200, and Premium over $200.

As you can see from this chart, the market is fairly balanced - Budget properties make up about 47% of the market, Mid-Market is 41%, and Premium properties are about 13%.

This distribution tells us that Sydney has a diverse accommodation market serving different traveler budgets."

---

## SLIDE 5: How We Cleaned The Data (35 seconds)

"Now, data science isn't glamorous at first. We started with messy, real-world data.

The raw dataset had over 18,000 listings with 79 different features - things like prices with dollar signs and commas, missing values for reviews and bathrooms, and extreme outliers like properties charging $5,000 per night.

Our cleaning process involved four key steps:
- We removed all the formatting from price data
- We handled missing values using smart imputation techniques
- We filtered out extreme outliers beyond $1,000 per night
- And we created new features from existing data

The result? A clean dataset of 15,544 listings ready for machine learning."

---

## SLIDE 6: Feature Engineering - Creating Useful Variables (40 seconds)

"Here's where we got creative. Raw data doesn't always capture what actually matters for pricing.

We engineered 6 new features that capture business logic:

**From location data**, we calculated distance from Sydney's CBD using the latitude and longitude coordinates. We also created a flag for popular tourist areas - Bondi, Manly, CBD, Darlinghurst, and Surry Hills.

**From property data**, we categorized property size into Small, Medium, Large, and Extra Large based on guest capacity. We also counted the number of amenities each property offers.

**And from host data**, we categorized host experience - are they managing a single property, a small portfolio, or a large portfolio? We also classified availability into Low, Medium, or High.

Why does this matter? These derived features capture patterns that raw numbers alone would miss. For example, 'distance from CBD' is more meaningful for pricing than separate latitude and longitude values."

---

## SLIDE 7: Feature Selection - Why These 19 Features? (35 seconds)

"You might wonder - why narrow down from 79 features to just 19?

We used four key criteria:

First, **relevance to pricing** - does this feature actually affect what people pay for an Airbnb?

Second, **data quality** - is the feature reliable and complete, or full of missing values?

Third, **avoid redundancy** - we don't want highly correlated features that tell us the same thing.

And fourth, **business interpretability** - can we explain this feature to property investors or hosts in plain English?

**[Point to the slide]**

Our final 19 features break down into four categories: 10 property characteristics like bedrooms and room type, 3 location features, 4 host quality indicators, and 2 social proof metrics like review scores.

This focused set gives us strong predictive power while keeping the model interpretable."

---

## SLIDE 8: What Does Our Data Look Like? (40 seconds)

"Once the data was clean, patterns started emerging immediately.

**[Point to the three boxplots]**

Look at these boxplots. They show guest capacity, bedrooms, and amenities across our three price categories.

Notice how Premium properties consistently have higher values? They accommodate more guests, have more bedrooms, and offer more amenities.

But here's what's interesting - there's also overlap between categories. Some Budget properties have features similar to Mid-Market ones, and some Mid-Market properties look like Premium ones. This overlap is exactly why we need machine learning - the boundaries aren't always clear cut.

This is the challenge we're trying to solve."

---

## SLIDE 9: Where Are The Properties? (35 seconds)

"Location, location, location - it's the oldest rule in real estate, and our data proves it.

**[Point to the map]**

This map shows all Sydney Airbnb properties. Green dots are Budget, orange are Mid-Market, and red are Premium.

See that concentration of red dots in the center and along the coast? That's the CBD, Bondi, and Manly - Sydney's most desirable areas.

Meanwhile, Budget properties - the green dots - are spread more evenly across Sydney's suburbs.

This geographic clustering tells us that where you are in Sydney matters just as much as what your property offers."

---

## SLIDE 10: Location Matters! (30 seconds)

"Let me drive this point home with data.

**[Point to the boxplot]**

This chart shows distance from Sydney's CBD for each price category. The pattern is crystal clear:

Premium properties are closest to the city center - typically within 5 kilometers.
Mid-Market properties are a bit further out.
And Budget properties are the furthest from the CBD.

This was one of our most significant findings - location emerged as the number one predictor of pricing. In Sydney's Airbnb market, proximity to the CBD can mean the difference between charging $80 and $250 per night."

---

## SLIDE 11: Room Type Tells a Story (30 seconds)

"Here's another interesting pattern we discovered.

**[Point to the stacked bar chart]**

This chart shows room type composition across price categories.

For Premium properties, almost all of them are entire homes or apartments. Makes sense, right? If you're paying over $200 a night, you want the whole place to yourself.

Mid-Market is mixed - some entire homes, some private rooms.

But look at Budget - there's significant variety including shared rooms and private rooms.

The takeaway? Room type is a strong signal of pricing tier in Sydney's market."

---

## SLIDE 12: Our Machine Learning Approach (35 seconds)

"So how did we actually build our prediction model?

We tested five different machine learning algorithms:
- Logistic Regression as our simple, interpretable baseline
- Random Forest to handle complex patterns with multiple features
- Support Vector Machine to create smart decision boundaries
- Linear Discriminant Analysis for efficient dimension reduction
- And K-Nearest Neighbors to leverage geographic patterns

Our training strategy was rigorous:
We split the data 70% for training and 30% for testing.
We used 5-fold cross-validation with 3 repetitions to ensure robustness.
And we performed hyperparameter tuning for optimal performance.

This comprehensive approach ensures our results are reliable and not just lucky guesses."

---

## SLIDE 13: Model Performance: The Results! (40 seconds)

"And here are the results!

**[Point to the bar chart]**

Random Forest was our champion, achieving 78.9% accuracy. That means it correctly classified nearly 8 out of every 10 properties.

To put this in perspective - if we just randomly guessed, we'd only get about 33% accuracy. If we always guessed the most common category, we'd get 47%.

So 79% is substantially better than any naive approach.

Support Vector Machine came in second at 76.5%, followed by Logistic Regression at 72.5%.

What's impressive is that ALL five models exceeded 70% accuracy, which tells us that Airbnb pricing patterns are genuinely learnable from property characteristics."

---

## SLIDE 14: What Features Matter Most? (40 seconds)

"You might be wondering - what actually drives these predictions?

From our Random Forest model, here are the top 5 most important features:

Number 1: Distance from CBD - location is king, as we've seen.
Number 2: Number of bedrooms - size matters.
Number 3: Guest capacity - how many people can stay.
Number 4: Neighbourhood - specific areas like Bondi, Manly, and the CBD command premiums.
And Number 5: Amenities count - more amenities, higher price.

What surprised us? Superhost status influences pricing significantly - quality of service matters.

Also, room type turned out to be a stronger predictor than property type. It's not whether you have a house versus an apartment - it's whether guests get the entire place that matters."

---

## SLIDE 15: Model Performance By Category (35 seconds)

"Let's look at performance across the three categories.

**[Point to the grouped bar chart]**

For Budget properties, all models performed well - Random Forest achieved an F1 score of 0.84. Budget properties have distinctive characteristics that make them easy to identify.

Premium properties were harder - around 0.68 F1 score. This makes sense because Premium is the smallest category with only 13% of properties.

But look at Mid-Market - this is the hardest category to classify correctly.

Why? Because Mid-Market properties share features with both Budget AND Premium. They're in the middle, creating natural overlap. This is where most of our classification errors occurred."

---

## SLIDE 16: Challenges We Faced (35 seconds)

"No data science project is without challenges. Here are the difficult parts we encountered:

First, class imbalance. Budget and Mid-Market dominate the dataset at 46% and 41%, while Premium is only 13%. This makes Premium properties harder to predict.

Second, boundary ambiguity. Properties priced near $100 or $200 thresholds are naturally tricky because they could reasonably fall into either adjacent category.

Third, that Mid-Market confusion we just discussed - it gets confused with both Budget AND Premium categories.

And fourth, missing data. Reviews, bathrooms, and host information were often missing, requiring us to develop smart imputation strategies.

Despite these challenges, we still achieved strong predictive performance."

---

## SLIDE 17: Limitations & What We'd Do Differently (30 seconds)

"Let's be honest about our study's limitations.

Our data is a single snapshot in time, so we can't capture seasonal trends or how prices change throughout the year.

These are self-reported prices, which may not reflect actual booking rates or revenue.

And these patterns are Sydney-specific - they might not generalize to Melbourne or Brisbane.

If we had more time and resources, we'd love to:
- Add time series modeling for seasonal patterns
- Perform text analysis on reviews and descriptions
- Build dynamic pricing predictions
- And compare across multiple Australian cities

These would make our model even more powerful and practical."

---

## SLIDE 18: Key Insights & Takeaways (45 seconds)

"So what did we actually learn from all this analysis?

Here are our five key insights:

One: Location is king. Properties within 5 kilometers of the CBD command 2 to 3 times the premium of outer suburbs. This is the single most important factor.

Two: Size matters. Each additional bedroom increases the likelihood of Premium classification by about 40%.

Three: Machine learning works. We achieved 79% accuracy - vastly better than random guessing at 33%.

Four: Market segmentation is clear. Our three-tier classification aligns with how real people think about accommodation budgets.

And five: Geographic patterns are pronounced. Premium properties cluster in Bondi, Manly, the CBD, and harbour-adjacent areas.

These aren't just statistics - they're actionable insights for real business decisions."

---

## SLIDE 19: Practical Recommendations (40 seconds)

"Let's make this practical. Here's what different stakeholders should do with our findings:

**For Property Investors:**
Location should be your first priority. CBD proximity is the number one driver of pricing.
If you want to move from Budget to Mid-Market tier, add bedrooms - that's your best investment.
But be aware: amenity upgrades have diminishing returns for Budget properties.

**For Airbnb Hosts:**
Use our model predictions to anchor your competitive pricing.
Work toward Superhost status - it genuinely matters for pricing power.
And if you want Premium rates, you need a complete amenity package, not just one or two standout features.

**For Policy Makers:**
The good news is 46.5% of properties are Budget-tier, supporting diverse tourism demographics.
But be aware that Premium concentration in high-demand areas may be impacting long-term housing availability.
Consider tier-specific regulations based on these market segments."

---

## SLIDE 20: Linking Back to Our Research Question (40 seconds)

"Let's come full circle to our original research question:

Can we predict whether a Sydney Airbnb property will be classified as Premium, Mid-Market, or Budget based on property characteristics, location and host factors?

Our answer is a definitive YES.

Here's the evidence:
- Our Random Forest model achieved 78.9% accuracy
- All five models exceeded 70% accuracy, compared to a 33% baseline
- We identified a clear feature importance hierarchy
- And we validated geographic and property patterns

The bottom line? Machine learning successfully predicts Airbnb price categories, providing actionable insights for investors, hosts, and policymakers in Sydney's short-term rental market.

This isn't just an academic exercise - it's a practical tool that could help real people make better decisions about Sydney's housing and tourism markets."

---

## SLIDE 21: Thank You! (15 seconds)

"To wrap up:

We analyzed over 18,000 Airbnb listings.
We trained and compared 5 machine learning models.
We achieved 79% prediction accuracy.
And we delivered clear insights for Sydney's rental market.

The big takeaway? Data science can transform complex housing data into practical decision-support tools.

Thank you for your attention. We're happy to take questions."

---

## Q&A PREPARATION (3 minutes)

### Common Questions & Suggested Answers:

**Q: Why did Random Forest perform best?**
A: "Random Forest excels at handling mixed data types - we have both numeric features like bedrooms and categorical features like neighbourhood. It also naturally captures non-linear relationships and feature interactions without us having to specify them. For example, it can learn that 'distance from CBD' matters differently depending on the neighbourhood, which simpler models can't easily capture."

**Q: How did you handle missing data?**
A: "Great question. We used domain knowledge to guide our imputation strategy. For reviews per month, missing meant no reviews, so we used zero. For bathrooms, we used median imputation. For bedrooms when missing, we estimated based on guest capacity divided by 2. We avoided deleting rows because we wanted to preserve as much data as possible."

**Q: Could your model be used in other cities?**
A: "The framework absolutely could be applied to other cities, but the specific patterns would differ. For instance, in Melbourne, proximity to the CBD might matter less because Melbourne is more decentralized. We'd need to retrain the model on local data and potentially engineer city-specific features."

**Q: What about seasonal pricing variations?**
A: "That's a great limitation of our study. We used a snapshot dataset, so we can't capture how prices fluctuate during peak tourist seasons like summer or major events like New Year's Eve. A time-series approach with historical data would be a valuable extension."

**Q: Is 79% accuracy good enough for real-world use?**
A: "For a three-class classification problem, 79% is quite strong - much better than random guessing. In practice, even if the model isn't perfect, it provides a data-driven starting point for pricing decisions. Hosts could use it to anchor their prices, then adjust based on their specific circumstances. It's a decision-support tool, not a replacement for human judgment."

**Q: What would improve your model?**
A: "Several things: First, actual booking and revenue data rather than listed prices. Second, text analysis of reviews and descriptions to capture qualitative factors. Third, temporal data for seasonality. Fourth, competitive pricing data - what are nearby listings charging? And fifth, actual amenity quality, not just count - a pool matters more than extra coat hangers."

**Q: How do you prevent overfitting?**
A: "Excellent technical question. We used cross-validation with 5 folds and 3 repetitions, which means each model was tested on data it hadn't seen during training. We also held out 30% of data as a final test set that we only used once for final evaluation. The fact that our cross-validation accuracy and test accuracy were similar suggests we didn't overfit."

**Q: Did you consider the business impact of different types of errors?**
A: "That's insightful. Classifying a Premium property as Mid-Market could mean underpricing and losing revenue, while classifying Mid-Market as Premium might mean no bookings. In future work, we could use a cost-sensitive learning approach where different misclassifications have different penalties. For now, we optimized for overall accuracy."

---

## TIMING BREAKDOWN

- Slides 1-3 (Introduction): 1 min 10 sec
- Slides 4-9 (Data & EDA): 2 min 30 sec
- Slides 10-13 (Methods & Results): 2 min 30 sec
- Slides 14-16 (Limitations & Insights): 1 min 50 sec
- Slides 17-18 (Recommendations & Conclusion): 1 min 20 sec
- Slide 19 (Thank You): 15 sec

**Total: ~7 minutes**

---

## PRESENTATION TIPS

1. **Speak to the audience, not the slides** - Make eye contact
2. **Use the visualizations** - Point to specific parts of charts
3. **Vary your pace** - Slow down for key insights, speed up for transitions
4. **Show enthusiasm** - You spent weeks on this, let your passion show!
5. **Practice transitions** - Know what's coming next
6. **Use pauses** - After important points, pause for 2-3 seconds
7. **Tell a story** - You're not just presenting data, you're telling a journey

Good luck with your presentation!
