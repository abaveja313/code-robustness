# What data do we want to collect?
**Which experiments to run?**
1. How does pass@1 ratio change with the number of model parameters? (expect -> more parameters -> better performance)
2. How does pass@1 ratio change with the sampling temperature? (expect -> higher temperature -> worse performance)
3. How do pass@k ratio change with the value of k? (expect -> higher k -> better performance)
4. How does pass@1 ratio change with additional code fine-tuning (expect -> better performance)?
5. How does pass@1 ratio change with polyglot vs monoglot training data? (expect -> polyglot -> better performance)

**For each model:**
- Quantify mutation difficulty by levenshtein distance
- Calculate support for each mutation type, filter for those with >= 5 support examples
- Understand training and model architecture
- Calculate pass rate for each canonical example
- Understand bad syntax rate for each mutation type
- Understand average pass@ on original normalized by position in the sequence (inverse weighted by length)
- Visual examples
- Compute benchmark score for robustness:
  - Average pass@1 ratio, normalized by mutation difficulty
  - Average pass@1 ratio by category, weighted average by mutation difficulty


**For each experiment**
- Heatmap/barplot comparing pass@k ratios by mutation type
- Filter to only statistically significant results
- Similar categories, see visual difference -> mode
- Mann Whitney U Test, p-value.
- Attempt to draw some conclusion on why?