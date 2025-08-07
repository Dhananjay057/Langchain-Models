import numpy as np
def run_consistency_evaluation():
    # Your code here that returns a consistency score (float)
    # For example:
    score = ...  # Replace with actual scoring logic
    return score
# Run the code 10 times and collect scores
scores = []
for _ in range(10):
    score = run_consistency_evaluation()
    scores.append(score)
# Compute the mean score
mean_score = np.mean(scores)
# Print results
print("All Consistency Scores:", scores)
print("Mean Consistency Score:", mean_score)