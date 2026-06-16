# TextMe Mood Classifier — Evaluation Report

**Checkpoint epoch:** 22

---

## Overall Metrics

| Metric | Score |
|---|---|
| Macro F1 | 0.5622 |
| Macro Precision | 0.5760 |
| Macro Recall | 0.5716 |
| Target F1 | 0.6500 |
| Target met | No |

---

## Per-Class F1 Scores

| Mood Class | F1 Score |
|---|---|
| casual | 0.6536 |
| emotional | 0.4130 |
| excited | 0.5307 |
| romantic | 0.6667 |
| angry | 0.5037 |
| anxious | 0.6424 |
| grateful | 0.7314 |
| apology | 0.5789 |
| supportive | 0.2933 |
| curious | 0.4969 |
| funny | 0.6737 |

---

## Full Classification Report

```
              precision    recall  f1-score   support

      casual       0.59      0.73      0.65      1740
   emotional       0.60      0.32      0.41       282
     excited       0.50      0.57      0.53       397
    romantic       0.61      0.74      0.67       227
       angry       0.53      0.48      0.50       709
     anxious       0.65      0.64      0.64        83
    grateful       0.77      0.69      0.73       798
     apology       0.51      0.67      0.58        49
  supportive       0.42      0.23      0.29       390
     curious       0.58      0.44      0.50       550
       funny       0.59      0.79      0.67       202

    accuracy                           0.59      5427
   macro avg       0.58      0.57      0.56      5427
weighted avg       0.59      0.59      0.58      5427

```

---

## Notes

- Best performing class  : **grateful** (F1: 0.7314)
- Worst performing class : **supportive** (F1: 0.2933)
- Classes present in test: 11 out of 18
- Total test samples     : 5427