# Cross-evaluation report (macro F1)

Rows = training labels, columns = evaluation labels. On-diagonal cells measure fit to a label standard; off-diagonal cells measure transfer between the gold (human) and silver (LLM) standards.

## English

**TF-IDF + LR**

|  | gold test | silver test |
|---|---|---|
| gold-trained | 0.707 | 0.557 |
| silver-trained | 0.592 | 0.794 |

**XLM-R**

|  | gold test | silver test |
|---|---|---|
| gold-trained | 0.768 | 0.568 |
| silver-trained | 0.589 | 0.874 |

## Tamil

**TF-IDF + LR**

|  | gold test | silver test |
|---|---|---|
| gold-trained | 0.630 | 0.656 |
| silver-trained | 0.560 | 0.780 |

**XLM-R**

|  | gold test | silver test |
|---|---|---|
| gold-trained | 0.614 | 0.752 |
| silver-trained | 0.583 | 0.841 |

## Malayalam

**TF-IDF + LR**

|  | gold test | silver test |
|---|---|---|
| gold-trained | 0.763 | 0.591 |
| silver-trained | 0.655 | 0.730 |

**XLM-R**

|  | gold test | silver test |
|---|---|---|
| gold-trained | 0.777 | 0.682 |
| silver-trained | 0.714 | 0.865 |
