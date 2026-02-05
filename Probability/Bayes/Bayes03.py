"""
Задача 3. Спам‑фильтрация
P_spam = 0.2 (доля спама).
P_word_spam = 0.4 (слово «приз» в спаме).
P_word_not_spam = 0.05 (слово «приз» в не‑спаме).

Найти: P_spam_given_word — вероятность спама, если слово есть.
"""
P_spam = 0.2
P_not_spam = 0.8
P_word_spam = 0.4
P_word_not_spam = 0.05

P_word = P_word_spam * P_spam + P_word_not_spam * P_not_spam
P_spam_given_word = (P_word_spam * P_spam) / P_word

print(f"P(Spam | Word) = {P_spam_given_word:.4f}")  # ≈ 0.6667

