"""Assignment 1 - Part 2: verification of probability answers."""
from math import log2

print("Q1")
pA, pB = 0.4, 0.3
pAB = pA * pB
print(f"  (a) P(A∩B) = {pA}*{pB} = {pAB:.2f}")
print(f"  (b) P(A∪B) = {pA}+{pB}-{pAB:.2f} = {pA + pB - pAB:.2f}")

print("Q2")
pA, pB, pA_given_B = 0.5, 0.4, 0.7
print(f"  P(A|B) = {pA_given_B} vs P(A) = {pA} -> independent? {pA_given_B == pA}")
print(f"  P(A∩B) = {pA_given_B*pB:.2f} vs P(A)P(B) = {pA*pB:.2f}")

print("Q3")
pA, pB_A, pB_notA = 0.6, 0.5, 0.2
pB = pB_A * pA + pB_notA * (1 - pA)
print(f"  P(B) = {pB:.2f}, P(A|B) = {pB_A*pA/pB:.4f}")

print("Q4")
pD, sens, spec = 0.02, 0.95, 0.90
pPos = sens * pD + (1 - spec) * (1 - pD)
print(f"  P(+) = {pPos:.4f}, P(D|+) = {sens*pD/pPos:.4f}")

print("Q5")
xs, ps = [85, 90, 95, 100], [0.375, 0.375, 0.125, 0.125]
EX = sum(x * p for x, p in zip(xs, ps))
EX2 = sum(x**2 * p for x, p in zip(xs, ps))
print(f"  (a) E[X] = {EX}")
print(f"  (b) E[X^2] = {EX2}, Var(X) = {EX2 - EX**2}")
sample = [85, 90, 85, 95, 90, 85, 100, 90]
mean = sum(sample) / len(sample)
print(f"  (c) sample mean = {sum(sample)}/{len(sample)} = {mean} (diff from E[X] = {mean-EX})")

print("Q6")
probs = [0.4, 0.3, 0.2, 0.1]
H = -sum(p * log2(p) for p in probs)
print(f"  (a) H(X) = {H:.4f} bits")
print(f"  (b) uniform H = log2(4) = {log2(4):.1f} bits")
