# Algorithms and LeetCode (Python, C++)

## Coding Interview Process

1. communication (clarify problems, data type/range, constraints)
2. explain solution and thoughts during writing codes
3. ask questions

## Real Problems

### find local maxima with a decreasing window.

**Question.** Given integer array `rawData` and integer `localArea`, return the sorted 0-based indices `i` for which values decrease **strictly** as one moves away from `rawData[i]` up to `localArea` positions to the left and right. At an array boundary, use the available positions only.

**Answer.** Precompute the consecutive decreasing-run length in each direction. An index qualifies if both runs cover the required number of available neighbors. O(n) time and O(n) space.

```python
def solution(rawData, localArea):
    n = len(rawData)
    if n == 0:
        return []

    left = [0] * n
    right = [0] * n

    # Number of strictly decreasing steps while moving left from i.
    for i in range(1, n):
        if rawData[i] > rawData[i - 1]:
            left[i] = left[i - 1] + 1

    # Number of strictly decreasing steps while moving right from i.
    for i in range(n - 2, -1, -1):
        if rawData[i] > rawData[i + 1]:
            right[i] = right[i + 1] + 1

    answer = []
    for i in range(n):
        required_left = min(localArea, i)
        required_right = min(localArea, n - 1 - i)
        if left[i] >= required_left and right[i] >= required_right:
            answer.append(i)
    return answer


if __name__ == "__main__":
    print(solution([2, 10, 6, 3, 1, 2, 4, 6, 9, 5, 2], 2))
    # [1, 8]
```

**Edge cases.** Equal adjacent values break strict decrease. At index 0 or `n−1`, the missing side is vacuously satisfied, so an endpoint may qualify if its existing side decreases. If `localArea == 0`, every index qualifies.

### add parentheses to minimize an arithmetic expression.

**Question.** Given a string such as `"112+422"` consisting of two positive integers with no zero digits, insert exactly one pair of parentheses. The `+` must be inside, with at least one digit of each operand inside. Adjacent numeric factors multiply. Return the smallest possible integer value.

**Answer.** Enumerate where the opening parenthesis falls in the left operand and where the closing parenthesis falls in the right operand. Any prefix/suffix outside parentheses is a multiplicative factor; absent factors count as 1. Precompute the numeric values of all relevant substrings, then evaluate O(L×R) candidate expressions; overall O(n²) time for a length-n string (under the usual arithmetic-operation model).

```python
def solution(expr):
    left, right = expr.split("+")
    best = float("inf")

    left_outside = [int(left[:i]) if i else 1 for i in range(len(left))]
    left_inside = [int(left[i:]) for i in range(len(left))]
    right_inside = [int(right[:j]) for j in range(1, len(right) + 1)]
    right_outside = [
        int(right[j:]) if j < len(right) else 1
        for j in range(1, len(right) + 1)
    ]

    for i in range(len(left)):
        for j in range(len(right)):
            value = (
                left_outside[i] *
                (left_inside[i] + right_inside[j]) *
                right_outside[j]
            )
            best = min(best, value)

    return int(best)


if __name__ == "__main__":
    assert solution("112+422") == 108
    assert solution("55+55") == 110
    print(solution("112+422"), solution("55+55"))
```

**Correct minimal placement for `112+422`:** `1(12+42)2 = 108`. The assertion is for that placement. The screenshot also gives `"55+55" → 110` via `(55+55)`.

## High Frequency Problem Type

- Recursion and Backtracking

- Trees

- Binary Search Trees

- Graphs

- Breadth-First Search

- Depth-First Search

- Heap and Priority Queue

## Should Know but usually not appear independently

- Two Pointers

- Binary Search

- Sorting

- Sliding Window

- Stack and Monotonic Stack

- Queue and Deque

- Linked List

- Arrays and Strings

- Hash Maps and Hash Sets

## Rare

- Topological Sort (heard Google like this, but I did not meet, usually hard)

- Dynamic Programming

- Bit Manipulation

- Greedy Algorithms

## After Finishing Coding

- Dry Run (Common Edge Cases)

- Time and Space Complexity

- Practice Problem Tracker
