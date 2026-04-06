# Day 22 — LeetCode #208 Implement Trie
# Difficulty: Medium ⚡
# Approach: Tree with 26 children per node
# Time: O(m) per op | Space: O(m*n)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Test
trie = Trie()
trie.insert("databricks")
trie.insert("delta")
trie.insert("spark")
print(trie.search("delta"))       # True
print(trie.search("delt"))        # False
print(trie.startsWith("data"))    # True
print(trie.startsWith("mlflow"))  # False


# Day 22 — LeetCode #139 Word Break
# Difficulty: Medium ⚡
# Approach: DP + Trie optimization
# Time: O(n²) | Space: O(n)

def wordBreak(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    # dp[i] = True if s[:i] can be segmented
    dp = [False] * (n + 1)
    dp[0] = True  # empty string always valid

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]

# Test cases
print(wordBreak("leetcode",
    ["leet","code"]))           # True
print(wordBreak("applepenapple",
    ["apple","pen"]))           # True
print(wordBreak("catsandog",
    ["cats","dog","sand","and","cat"]))  # False
