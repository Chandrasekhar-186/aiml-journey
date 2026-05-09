# Can you write LRU Cache from memory in 10 min?
# Test yourself — timer starts NOW!

from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)


# Optimized: Bidirectional BFS
# Much faster for long word ladders!

def ladderLength_bidir(beginWord, endWord,
                        wordList):
    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    # Two frontiers
    front = {beginWord}
    back = {endWord}
    steps = 1

    while front and back:
        steps += 1
        # Always expand smaller frontier
        if len(front) > len(back):
            front, back = back, front

        next_front = set()
        for word in front:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = (word[:i] + c +
                                word[i+1:])
                    if new_word in back:
                        return steps
                    if new_word in word_set:
                        next_front.add(new_word)
                        word_set.discard(new_word)
        front = next_front

    return 0

print(ladderLength_bidir("hit","cog",
    ["hot","dot","dog","lot","log","cog"]))  # 5
