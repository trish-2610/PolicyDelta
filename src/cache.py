"""
Step 7: Cache (remember answers, so the same question is answered instantly the second time)

One small interface, two ways to store:
    InMemoryCache : a Python dictionary inside the running program (used locally, nothing to install)
    RedisCache    : a Redis server (used later when deployed, Phase 2). Switched on by REDIS_URL in .env

The rest of the code only calls cache.get(question) and cache.set(question, answer),
so switching from memory to Redis later changes nothing else.

The cache key is the MASKED question (personal data is never stored in the cache),
in lower case with single spaces, so "What is X?" and "what is  x?" hit the same entry.
"""

## importing required libraries
import json
import os
import re
import time

## settings
TTL_SECONDS = 24 * 60 * 60    ## an answer is kept for 1 day, then asked again (documents may change)


def make_key(question):
    """lower case + single spaces, so small differences in typing still hit the cache"""

    return "answer:" + re.sub(r"\s+", " ", question).strip().lower()


class InMemoryCache:
    """Cache inside the program (lost when the program stops)"""

    def __init__(self):
        self.store = {}

    def get(self, question):
        item = self.store.get(make_key(question))
        if item is None:
            return None
        saved_at, answer = item
        if time.time() - saved_at > TTL_SECONDS:      ## too old: forget it
            del self.store[make_key(question)]
            return None
        return answer

    def set(self, question, answer):
        self.store[make_key(question)] = (time.time(), answer)


class RedisCache:
    """Cache in a Redis server (kept even if the program restarts, shared by many copies of the app)"""

    def __init__(self, url):
        import redis    ## only needed when Redis is used
        self.client = redis.Redis.from_url(url)

    def get(self, question):
        value = self.client.get(make_key(question))
        return json.loads(value) if value else None

    def set(self, question, answer):
        self.client.setex(make_key(question), TTL_SECONDS, json.dumps(answer))


def get_cache():
    """Redis if REDIS_URL is set in .env, otherwise in-memory"""

    url = os.getenv("REDIS_URL")
    if url:
        return RedisCache(url)
    return InMemoryCache()