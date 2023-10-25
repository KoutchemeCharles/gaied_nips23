from difflib import SequenceMatcher

def convertToNumber (s):
    return int.from_bytes(s.encode(), 'little')

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_closest_in_dict(d, key):
    if key in d:
        return d[key]
    
    return d.get(key, d[min(d.keys(), key=lambda k: similar(k, key))])
