"""Cleaning of the CUB `description` column (ten captions in one newline-separated string)."""
import re

# Hand-built from the train vocabulary. Real bird terms (wingbar, eyering, rectrices,
# superciliary, undertail, tarsuses, ...) are deliberately left alone.
TYPO_FIXES = {
    "retrices": "rectrices", "retricies": "rectrices", "retricles": "rectrices",
    "rectices": "rectrices", "rectricles": "rectrices", "recticles": "rectrices",
    "retrice": "rectrice",
    # second pass: spelling typos
    "smal": "small", "blck": "black", "balck": "black", "beack": "beak",
    "bakc": "back", "bak": "back", "brwon": "brown", "bornw": "brown",
    "corwn": "crown", "cown": "crown", "plummage": "plumage",
    "adn": "and", "amd": "and", "hwite": "white", "ehite": "white", "whie": "white", "shite": "white",
    "yelow": "yellow", "yello": "yellow", "yekkiw": "yellow", "gren": "green", "greay": "gray", "bluie": "blue",
    "brid": "bird", "bir": "bird", "bord": "bird", "burd": "bird",
    "bellt": "belly", "bellly": "belly", "gelly": "belly", "bellty": "belly", "bellyl": "belly", "eblly": "belly",
    "breats": "breast", "breat": "breast", "ehad": "head", "billl": "bill",
    "wiht": "with", "wih": "with", "wth": "with", "wtih": "with", "wiith": "with",
    "thraot": "throat", "thorat": "throat", "eyebro": "eyebrow", "eyelings": "eyerings", "eyeings": "eyerings",
    "tarus": "tarsus", "tarsas": "tarsus", "tarsis": "tarsus",
    "covarts": "coverts", "corvets": "coverts", "secondaires": "secondaries",
    "trinagular": "triangular", "camoflauged": "camouflaged", "camoflague": "camouflage",
    "irridescent": "iridescent", "flourescent": "fluorescent", "metalic": "metallic",
    "spikey": "spiky", "patterend": "patterned", "fluffly": "fluffy", "hgas": "has",
    # run-together words
    "withwhite": "with white", "withblack": "with black", "thisis": "this is",
    "andthe": "and the", "areblack": "are black", "greywith": "grey with",
    "supercillary": "superciliary", "supercilliary": "superciliary", "supercilary": "superciliary",
    "superciliaries": "superciliary",
    "bil": "bill", "beek": "beak", "whtie": "white", "bely": "belly",
    "iwth": "with", "witha": "with a", "redish": "reddish", "poitned": "pointed", "larsus": "tarsus",
}

# Contractions after the apostrophe is removed ("that's" -> "thats"). "its" is left alone: captions use
# "it's" for both "it is" and the possessive, and "its" reads correctly either way.
CONTRACTIONS = {
    "thats": "that is", "theres": "there is", "whos": "who is", "hes": "he is", "theyre": "they are",
    "doesnt": "does not", "dont": "do not", "cant": "can not",
}

# Written with a hyphen or a space, but one word: "multi-colored" -> "multicolored", "white-ish" -> "whiteish".
PREFIXES = {"multi"}
SUFFIXES = {"ish"}

MIN_WORDS = 3  # shorter lines ("this") describe nothing; the shortest real captions are like "a red bird"


def _join_affixes(words):
    joined = []
    for word in words:
        if joined and (word in SUFFIXES or joined[-1] in PREFIXES):
            joined[-1] += word
        else:
            joined.append(word)
    return joined


def clean_caption(text):
    """Normalize one caption: lowercase, fix punctuation/whitespace, correct known typos."""
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"['`]", "", text)           # bird's -> birds
    text = re.sub(r"[^a-z\s]", " ", text)      # punctuation, digits and corrupted characters -> space
    words = _join_affixes(text.split())         # split also collapses whitespace
    return " ".join(CONTRACTIONS.get(w, TYPO_FIXES.get(w, w)) for w in words)


def clean_description(description):
    """Split a raw description on newlines and return its cleaned, unique captions of at least MIN_WORDS words."""
    captions = [clean_caption(c) for c in description.split("\n")]
    captions = [c for c in captions if len(c.split()) >= MIN_WORDS]  # drop empty and too short
    return list(dict.fromkeys(captions))        # drop duplicates, keep first copy and order


def clean_descriptions(descriptions):
    """Clean a list of raw descriptions (one per image)."""
    return [clean_description(d) for d in descriptions]
