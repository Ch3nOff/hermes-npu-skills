"""
script_check.py — measure whether Chinese text is Simplified or Traditional.

Used to vet datasets before trusting them, and to detect the failure mode where
Whisper transcribes Traditional-Chinese audio correctly but emits Simplified
characters. Raw CER would call that an error when the words are right.

Method: count characters that exist in only one of the two orthographies. Chars
shared by both (的, 我, 是, ...) carry no signal and are ignored.
"""

# Left = Simplified-only form, right = Traditional-only form.
PAIRS = [
    ("国", "國"), ("学", "學"), ("说", "說"), ("这", "這"), ("来", "來"), ("时", "時"),
    ("过", "過"), ("发", "發"), ("现", "現"), ("东", "東"), ("车", "車"), ("门", "門"),
    ("马", "馬"), ("鸟", "鳥"), ("风", "風"), ("气", "氣"), ("长", "長"), ("书", "書"),
    ("买", "買"), ("卖", "賣"), ("爱", "愛"), ("汉", "漢"), ("语", "語"), ("们", "們"),
    ("个", "個"), ("为", "為"), ("与", "與"), ("难", "難"), ("题", "題"), ("请", "請"),
    ("确", "確"), ("译", "譯"), ("认", "認"), ("对", "對"), ("会", "會"), ("样", "樣"),
    ("变", "變"), ("务", "務"), ("经", "經"), ("济", "濟"), ("应", "應"), ("从", "從"),
    ("众", "眾"), ("体", "體"), ("产", "產"), ("业", "業"), ("动", "動"), ("单", "單"),
    ("间", "間"), ("问", "問"), ("总", "總"), ("统", "統"), ("织", "織"), ("红", "紅"),
    ("绿", "綠"), ("给", "給"), ("续", "續"), ("网", "網"), ("传", "傳"), ("视", "視"),
    ("头", "頭"), ("岁", "歲"), ("点", "點"), ("边", "邊"), ("还", "還"), ("儿", "兒"),
    ("觉", "覺"), ("讲", "講"), ("话", "話"), ("开", "開"), ("关", "關"), ("电", "電"),
    ("师", "師"), ("结", "結"), ("构", "構"), ("台", "臺"), ("湾", "灣"), ("务", "務"),
    ("龙", "龍"), ("岛", "島"), ("图", "圖"), ("专", "專"), ("术", "術"), ("质", "質"),
    ("显", "顯"), ("举", "舉"), ("权", "權"), ("义", "義"), ("论", "論"), ("证", "證"),
]
SIMP_ONLY = {a for a, b in PAIRS}
TRAD_ONLY = {b for a, b in PAIRS}


def script_verdict(texts):
    """Return {'simp': n, 'trad': n, 'chars': n, 'verdict': str} for a list of strings."""
    if isinstance(texts, str):
        texts = [texts]
    simp = trad = chars = 0
    for t in texts:
        chars += len(t)
        for ch in t:
            if ch in SIMP_ONLY:
                simp += 1
            elif ch in TRAD_ONLY:
                trad += 1
    if trad > simp * 3 and trad > 0:
        v = "TRADITIONAL"
    elif simp > trad * 3 and simp > 0:
        v = "SIMPLIFIED"
    elif simp == 0 and trad == 0:
        v = "NO_SIGNAL"
    else:
        v = "MIXED"
    return {"simp": simp, "trad": trad, "chars": chars, "verdict": v}


def to_traditional(text):
    """Convert Simplified -> Traditional, character by character.

    Deliberately NOT opencc's phrase-level s2twp: that rewrites text that is ALREADY
    Traditional. Measured failure: 說明了 -> 說明瞭 ("了" misread as "瞭" by the
    phrase table), which added a fresh error to a clip whisper-medium had transcribed
    perfectly. Character-level s2t is idempotent — converting Traditional input
    returns it unchanged — so it can be applied to mixed output safely.
    """
    try:
        from opencc import OpenCC
        cc = OpenCC("s2t")
        out = []
        for ch in text:
            conv = cc.convert(ch)
            out.append(conv if len(conv) == 1 else ch)
        return "".join(out)
    except Exception:
        # fall back to the char table so scoring still works without opencc
        table = {a: b for a, b in PAIRS}
        return "".join(table.get(c, c) for c in text)


if __name__ == "__main__":
    import json
    import sys
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    refs = [d["reference"] for d in data]
    print(json.dumps(script_verdict(refs), ensure_ascii=False, indent=2))
