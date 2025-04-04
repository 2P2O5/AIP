import os
punctuation = ['!', '?', '…', ",", ".", "'", '-']
pu_symbols = punctuation + ["SP", "UNK"]
pad = '_'
zh_symbols = ['E', 'En', 'a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng', 'er', 'f', 'g', 'h',
       'i', 'i0', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'ir', 'iu', 'j', 'k', 'l', 'm', 'n', 'o',
       'ong',
       'ou', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai', 'uan', 'uang', 'ui', 'un', 'uo', 'v', 'van', 've', 'vn',
       'w', 'x', 'y', 'z', 'zh',
        "AA", "EE", "OO"]
num_zh_tones = 6
ja_symbols = ['I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky',
              'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh', 't', 'ts', 'u', 'V', 'w', 'y', 'z']
num_ja_tones = 1
en_symbols = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy',
              'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's',
              'sh', 't', 'th', 'uh', 'uw', 'V', 'w', 'y', 'z', 'zh']
num_en_tones = 4
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
symbols = [pad] + normal_symbols + pu_symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]
num_tones = num_zh_tones + num_ja_tones + num_en_tones

language_id_map = {
    'ZH': 0,
    "JA": 1,
    "EN": 2
}
num_languages = len(language_id_map.keys())
language_tone_start_map = {
    'ZH': 0,
    "JA": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones
}
current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {line.split("\t")[0]: line.strip().split("\t")[1] for line in
                        open(os.path.join(current_file_path, 'opencpop-strict.txt')).readlines()}
rep_map = {
    '：': ',',
    '；': ',',
    '，': ',',
    '。': '.',
    '！': '!',
    '？': '?',
    '\n': '.',
    "·": ",",
    '、': ",",
    '...': '…',
    '$': '.',
    '“': "'",
    '”': "'",
    '‘': "'",
    '’': "'",
    '（': "'",
    '）': "'",
    '(': "'",
    ')': "'",
    '《': "'",
    '》': "'",
    '【': "'",
    '】': "'",
    '[': "'",
    ']': "'",
    '—': "-",
    '～': "-",
    '~': "-",
    '「': "'",
    '」': "'",
}


symbol_to_id = {s: i for i, s in enumerate(symbols)}