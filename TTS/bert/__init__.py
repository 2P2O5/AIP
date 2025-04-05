import re
import cn2an
from pypinyin import lazy_pinyin, Style
import torch
import os
import jieba_fast.posseg as psg
from transformers import AutoTokenizer, AutoModelForMaskedLM

from ..__symbols__ import *

def _intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def _cleaned_text_to_sequence(cleaned_text, tones, language):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        Returns:
        List of integers corresponding to the symbols in the text
    '''
    phones = [symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def _clean_text(text, _):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = text.replace("嗯", "恩").replace("呣","母")
    pattern = re.compile('|'.join(re.escape(p) for p in rep_map.keys()))
    text = pattern.sub(lambda x: rep_map[x.group()], text)
    norm_text = re.sub(r'[^\u4e00-\u9fa5'+"".join(punctuation)+r']+', '', text)
    pattern = r'(?<=[{0}])\s*'.format(''.join(punctuation))
    sentences = [i for i in re.split(pattern, norm_text) if i.strip()!='']
    phones = []
    tones = []
    word2ph = []
    for seg in sentences:
        seg = re.sub('[a-zA-Z]+', '', seg)
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = seg_cut
        for word, pos in seg_cut:
            if pos == 'eng':
                continue
            sub_initials = []
            sub_finals = []
            orig_initials = lazy_pinyin(
                word, neutral_tone_with_five=True, style=Style.INITIALS)
            orig_finals = lazy_pinyin(
                word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for c, v in zip(orig_initials, orig_finals):
                sub_initials.append(c)
                sub_finals.append(v)
            initials.append(sub_initials)
            finals.append(sub_finals)

        initials = sum(initials, [])
        finals = sum(finals, [])
        
        for c, v in zip(initials, finals):
            raw_pinyin = c+v
            
            
            if c == v:
                assert c in punctuation
                phone = [c]
                tone = '0'
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c+v_without_tone
                assert tone in '12345'
                if c:
                    v_rep_map = {
                        "uei": 'ui',
                        'iou': 'iu',
                        'uen': 'un',
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c+v_rep_map[v_without_tone]
                else:
                    pinyin_rep_map = {
                        'ing': 'ying',
                        'i': 'yi',
                        'in': 'yin',
                        'u': 'wu',
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            'v': 'yu',
                            'e': 'e',
                            'i': 'y',
                            'u': 'w',
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]]+pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                phone = pinyin_to_symbol_map[pinyin].split(' ')
                word2ph.append(len(phone))

            phones += phone
            tones += [int(tone)] * len(phone)
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(norm_text) 
    phones = ['_'] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return norm_text, phones, tones, word2ph


global c_tokenizer, c_model
c_tokenizer = None
c_model = None
global c_DEVICE

def LoadBert(model_dir,device="cpu"):
    global c_tokenizer, c_model
    global c_DEVICE
    c_DEVICE = torch.device(device)
    c_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    c_model = AutoModelForMaskedLM.from_pretrained(model_dir).to(c_DEVICE, dtype=torch.float32)

def get_bert_feature(text, word2ph):
    global c_DEVICE
    inputs = c_tokenizer(text, return_tensors="pt")
    for i in inputs:
        inputs[i] = inputs[i].to(c_DEVICE)
    res = c_model(**inputs, output_hidden_states=True)
    res = torch.cat(res["hidden_states"][-3:-2], -1)[0].to(c_DEVICE)

    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
    return torch.cat(phone_level_feature, dim=0).T

text__lang_bert_func_map = {
    'ZH': get_bert_feature
}

def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = _clean_text(text, language_str)
    phone, tone, language = _cleaned_text_to_sequence(phone, tone, language_str)
    if hps.data.add_blank:
        phone = _intersperse(phone, 0)
        tone = _intersperse(tone, 0)
        language = _intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2

        word2ph[0] += 1

    bert = text__lang_bert_func_map[language_str](norm_text, word2ph)
    del word2ph
    assert bert.shape[-1] == len(phone)
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, phone, tone, language

