import torch

from .token import (flt, hard, num, num_mag, num_mag_level, post_num,
                    sub_num_mag, unit, spoken_alpb)


'''
'clean_num' is the main function to convert literal to sequence
'''

def convert_to_litnum(str_in):
    words = str_in.split()
    swap_flt(words)
    # print(words)
    words = lit2num(words)
    return ' '.join(words)

def token_label(ws):
    words = ws.copy()
    for i,w in enumerate(words):
        if w in num:
            words[i] = 'num'
        elif w in post_num:
            words[i] = 'post_num'
        elif w in flt:
            words[i] = 'flt'
        elif w in unit:
            words[i] = 'unit'
        elif w in num_mag:
            words[i] = 'num_mag'
        elif w in sub_num_mag:
            words[i] = 'sub_num_mag'
        else:
            words[i] = 'unknown'
    return words

def swap_flt(words):
    for i in swap_flt_ind(token_label(words)):
        temp = words[i]
        words[i] = words[i-1]
        words[i-1] = temp

def swap_flt_ind(words):
    for i in range(1, len(words)):
        if words[i] == 'flt':
            yield i

def clean_num_abb(words, embedding_model, label_model):
    words = words.split()

    # Convert and pad for string
    pad_string = [torch.tensor(embedding_model.get_vector(word).reshape(1,-1)) for word in words]
    for i in range(len(words), 13):
        pad_string.append(torch.zeros((1, 400)))
    pad_string = torch.cat(pad_string, dim=0)

    # Get labels
    label = label_model(pad_string.unsqueeze(dim=0))
    label = label.argmax(dim=2)[0]
    # print(label)
    i = 0
    rt = []
    print('Detected number/abbreviation phrases:\n')
    while i < len(words):
        # Num phrase must start with 'num'
        if (words[i] in num) and (words[i] not in hard or words[i] in hard and label[i].item() == 1):
            st = i
            while i < len(words) and (words[i] not in hard or words[i] in hard and label[i].item() == 1) and (words[i] in num | post_num | flt | num_mag or words[i] in sub_num_mag):
                i += 1
            lt = i
            print(' '.join(words[st:lt]),'\n')
            rt.append(lit2num(words[st:lt]))
            i -= 1
        elif words[i] in post_num:
            print(str(words[i]),'\n')
            rt.append(str(post_num[words[i]]))
        elif words[i] in spoken_alpb:
            st = i
            while i < len(words) and words[i] in spoken_alpb:
                i += 1
            lt = i
            if lt-st >= 2:
                rt.append(merge_abb(words[st:lt]))
            i -= 1
        else:
            rt.append(words[i])
        i += 1
    retlb = []
    for lb in label[:len(words)]:
        if lb == 1:
            retlb.append('num')
        elif lb == 2:
            retlb.append('unknown')
        elif lb == 3:
            retlb.append('abb')
    # ['num' if lb == 1 else 'unknown' for lb in label[:len(words)]]
    return ' '.join(rt), ' '.join(retlb)

def lit2num(words):
    i = 0
    # A stack contains (value, level), level: level of magnitude
    stk = [(0,0)]
    # words = words.split()
    swap_flt(words)
    max_num_mag = 0
    while i < len(words):
        if words[i] in num and words[i] != 'mười':
            num_con = []
            level = 0
            while i < len(words) and words[i] in num|post_num:
                num_con.append(num[words[i]] if words[i] in num else post_num[words[i]])
                i += 1
                level += 1
            i -= 1
            stk.append((float(''.join(num_con)), level))
        elif words[i] in post_num:
            stk.append((float(post_num[words[i]]),0))
        elif words[i] in num and words[i] == 'mười':
            if i+1 < len(words) and words[i] in num|post_num:
                dummy = float(num[words[i+1]]) if words[i+1] in num else float(post_num[words[i+1]])
                stk.append((10+float(dummy), 1))
            else:
                stk.append((float(10),2))
            i += 1
        elif words[i] in num_mag:
            if max_num_mag < num_mag_level[words[i]]:
                stk = [(sum([num[0] for num in stk])*float(num_mag[words[i]]), num_mag_level[words[i]])]
                max_num_mag = num_mag_level[words[i]]
            else:
                j = len(stk)-1
                while j > -1:
                    if stk[j][1] > num_mag_level[words[i]]:
                        break;
                    j -= 1
                j += 1
                stk[j:] = [(sum([num[0] for num in stk[j:]])*float(num_mag[words[i]]), num_mag_level[words[i]])]
        elif words[i] in flt:
            stk[-1] = (stk[-1]+float(flt[words[i]]), stk[-1][1])
        i += 1
        # print(stk)
    return str(sum([num[0] for num in stk]))

def merge_abb(words):
    return ''.join(spoken_alpb[word] for word in words)