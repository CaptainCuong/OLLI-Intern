import torch
from underthesea import pos_tag

from .token import (POS_TAG, abb_prior, abbreviation_list, flt, hard, num,
                    num_mag, num_mag_level, post_num, pronoun, spoken_alpb,
                    sub_num_mag, unit)


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

def clean_num_abb(words, embedding_model, label_model, seq_len):
    words = words.lower().split()

    # Convert and pad for string
    pad_string = [torch.tensor(embedding_model.get_vector(word).reshape(1,-1)) for word in words]
    for i in range(len(words), seq_len):
        pad_string.append(torch.zeros((1, 400)))
    pad_string = torch.cat(pad_string, dim=0).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # Get labels
    label = label_model(pad_string.unsqueeze(dim=0)) # Shape (batch_size=1, sequence_len=seq_len, dim = 4)
    label = label.argmax(dim=2)[0]
    i = 0
    rt = []
    rt_prclabel = []
    print('Detected number/abbreviation phrases:\n')
    while i < len(words):
        # Num phrase must start with 'num'
        if (words[i] in num) and (words[i] not in hard or words[i] in hard and label[i].item() == 1):
            st = i
            while i < len(words) and (words[i] not in hard or words[i] in hard and label[i].item() == 1) and (words[i] in num | post_num | flt | num_mag or words[i] in sub_num_mag):
                i += 1
            lt = i
            if lt-st >= 2:
                print(' '.join(words[st:lt]),'\n')
                rt.append(lit2num(words[st:lt]))
                rt_prclabel.extend(['num' for _ in range(lt-st)])
            elif i > 0 and words[i-1] in pronoun:
                rt.append(words[st])
                rt_prclabel.append('unknown')
            elif label[st].item() == 1:
                rt.append(lit2num(words[st:lt]))
                rt_prclabel.extend(['num' for _ in range(lt-st)])
            else:
                rt.append(words[st:lt])
                rt_prclabel.extend('unknown' for _ in range(lt-st))
            i -= 1
        elif words[i] in post_num:
            print(str(words[i]),'\n')
            rt.append(str(post_num[words[i]]))
            rt_prclabel.append('num')
        elif words[i] in spoken_alpb:
            st = i
            while i < len(words) and words[i] in spoken_alpb:
                i += 1
            lt = i
            if lt-st >= 2:
                # Ex: 'công ty ép pi ti'
                if words[st] in ['ti', 'ty'] and st > 0 and words[st-1] == 'công':
                    rt.append(words[st])
                    rt_prclabel.append('unknown')
                    st += 1
                    rt.append(merge_abb(words[st:lt]))
                    rt_prclabel.extend(['abb' for _ in range(lt-st)])
                # Ex: 'tập đoàn ép pi ti', 'đại học ép pi ti'
                elif any(prior_exist>=0 for prior_exist in [' '.join(words[:st]+words[lt:]).find(x) for x in abb_prior]):
                    rt.append(merge_abb(words[st:lt]))
                    rt_prclabel.extend(['abb' for _ in range(lt-st)])
                # Ex: 'sơn tùng em ti pi'
                else:
                    # Abbreviation exists in database
                    mw = merge_abb(words[st:lt])
                    if mw in abbreviation_list[spoken_alpb[words[st]]]:
                        rt.append(mw)
                        rt_prclabel.extend(['abb' for _ in range(lt-st)])
                    # Use model
                    else:
                        for j in range(st, lt):
                            if label[j] != 3:
                                if j > st:
                                    rt.append(merge_abb(words[st:j]))
                                    rt_prclabel.extend(['abb' for _ in range(j-st)])
                                rt.append(words[j])
                                rt_prclabel.append('unknown')
                                st = j+1
                        if lt > st:
                            rt.append(merge_abb(words[st:lt]))
                            rt_prclabel.extend(['abb' for _ in range(lt-st)])
            else:
                rt.append(words[i-1])
                rt_prclabel.append('unknown')
            i -= 1
        else:
            rt.append(words[i])
            rt_prclabel.append('unknown')
        i += 1

    retlb = []
    for lb in label[:len(words)]:
        if lb == 1:
            retlb.append('num')
        elif lb == 2:
            retlb.append('unknown')
        elif lb == 3:
            retlb.append('abb')
        else:
            retlb.append('padding')
    #assert len(words) == len(rt_prclabel), 'Len of label is not consistent with that of sentence. '+' '.join(rt)+'\nLen of sentence:%d'%(len(words))+'\nLen of label:%d'%(len(rt_prclabel))
    #return ' '.join(rt), ' '.join(rt_prclabel)
    assert len(words) == len(retlb), 'Len of label is not consistent with that of sentence. '+' '.join(rt)+'\nLen of sentence:%d'%(len(words))+'\nLen of label:%d'%(len(retlb))
    return ' '.join(rt), ' '.join(retlb)

def clean_abb(words, embedding_model, label_model, seq_len):
    words = words.lower().split()

    # Convert and pad for string
    pad_string = []
    for word in words:
        try:
            pad_string.append(torch.tensor(embedding_model.get_vector(word).reshape(1,-1)))
        except:
            pad_string.append(torch.tensor([[0.0625 for _ in range(400)]]))
    for i in range(len(words), seq_len):
        pad_string.append(torch.zeros((1, 400)))
    pad_string = torch.cat(pad_string, dim=0).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # Get pos_tag
    # POS_tag = [POS_TAG[tag] for prs, tag in pos_tag(' '.join(words)) for _ in range(len(prs.split()))]
    POS_tag = []
    for prs, tag in pos_tag(' '.join(words)):
        for _ in range(len(prs.split())):
            try:
                POS_tag.append(POS_TAG[tag])
            except:
                POS_tag.append(0)
    for i in range(len(words), seq_len):
        POS_tag.append(0)
    POS_tag = torch.tensor(POS_tag)
    # Get labels
    label = label_model(pad_string.unsqueeze(dim=0), POS_tag.unsqueeze(dim=0)) # Shape (batch_size=1, sequence_len=seq_len, dim = 4)
    label = label.argmax(dim=2)[0]
    i = 0
    rt = []
    rt_prclabel = []
    print('Detected number/abbreviation phrases:\n')
    while i < len(words):
        # Num phrase must start with 'num'
        if (words[i] in num) and (words[i] not in hard or words[i] in hard and label[i].item() == 1):
            st = i
            while i < len(words) and (words[i] not in hard or words[i] in hard and label[i].item() == 1) and (words[i] in num | post_num | flt | num_mag or words[i] in sub_num_mag):
                i += 1
            lt = i
            if lt-st >= 2:
                print(' '.join(words[st:lt]),'\n')
                rt.append(lit2num(words[st:lt]))
                rt_prclabel.extend(['num' for _ in range(lt-st)])
            elif i > 0 and words[i-1] in pronoun:
                rt.append(words[st])
                rt_prclabel.append('unknown')
            elif label[st].item() == 1:
                rt.append(lit2num(words[st:lt]))
                rt_prclabel.extend(['num' for _ in range(lt-st)])
            else:
                rt.extend(words[st:lt])
                rt_prclabel.extend('unknown' for _ in range(lt-st))
            i -= 1
        elif words[i] in post_num:
            print(str(words[i]),'\n')
            rt.append(str(post_num[words[i]]))
            rt_prclabel.append('num')

        ####################### ABBREVIATION #######################
        # elif words[i] in spoken_alpb:
        #     st = i
        #     while i < len(words) and words[i] in spoken_alpb:
        #         i += 1
        #     lt = i
        #     if lt-st >= 3:
        #         # Ex: 'công ty ép pi ti'
        #         if words[st] in ['ti', 'ty'] and st > 0 and words[st-1] == 'công':
        #             rt.append(words[st])
        #             rt_prclabel.append('unknown')
        #             st += 1
        #             rt.append(merge_abb(words[st:lt]))
        #             rt_prclabel.extend(['abb' for _ in range(lt-st)])
        #         else:
        #             # Abbreviation exists in database
        #             mw = merge_abb(words[st:lt])
        #             rt.append(mw)
        #             rt_prclabel.extend(['abb' for _ in range(lt-st)])
        #     elif lt-st == 2 and ' '.join(words[st:lt]) != 'anh em':
        #         mw = merge_abb(words[st:lt])
        #         rt.append(mw)
        #         rt_prclabel.extend(['abb' for _ in range(lt-st)])
        #     else:
        #         rt.append(words[i-1])
        #         rt_prclabel.append('unknown')
        #     i -= 1
        else:
            rt.append(words[i])
            rt_prclabel.append('unknown')
        i += 1

    retlb = []
    for lb in label[:len(words)]:
        if lb == 1:
            retlb.append('num')
        elif lb == 2:
            retlb.append('unknown')
        elif lb == 3:
            retlb.append('abb')
        else:
            retlb.append('padding')
    assert len(words) == len(rt_prclabel), 'Len of label is not consistent with that of sentence. '+' '.join(rt)+'\nLen of sentence:%d'%(len(words))+'\nLen of label:%d'%(len(rt_prclabel))
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
            # print(stk)
            # print(stk[-1]+float(flt[words[i]]))
            # print(stk[-1][1])
            stk[-1] = (stk[-1][0]+float(flt[words[i]]), stk[-1][1])
        i += 1
        # print(stk)
    return str(sum([num[0] for num in stk]))

def merge_abb(words):
    return ''.join(spoken_alpb[word] for word in words)
 
'''
số lượng từ pâ >=3, phía trước ko có 'công ty' -> ghép lại

từ 2 trở lên ngoại trừ 'anh em'
'''
