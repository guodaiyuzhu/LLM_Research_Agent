import spacy

nlp = spacy.load('zh_core_web_trf')

def get_child(token):
    children = list(token.children)
    lst = []
    if len(children) > 0:
        for i in children:
            lst.extend(get_child(i))
    lst.append(token)
    return lst

def extract_words(sent):
    doc = nlp(sent)
    ent_lst = []
    for ent in doc.ents:
        if ent.text not in ent_lst and ent.label_ not in ('CARDINAL', 'QUANTITY'):
            ent_lst.append(ent.text)
    nlp.tokenizer.pkuseg_update_user_dict(ent_lst)
    doc = nlp(sent)
    key_words = []
    for token in doc:
        if token.dep_ == 'ROOT':
            children = list(token.children)
            children.append(token)
            children.sort(key=lambda x: x.idx)
            for i in children:
                if i.dep_ in ('cc', 'punct', 'case', 'advmod', 'conj'):
                    continue
                elif i == token:
                    key_words.append(i)
                else:
                    key_words.extend(get_child(i))
    res_lst = []
    words_lst = []
    tag = ''
    for token in key_words:
        new_tag = token.tag_
        if new_tag != tag:
            if len(words_lst) > 0:
                res_lst.append(''.join(words_lst))
            words_lst = []
        if token.tag_ in ('NT', 'NR', 'NN'):
            words_lst.append(token.text)
        tag = new_tag
    else:
        if token.tag_ in ('NT', 'NR', 'NN'):
            words_lst.append(token.text)
    if len(words_lst) > 0:
        res_lst.append(''.join(words_lst))
    return ','.join(res_lst)
