import re


def clean_sub_categ(paragraph):
    paragraph = re.sub(r'[\{\(\[][^{}\(\)\[\]]*?[\}\)\]]|[\r\n]|(\[\w+\s*\[\w+\]\])', '', paragraph[:110])
        
    sub_category = re.search(r'#(.*?)(N\.|—N\.|Adv\.|\.—)', paragraph)
    sub_category = sub_category.group(1) if sub_category else ''
    sub_category = str.strip(sub_category)
        
    sub_category = re.sub(r'&c|\s\.\s|\[\w*\]|\(\w*\)|\{\w*\}|[\(\)\[\]\{\}\|]|[.,]$', '', sub_category)
    sub_category = re.sub(r'\s\.', ' ', sub_category)
    sub_category = str.strip(sub_category)
    sub_category = re.sub(r'\.\s+', '. ', sub_category)
    sub_category = re.sub(r'\.\s\d|\d+\.\s|\d+\w\.\s', '', sub_category)
    sub_category = re.sub(r'\.', ',', sub_category)
    return sub_category.capitalize()


def tokenize_text(text):
    return tokenizer(text, truncation=True, padding='max_length', max_length=128)


def clustering():
    pass


def cosine_similarity_report():
    pass

