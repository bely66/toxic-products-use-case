import re
import string

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations
arabic_stop_words = ['من', 'في', 'على', 'و', 'فى', 'عن', 'هو', 'التي', 'الذي', 'كما', 'لم', 'لن', 'مع', 'هذا', 'وأن', 'ثم', 'أن', 'هذه', 'قد', 'ما', 'كان', 'لكن', 'علي', 'أنا', 'ذلك', 'إلى', 'أو', 'كل', 'هل', 'يمكن', 'بما', 'أي', 'هي']


def clean_text(text):
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    #remove nan
    text = re.sub('\bnan\b', '', text)
    text = re.sub(r'\b[nN][aA][nN]\b', '', text)
    # remove urls
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    
    #remove &nbsp;
    text = re.sub('&nbsp;', ' ', text)
    
    # remove html tages
    text = re.sub('<.*?>+', ' ', text)
    
    # Removing @user
    text = re.sub(r'@[^\s]+', ' ', text)
    
    # remove #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    
    # remove punctuation
    text = re.sub('[%s]' % re.escape(punctuations_list), ' ', text)
    
    # remove new line
    text = re.sub('\n', ' ', text)
    
    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Replace 3 or more consecutive letters by 2 letter.
    text = re.sub(sequencePattern, seqReplacePattern, text)
    
    # Removing English words and numbers and make right strip
    text = re.sub(r'\s*[0-9]+\b', '' , text).rstrip()
    
    # lower case
    text = text.lower()
    
    return text

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_emojis(text): 
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)

def remove_stop_words(text, stop_words):
    # Basic preprocessing to keep only Arabic letters and remove stop words
    text = re.sub("[^\u0600-\u06FF\s]", " ", text)  # Keep Arabic characters only
    words = text.split()
    words_filtered = [word for word in words if word not in stop_words]
    return words_filtered

def preprocess_data(text):
    
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    
    # Normalize the text 
    text = normalize_arabic(text)

    # Remove emojis
    text = remove_emojis(text)

    text = ' '.join(remove_stop_words(text, arabic_stop_words))

    return text