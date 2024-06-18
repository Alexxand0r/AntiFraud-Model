import urllib.parse
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import joblib
import spacy
import tldextract
import pandas as pd
import requests
import re
import torch
from bs4 import BeautifulSoup
from sklearn import preprocessing
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import asyncio
import websockets
import json
#Загрузка модели TF-IDF и классификатора текстовых данных
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
classifier = joblib.load('classifier.pkl')
#Количество ссылок на сайте
def hyper_links(url, num):
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all('a')
    count = 0
    for link in links:
        href = link.get('href')
        if href:
            parsed_href = urllib.parse.urlparse(href)
            if num==1: #Общее количество ссылок
                if parsed_href.netloc:
                    count += 1
            elif num==2: #Количество ссылок на этот же ресурс
                if parsed_href.netloc and not parsed_href.netloc.__contains__(urllib.parse.urlparse(url).netloc):
                    count += 1
            elif num==3: #Количество внешних ссылок
                if parsed_href.netloc and parsed_href.netloc.__contains__(urllib.parse.urlparse(url).netloc):
                    count += 1
    return count
#Поиск подозрительных слов в ссылке
def phishing_hints(url_path):
    count = 0
    HINTS = ['wp', 'login', 'includes', 'admin', 'content', 'site', 'images', 'js', 'alibaba', 'css', 'myaccount', 'dropbox', 'themes', 'plugins', 'signin', 'view']
    for hint in HINTS:
        count += url_path.lower().count(hint)
    return count
#Извлечение слов из ссылки
def words_raw_extraction(domain, subdomain, path):
    w_domain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", domain.lower())
    w_subdomain = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", subdomain.lower())
    w_path = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", path.lower())
    raw_words = w_domain + w_path + w_subdomain
    w_host = w_domain + w_subdomain
    return list(filter(None,raw_words)), list(filter(None,w_host)), list(filter(None,w_path))
#Количество перенаправлений по ссылке
def nb_external(page, domain):
    count = 0
    if len(page.history) == 0:
        return 0
    else:
        for i, response in enumerate(page.history,1):
            if domain.lower() not in response.url.lower():
                count+=1
        return count
#Наличие формы подтверждения отправки E-mail
def submitting_to_email(Form):
    for form in Form['internals'] + Form['externals']:
        if "mailto:" in form or "mail()" in form:
            return 1
        else:
            return 0
    return 0
#Наличие формы для авторизации
def login_form_exist(Form):
    p = re.compile('([a-zA-Z0-9\_])+.php')
    if len(Form['externals'])>0 or len(Form['null'])>0:
        return 1
    for form in Form['internals']+Form['externals']:
        if p.match(form) != None :
            return 1
    return 0
#Получении информации о CSS
def CSS_get(hostname, domain, soup):
    CSS = {'internals': [], 'externals': [], 'null': []}
    for link in soup.find_all('link', rel='stylesheet'):
        dots = [x.start(0) for x in re.finditer('\.', link['href'])]
        if hostname in link['href'] or domain in link['href'] or len(dots) == 1 or not link['href'].startswith('http'):
            if not link['href'].startswith('http'):
                if not link['href'].startswith('/'):
                    CSS['internals'].append(hostname+'/'+link['href'])
                elif link['href'] in ("", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                                      "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"):
                    CSS['null'].append(link['href'])
                else:
                    CSS['internals'].append(hostname+link['href'])
        else:
            CSS['externals'].append(link['href'])
    return CSS
#Получении информации о формах на сайте
def Form_get(hostname, domain, soup):
    Form = {'internals': [], 'externals': [], 'null': []}
    for form in soup.findAll('form', action=True):
        dots = [x.start(0) for x in re.finditer('\.', form['action'])]
        if hostname in form['action'] or domain in form['action'] or len(dots) == 1 or not form['action'].startswith('http'):
            if not form['action'].startswith('http'):
                if not form['action'].startswith('/'):
                    Form['internals'].append(hostname+'/'+form['action'])
                elif form['action'] in ("", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                                        "#content", "javascript::void(0)", "javascript::void(0);",
                                        "javascript::;", "javascript") or form['action'] == 'about:blank':
                    Form['null'].append(form['action'])
                else:
                    Form['internals'].append(hostname+form['action'])
        else:
            Form['externals'].append(form['action'])
    return Form
#Получении информации о гиперссылках на сайте
def Anchor_get(hostname, domain, soup):
    Anchor = {'safe': [], 'unsafe': []}
    for href in soup.find_all('a', href=True):
        dots = [x.start(0) for x in re.finditer('\.', href['href'])]
        if hostname in href['href'] or domain in href['href'] or len(dots) == 1 or not href['href'].startswith('http'):
            if "#" in href['href'] or "javascript" in href['href'].lower() or "mailto" in href['href'].lower():
                Anchor['unsafe'].append(href['href'])
        else:
            Anchor['safe'].append(href['href'])
    return Anchor
#Получение информации о фото на сайте
def Media_get(hostname, domain, soup):
    Media = {'internals': [], 'externals': [], 'null': []}
    if not soup:
        return None, None
    if not soup.find_all('img', src=True):
        for img in soup.find_all('img', src=True):
            dots = [x.start(0) for x in re.finditer('\.', img['src'])]
            if hostname in img['src'] or domain in img['src'] or len(dots) == 1 or not img['src'].startswith('http'):
                if not img['src'].startswith('http'):
                    if not img['src'].startswith('/'):
                        Media['internals'].append(hostname+'/'+img['src'])
                    elif img['src'] in ("", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                                        "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"):
                        Media['null'].append(img['src'])
                    else:
                        Media['internals'].append(hostname+img['src'])
            else:
                Media['externals'].append(img['src'])
    return Media, soup.title.string
#Извлечение всех численных признаков по ссылке
def get_url_info(url):
    try:
        urllib.parse.urlparse(url) # Проверка существования URL
    except Exception as e:
        print("Ошибка: Недействительный URL.")
        return None
    try:
        # Разбор URL и задание переменных
        parsed_url = urllib.parse.urlparse(url)
        domainList = {'.com', '.org', '.net', '.ru','.рф','.by','.бел','.kz','ua'}
        domain = tldextract.extract(url).domain
        subdomain = tldextract.extract(url).subdomain
        urlPath = parsed_url.path + parsed_url.query
        hostname_length = len(parsed_url.hostname)
        try:
            page = requests.get(url, timeout=5) #Попытка подключения к ресурсу
        except:
            parsed = urllib.parse.urlparse(url)
            url = parsed.scheme+'://'+parsed.netloc
            if not parsed.netloc.startswith('www'):
                url = parsed.scheme+'://www.'+parsed.netloc
                try:
                    page = requests.get(url, timeout=50)
                    Content = page.content
                    soup = BeautifulSoup(Content, 'html.parser', from_encoding='iso-8859-1')
                except:
                    page = None
                    Content = None
                    soup = None
        #Извлечение признаков по ссылке
        words_raw, hostRaw, pathRaw = words_raw_extraction(domain, subdomain, urlPath)
        try:
            CSS = CSS_get(Content, parsed_url.hostname, domain, soup)
        except:
            CSS = {'internals': [], 'externals': [], 'null': []}
        try:
            Anchor = Anchor_get(Content, parsed_url.hostname, domain, soup)
        except:
            Anchor = {'safe': [], 'unsafe': []}
        try:
            Form = Form_get(parsed_url.hostname, domain, soup)
        except:
            Form = {'internals': [], 'externals': [], 'null': []}
        try:
            Media, title = Media_get(Content, parsed_url.hostname, domain, soup)
        except:
            Media, title = {'internals': [], 'externals': [], 'null': []}, "None"
        nb_at=url.count('@')
        nb_qm=url.count('?')
        nb_and=url.count('&')
        nb_or=url.count('|')
        nb_tilde=url.count('~')
        nb_percent=url.count('%')
        nb_star=url.count('*')
        nb_colon=url.count(';')
        nb_comma=url.count(',')
        nb_semicolumn=url.count(';')
        nb_underscore=url.count('_')
        nb_dollar=url.count('$')
        nb_space=url.count('%20')
        https_token = 1 if "https" not in url else 0
        ratio_digits_url = int(sum(c.isdigit() for c in url) / len(url)) if len(url) > 0 else 0
        port = 1 if parsed_url.port is not None else 0
        tld_in_path = 1 if any(urlPath.__contains__(tld) for tld in domainList) else 0
        tld_in_subdomain = sum(1 for tld in domainList if parsed_url.netloc.__contains__(tld))
        abnormal_subdomain = 1 if re.search('(http[s]?:\/\/(w[w]?|\d))([w]?(\d|-))',url) else 0
        nb_subdomains = parsed_url.netloc.count('.') - 1
        prefix_suffix = 1 if re.search('(https?:\/\/[^\-]+-[^\-]+\/)',url) else 0
        random_domain = 1 if ((not any(parsed_url.netloc.endswith(tld) for tld in domainList))) else 0
        try:
            nb_redirection = 1 if len(page.history) else 0
        except:
            nb_redirection = 0
        nb_external_redirection = nb_external(page,domain)
        length_words_raw = len(words_raw)
        avg_words_raw = 0 if len(words_raw) == 0 else int((sum(len(word) for word in words_raw) / len(words_raw)))
        avg_word_host = 0 if len(hostRaw) == 0 else int((sum(len(word) for word in hostRaw) / len(hostRaw)))
        avg_word_path = 0 if len(pathRaw) == 0 else int((sum(len(word) for word in pathRaw) / len(pathRaw)))
        phish_hints = phishing_hints(urlPath)
        brands = open("allbrands.txt", "r")
        domain_in_brand = 1 if domain in brands else 0
        brand_in_subdomain = 1 if subdomain in brands else 0
        brand_in_path = 1 if urlPath in brands else 0
        suspecious_tld = 1 if domain in ('fit','tk', 'gp', 'ga', 'work', 'ml', 'date', 'wang', 'men', 'icu', 'online', 'click',
                                         'country', 'stream', 'download', 'xin', 'racing', 'jetzt', 'ren', 'mom', 'party', 'review',
                                         'trade', 'accountants', 'science', 'work', 'ninja', 'xyz', 'faith', 'zip', 'cricket', 'win',
                                         'accountant', 'realtor', 'top', 'christmas', 'gdn',
                                         'link', 'asia', 'club', 'la', 'ae', 'exposed', 'pe', 'go.id', 'rs', 'k12.pa.us', 'or.kr',
                                         'ce.ke', 'audio', 'gob.pe', 'gov.az', 'website', 'bj', 'mx', 'media', 'sa.gov.au') else 0
        nb_hyperlinks = hyper_links(url, 1)
        ratio_intHyperlinks = hyper_links(url, 3)
        ratio_extHyperlinks = hyper_links(url, 2)
        ratioLinks = 1 if ratio_extHyperlinks == 0 else int (ratio_intHyperlinks/ratio_extHyperlinks)
        nb_extCSS = len(CSS['externals'])
        login_form = login_form_exist(Form)
        links_in_tags = 0 if nb_hyperlinks == 0 else int((ratio_intHyperlinks/nb_hyperlinks * 100))
        submit_email = submitting_to_email(Form)
        ratio_intMedia = 0 if (len(Media['internals']) + len(Media['externals'])) == 0 else \
            int((len(Media['internals'])/(len(Media['internals']) + len(Media['externals'])) * 100))
        ratio_extMedia = 0 if (len(Media['internals']) + len(Media['externals'])) == 0 else \
            int(len(Media['externals'])/(len(Media['internals']) + len(Media['externals'])) * 100)
        sfh = 1 if len(Form['null'])>0 else 0
        safe_anchor = 0 if (len(Anchor['unsafe']) + len(Anchor['safe'])) == 0 else \
            int(len(Anchor['unsafe'])/(len(Anchor['unsafe']) + len(Anchor['safe'])) * 100)
        try:
            onmouseover = 1 if 'onmouseover="window.status=' in str(Content).lower().replace(" ","") else 0 #Наличие обработчика
            right_clic = 1 if re.findall("event.button ?== ?2", str(Content).lower()) else 0 #Наличие обработчика
        except:
            onmouseover = 0
            right_clic = 0
        empty_title = 0 if title else 1
        domain_in_title = 0 if domain.lower() in title.lower() else 1
        # Вывод результатов в виде словаря
        return {'url': [url], 'length_url': [len(url)], 'length_hostname': [hostname_length], 'nb_at': [nb_at], 'nb_qm': [nb_qm],
                'nb_and': [nb_and], 'nb_or': [nb_or], 'nb_underscore': [nb_underscore], 'nb_tilde': [nb_tilde], 'nb_percent': [nb_percent],
                'nb_star':  [nb_star], 'nb_colon': [nb_colon], 'nb_comma': [nb_comma], 'nb_semicolumn': [nb_semicolumn],
                'nb_dollar': [nb_dollar], 'nb_space':  [nb_space], 'https_token': [https_token], 'ratio_digits_url': [ratio_digits_url],
                'port': [port], 'tld_in_path': [tld_in_path], 'tld_in_subdomain': [tld_in_subdomain], 'abnormal_subdomain': [abnormal_subdomain],
                'nb_subdomains': [nb_subdomains], 'prefix_suffix': [prefix_suffix], 'random_domain': [random_domain],
                'nb_redirection': [nb_redirection], 'nb_external_redirection': nb_external_redirection, 'length_words_raw': [length_words_raw],
                'avg_words_raw': [avg_words_raw], 'avg_word_host': [avg_word_host], 'avg_word_path': [avg_word_path], 'phish_hints': [phish_hints],
                'domain_in_brand': [domain_in_brand], 'brand_in_subdomain': [brand_in_subdomain], 'brand_in_path': [brand_in_path],
                'suspecious_tld': [suspecious_tld], 'nb_hyperlinks': [nb_hyperlinks],
                'ratio_intHyperlinks': [ratio_intHyperlinks], 'ratio_extHyperlinks': [ratio_extHyperlinks], 'ratioLinks': [ratioLinks],
                'nb_extCSS': [nb_extCSS], 'login_form': [login_form], 'links_in_tags': [links_in_tags], 'submit_email': [submit_email],
                'ratio_intMedia': [ratio_intMedia], 'ratio_extMedia': [ratio_extMedia], 'sfh': [sfh], 'safe_anchor': [safe_anchor],
                'onmouseover': [onmouseover], 'right_clic': [right_clic], 'empty_title': [empty_title], 'domain_in_title': [domain_in_title]}
    except Exception as e:
        #Вывод ошибки, если не получилось извлечь данные
        print("Ошибка при обработке URL:", e)
        return None
#Объявление модели
class ChurnModel(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(ChurnModel, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden1)
        self.layer_2 = nn.Linear(hidden1, hidden2)
        self.layer_out = nn.Linear(hidden2, output_dim)
        self.conv1 = nn.Conv3d(input_dim, output_dim, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hidden1)
        self.batchnorm2 = nn.BatchNorm1d(hidden2)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        return x
#Функция классификации числовой информации
def load_and_predict(model_path, parameters):
    df_data = pd.DataFrame(parameters)
    numeric_columns = df_data.select_dtypes(include=['number']).columns
    X = df_data[numeric_columns]
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler.fit_transform(X.values.reshape(51,-1))
    X_scaled = X_scaled.reshape(-1,51)
    X_tensor = torch.from_numpy(X_scaled).float()
    dataset = TensorDataset(X_tensor)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    n_input_dim = X.shape[1]
    n_hidden1 = 400
    n_hidden2 = 200
    n_output = 1
    model = ChurnModel(n_input_dim, n_hidden1, n_hidden2, n_output)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for xb in data_loader:
            xb = xb[0]  # Retrieve the tensor from the dataset
            y_pred = model(xb)
    return y_pred
#Задание некоторых обработчиков текста
nlp = spacy.load('en_core_web_sm')
stop_words_en = set(stopwords.words('english'))
stop_words_ru = set(stopwords.words('russian'))
stemmer_ru = SnowballStemmer("russian")
#Функция предварительной обработки текста с ресурса
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    processed_tokens = []
    for token in tokens:
        if token in stop_words_en or token in stop_words_ru:
            continue
        if re.match(r'[a-zA-Z]', token):
            # Лемматизация для английских слов
            doc = nlp(token)
            token = doc[0].lemma_
        else:
            # Стемминг для русских слов
            token = stemmer_ru.stem(token)
        processed_tokens.append(token)
    return ' '.join(processed_tokens)
#Функция классификации текста
def classify_text(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    prob = classifier.predict_proba(text_tfidf)[0][1]
    return prob
#Функция сохранения файла с текстом
def save_text_to_file(text, filename, labelsfile, result):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(text + '\n')
    with open(labelsfile, 'a', encoding='utf-8') as file:
        file.write(result + '\n')
#Обработка поступающих данных
async def process_data(data):
    parsed_data = json.loads(data)
    url = parsed_data.get('url')
    raw_text = parsed_data.get('text')
    text = preprocess_text(raw_text) #Получение текстовых признаков
    print(url)
    response_data = {
        "status": "unsuccessful",
        "url": url
    }
    for i in range (1,2):
        try:
            data = get_url_info(url) #Получение численных признаков
        except:
            print("Ошибка обработки данных")
            continue
        if not data or not text:
            print("Данные пустые")
            continue
        #Задание изначальных коэффициентов
        tf_idf_coef = 0.5
        NN_coef = 0.5
        try:
            probabilities = load_and_predict('model.pth', data) #Классификация числовых данных
        except:
            count = 0
            for d in data.keys():
                if not data[d]:
                    data[d].append(0)
                else:
                    count += 1
            probabilities = 0
            if count == 0:
                NN_coef = 0
                tf_idf_coef = 1
            else:
                NN_coef = 0.5*(count/51)
                tf_idf_coef = 1 - NN_coef
        print("Вероятность NN: ", round(probabilities[0][0].item()*100,2))
        try:
            prob = classify_text(text) #Классификация текстовых данных
            print("Вероятность TF-IDF: ", round(prob*100,2))
        except:
            tf_idf_coef = 0
            NN_coef = 1
        if not prob == None and not probabilities == None: #Если числовые и текстовые вероятности удалось посчитать
            result = tf_idf_coef * prob+ NN_coef * probabilities
            print("Вероятность того, что сайт мошеннический: ", round(result[0][0].item()*100,2), "%")
        else:
            print("Не удалось классифицировать")
            continue
        result = 'legitimate' if result < 0.4 else 'phishing' #При результате меньшем, чем 0.4, сайт объявляется легитимным
        data['status']=result
        df = pd.DataFrame(data)
        #Сохранение полученных данных в журнал
        save_text_to_file(text, "TF-IDF data.txt", "TF-IDF data labels.txt", '0' if result == 'legitimate' else '1')
        df.to_csv('url_data1.csv', index=False, mode='a', header=False)
        #Создание ответа пользователю
        response_data = {
            "status": "success",
            "message": f"{result}",
            "url": url
        }
    return json.dumps(response_data)
#Запуск обработчика
async def handler(websocket):
    async for message in websocket:
        print(f"Received message")
        # Передача данных в функцию обработки
        response = await process_data(message)
        #Отправка ответного сообщения
        try:
            await websocket.send(response)
            print("Отправлено")
        except:
            print("Ошибка отправки")

async def main():
    async with websockets.serve(handler, "localhost", 8080):
        print("WebSocket server is running on ws://localhost:8080")
        # Запуск бесконечного цикла для ожидания событий
        await asyncio.Future()
#Запуск процесса
asyncio.run(main())