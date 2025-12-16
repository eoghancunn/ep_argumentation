import re 
import requests
from bs4 import BeautifulSoup


pattern = re.compile(r'^\s*(\d+)\.\s*(.+)$')
def parse_agenda_title(text):
    match = pattern.match(text.strip())
    if match:
        return {
            "chapter_num": match.group(1),
            "title": match.group(2).strip()
        }
    return None

def scrape_session_html(url):

    resp = requests.get(url)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, 'html.parser')

    # The CRE content appears inside a main div with class 'text' or similar - let's just use body for now

    agenda_items = []

    # Find all agenda item title tables by searching for td with class 'doc_title'
    title_cells = soup.find_all('td', class_='doc_title')

    for title_cell in title_cells:
        title_text = title_cell.get_text(strip=True)

        agenda_metadata = parse_agenda_title(title_text)
        if agenda_metadata is None:
            continue
        # The agenda item table is usually the parent table of this cell
        agenda_table = title_cell.find_parent('table')

        # Interventions come immediately after this agenda_table
        interventions = []

        # Next siblings after the agenda_table (sometimes several tables)
        next_sibs = []
        sib = agenda_table.find_next_sibling('table')
        while sib:
            # If sib contains a 'td' with class 'doc_title', means next agenda item started, break
            if sib.find('td', class_='doc_title'):
                break
            next_sibs.append(sib)
            sib = sib.find_next_sibling('table')

        # Parse interventions inside next_sibs
        for table in next_sibs:
            # Speaker info usually in first 'td' or within nested tables
            speaker = None
            text = None

            # Try to find speaker name inside bold or spans with class 'bold'
            speaker_tag = table.find(lambda tag: tag.name in ['b', 'span'] and ('bold' in tag.get('class', []) or tag.name == 'b'))
            if speaker_tag:
                speaker = speaker_tag.get_text(strip=True)

            #find all urls 
            imgs = table.find_all('img', attrs={'alt': 'MPphoto'})
            for img in imgs:
                if 'mepphoto' in img.get('src'):
                    mep_id = img.get('src').split('/')[-1].split('.')[0]
                else:
                    mep_id = None

            # Find speech text: often in a <p> with class 'contents' or 'italic', or just text in a td
            text_tag = table.find_all('p', class_='contents')
            if text_tag:
                text = "\n".join(x.get_text(separator=' ', strip=True) for x in text_tag)
            else:
                # fallback: collect all text in second td
                tds = table.find_all('td')
                if len(tds) > 2:
                    text = tds[2].get_text(separator=' ', strip=True)
                elif len(tds) > 1:
                    text = tds[1].get_text(separator=' ', strip=True)

            identifier = table.find_previous_sibling('a', attrs={'name': True})
            if identifier:
                identifier = identifier.get('name')
            if speaker and text:
                interventions.append({
                    'speaker': speaker,
                    'mep_id': mep_id,
                    'text': text,
                    'identifier': identifier
                })

        if agenda_metadata:
            agenda_items.append({
                'chapter_num': agenda_metadata.get('chapter_num', ''),
                'title': agenda_metadata.get('title', ''),
                'interventions': interventions
            })
    
    return agenda_items

def get_multilingual_transcript(url):

    
    ml_transcript = {}
    try:
        english_transcripts = scrape_session_html(url)
    except Exception as e:
        print(f"Error scraping debate {url}: {e}")
        return None

    chapter_num = url.split('ITEM-')[1][1:3]
    for english_transcript in english_transcripts:
        if english_transcript['chapter_num'] == chapter_num:
            break

    title = english_transcript['title']

    language_pattern = r"[−–—]\s\((\w{2})\)"
    lang_set = set()
    
    for intervention in english_transcript['interventions']:
        match = re.search(language_pattern, intervention['text'])
        lang = match.group(1) if match else "EN"  # Default to EN if no language code found
        lang_set.add(lang)
        meta = {
            "EN": intervention['text'],
            "speaker": intervention['speaker'],
            "lang": lang,
            "title": title
                }
        ml_transcript[intervention['identifier']] = meta
    
    for lang in lang_set:
        lang_url = url.replace('EN', lang)
        lang_transcripts = scrape_session_html(lang_url)
        for lang_transcript in lang_transcripts:
            if lang_transcript['chapter_num'] == chapter_num:
                break
        for intervention in lang_transcript['interventions']:
            if intervention['identifier'] in ml_transcript:
                ml_transcript[intervention['identifier']][lang] = intervention['text']

    
    transcript = {}
    for k,v in ml_transcript.items():
        transcript[k] = {'speaker': v['speaker'], 'english': v['EN'],
         'lang': v['lang'], 'original': v[v['lang']], 'agenda_item': v['title']}

    return transcript