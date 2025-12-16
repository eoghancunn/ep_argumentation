from utils.scraping import get_multilingual_transcript
import pandas as pd 
import re 
import json
import argparse
import os
from pathlib import Path
import shutil
from tqdm import tqdm
from utils.motion_scraper import EuropeanParliamentScraper
# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'



def extract_report_urls(t):
    """
    Extract report ID from introduction. report ids are of the form [( A6-0086/2006 ), ( A5-0121/2005 )]
    """
    matches = re.findall(r'\(\s*(A\d+-\d+/\d+)\s*\)', t)
    urls = []
    for m in matches:
        term = m[1]
        year = m.split('/')[1]
        identifier = m.split('-')[1].split('/')[0]
        urls.append(f'https://www.europarl.europa.eu/doceo/document/A-{term}-{year}-{identifier}_EN.html')
    return urls

def get_debate_urls(term, meta_data_file):

    # load meta data
    with open(meta_data_file, 'r') as infile:
        data = json.load(infile)

    # store activities in dataframe 
    cres = []
    for d in data:
        if 'CRE' in d:
            cres.extend(d['CRE'])
    cre_df = pd.DataFrame(cres)
    term = cre_df[cre_df['term'] == term]

    debate_urls = term[term['title'].str.contains(r'\(debate\)')]['url'].map(lambda x: x.split('&')[0])
    return sorted(set(debate_urls))

def scrape_report(url, overwrite = False):
    """
    scrape a report and store each intervention
    """
    scraper = EuropeanParliamentScraper()
    try:
        d = scraper.scrape_report(url)
    except Exception as e:
        return None
    report_meta = {}
    report_meta['url'] = url
    report_meta['report_id'] = url.split('document/')[1].split('_')[0]
    report_meta['paragraphs'] = d['paragraphs']
    return report_meta

def scrape_debate(url, overwrite = False):
    """
    scrape a debate and store each interventio
    """
    
    debate_id = "-".join(url.split('+')[1:-3])
    debate_dir = DATA_DIR / 'debates' / debate_id 

    if not debate_dir.exists():
        (debate_dir / 'interventions').mkdir(parents=True, exist_ok=True)
        (debate_dir / 'report_statements').mkdir(parents=True, exist_ok=True)
    elif not overwrite: 
        return None

    try: 
        data = get_multilingual_transcript(url)
    except Exception as e:
        print(f"Error scraping debate {url}: {e}")
        return None
    
    if data is None:
        return None
    
    first_intervention = list(data.values())[0]

    # store debate metadata
    with open(debate_dir / 'metadata.json', 'w+', encoding='utf-8') as outfile:
        json.dump({"title": first_intervention['agenda_item'], "url": url}, outfile, ensure_ascii=False)

    # store interventions
    for intervention_id, meta in data.items():
        meta['debate_id'] = debate_id
        meta['intervention_id'] = intervention_id
        intervention_file = debate_dir / 'interventions' / f'{intervention_id}.json'
        with open(intervention_file, 'w+', encoding='utf-8') as outfile:
            json.dump(meta, outfile, ensure_ascii=False)

    # store report metadata
    report_urls = extract_report_urls(first_intervention['english'])
    for report_url in report_urls:
        report_meta = scrape_report(report_url)
        if report_meta is None:
            continue
        report_file = debate_dir / 'report_statements'/ f'{report_meta['report_id']}.json'
        with open(report_file, 'w+', encoding='utf-8') as outfile:
            json.dump(report_meta, outfile, ensure_ascii=False, indent=4)
    return data
    

def remove_empty_debate_dirs():
    """
    cleanup the data directory
    """
    for debate_dir in (DATA_DIR / 'debates').iterdir():
        if debate_dir.is_dir():
            report_statements_dir = debate_dir / 'report_statements'
            if report_statements_dir.exists() and not any(report_statements_dir.iterdir()):
                shutil.rmtree(debate_dir)

def main(term, meta_data_file, num_debates = 100, overwrite = False, cleanup = False):
    debate_urls = get_debate_urls(term, meta_data_file)
    for url in tqdm(debate_urls[:num_debates]):
        scrape_debate(url, overwrite)
    if cleanup:
        remove_empty_debate_dirs()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--term', type=int, default=6)
    parser.add_argument('--meta-data-file', type=str, 
                       default=str(DATA_DIR / 'ep_mep_activities.json'))
    parser.add_argument('--num-debates', type=int, default=100)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()
    main(args.term, args.meta_data_file, args.num_debates, args.overwrite, args.cleanup)