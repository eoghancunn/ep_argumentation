import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional


class EuropeanParliamentScraper:
    """Scraper for European Parliament motions and resolutions."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_motion(self, url: str) -> Dict:
        """
        Scrape a European Parliament motion/resolution.
        
        Args:
            url: URL of the motion document
            
        Returns:
            Dictionary containing motion details
        """
        response = self.session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the full text content
        content = soup.get_text(separator='\n', strip=True)


        full_content = self._extract_resolution_content(soup)
        
        # Parse the structured data
        motion_data = {
            'url': url,
            'title': self._extract_title(content),
            'document_reference': self._extract_doc_reference(content, url),
            'agreed_text_url': self.extract_agreed_text_url(soup),
            'date': self._extract_date(content),
            'authors': self._extract_authors(content),
            'group': self._extract_group(content),
            'full_content': full_content['full_content'],
            'citations': full_content['citations'],
            'recitals': full_content['recitals'],
            'paragraphs': full_content['paragraphs'],
            'procedure_reference': self._extract_procedure_ref(content)
        }
        
        return motion_data

    def scrape_agreed_text(self, url: str) -> Dict:
        """
        Scrape a European Parliament agreed text.
        
        Args:
            url: URL of the agreed text document
        """
        response = self.session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the full text content
        content = soup.get_text(separator='\n', strip=True)
        
        full_content = self._extract_resolution_content(soup)
        
        # Parse the structured data
        agreed_text_data = {
            'url': url,
            'title': self._extract_agreed_text_title(soup),
            'document_reference': self._extract_doc_reference(content, url),
            'full_content': full_content['full_content'],
            'citations': full_content['citations'],
            'recitals': full_content['recitals'],
            'paragraphs': full_content['paragraphs'],
            'procedure_reference': self._extract_procedure_ref(content)
        }
        
        return agreed_text_data

    def scrape_report(self, url: str) -> Dict:
        """
        Scrape a European Parliament report.
        
        Args:
            url: URL of the report document
        """
        response = self.session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')

        full_content = self._extract_report_content(soup)
        
        full_content['url'] = url
        full_content['agreed_text_url'] = self.extract_agreed_text_url(soup)

        return full_content
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract the motion title."""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('MOTION FOR A RESOLUTION'):
                return line.strip()
        return None

    def _extract_agreed_text_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the agreed text title."""
        h2 = soup.find('tr', {'class': 'doc_title'})
        if h2:
            return h2.get_text(strip=True)
        return None
    
    def _extract_doc_reference(self, content: str, url: str) -> Optional[str]:
        """Extract document reference (e.g., B8-0880/2015)."""
        if 'RC' in url or 'TA' in url:
            pattern = r'(RC-?)B\d+-\d+/\d+'
        else:
            pattern = r'B\d+-\d+/\d+'
        match = re.search(pattern, content)
        return match.group(0) if match else None
    
    def extract_agreed_text_url(self, soup) -> Optional[str]:
        """Extract agreed text reference (e.g., TA-8-0880/2015)."""
        urls = soup.find_all('a')
        for url in urls:
            if 'TA' in url.get('href', ''):
                return url['href']
        return None
    
    def _extract_date(self, content: str) -> Optional[str]:
        """Extract the date."""
        match = re.search(r'\d{1,2}\.\d{1,2}\.\d{4}', content)
        return match.group(0) if match else None
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extract the list of authors/members."""
        # Find the line with authors (between date and "on behalf of")
        lines = content.split('\n')
        authors = []
        
        for i, line in enumerate(lines):
            # Authors typically appear after the date and before "on behalf of"
            if re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', line):
                # Next non-empty lines should contain authors
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j] and 'on behalf of' in lines[j]:
                        # Authors are in the line before this one
                        author_line = lines[j-1]
                        # Split by comma
                        authors_found = [a.strip() for a in author_line.split(',') if a.strip()]
                        authors.extend(authors_found)
                        break
                break
        
        return authors
    
    def _extract_group(self, content: str) -> Optional[str]:
        """Extract the political group."""
        match = re.search(r'on behalf of the ([^\s]+) Group', content)
        return match.group(1) if match else None
    
    def _extract_procedure_ref(self, content: str) -> Optional[str]:
        """Extract procedure reference."""
        match = re.search(r'\d{4}/\d{4}\(RSP\)', content)
        return match.group(0) if match else None

    def _extract_resolution_content(self, soup: BeautifulSoup) -> str:
        """Extract the main resolution content using paragraph tags."""
        clauses = []
        clause_types = {
            'citation': [],
            'recital': [],
            'paragraph': []
        }
        current_clause = []
        
        # Find all paragraph tags
        for p in soup.find_all('p'):
            # Skip paragraphs that are subheadings (contain bold tags)
            bold_elements = (
                p.find_all('span', class_='bold') + p.find_all('b') + p.find_all('strong') + p.find_all('span', style=lambda x: x and 'font-weight:bold;' in x)
            )
            if bold_elements:
                # Calculate how much of the text is bold
                bold_text_length = sum(len(elem.get_text(strip=True)) for elem in bold_elements)
                total_text_length = len(p.get_text(strip=True))
                
                # If more than 90% is bold, skip this paragraph (it's a subheading)
                if total_text_length > 0 and bold_text_length > total_text_length * 0.9:
                    continue
            
            text = p.get_text(strip=True)
            
            # Skip empty paragraphs and legal notice sections
            if not text or text.startswith('Legal notice'):
                continue
            
            # Check if this starts a new clause
            # Citations start with "–" or "-"
            # Recitals start with a letter followed by "."
            # Paragraphs start with a number followed by "."

            is_new_clause = (
                text.startswith('–') or 
                text.startswith('-') or
                re.match(r'^[A-Z]+\.', text) or  # Letter followed by period and space
                re.match(r'^\d+\.', text)  # Number followed by period and space
            )
            
            if is_new_clause:
                # Save the previous clause if it exists
                if current_clause:
                    if clause_type:
                        clause_types[clause_type].append(' '.join(current_clause)+'\n')
                    clauses.append(' '.join(current_clause)+'\n')

                # Start a new clause
                if text.startswith('–') or text.startswith('-'):
                    clause_type = 'citation'
                elif re.match(r'^[A-Z]+\.', text):
                    clause_type = 'recital'
                elif re.match(r'^\d+\.', text):
                    clause_type = 'paragraph'
                current_clause = [text]

            else:
                # Continue the current clause
                if current_clause:
                    current_clause.append(text)
                else:
                    # This might be introductory text before any clauses
                    clauses.append(text)
        
        # Don't forget the last clause
        if current_clause:
            if clause_type:
                clause_types[clause_type].append(' '.join(current_clause)+'\n')
            clauses.append(' '.join(current_clause)+'\n')
        
        return {'full_content': "\n".join(clauses), 'citations': clause_types['citation'], 'recitals': clause_types['recital'], 'paragraphs': clause_types['paragraph']}

    def _extract_explanatory_statement(self, soup: BeautifulSoup) -> str:
        """Extract the explanatory statement using paragraph tags."""
        clauses = []
        current_clause = []
        explanatory_statement = None

        for h2 in soup.find_all('h2'):
            if "EXPLANATORY STATEMENT" in h2.get_text():
                explanatory_statement = h2
                break

        if explanatory_statement is None:
            return None
        paragraphs = []
        next_element = explanatory_statement.find_next_sibling()
        if next_element.name == 'div':
            next_element = next_element.find()
        while next_element:
            if next_element.name == 'p':
                paragraphs.append(next_element)
            if next_element.name == 'h2':
                break
            next_element = next_element.find_next_sibling()

        for p in paragraphs:
            clauses.append(p.get_text(strip=True))

        return "\n".join(clauses)

    def _extract_report_content(self, soup: BeautifulSoup) -> str:
        """Extract the report content using paragraph tags."""
        clauses = []
        current_clause = []

        # Find h2 containing span with text "MOTION FOR A EUROPEAN PARLIAMENT RESOLUTION"
        for h2 in soup.find_all('h2'):
            if "MOTION FOR A EUROPEAN PARLIAMENT RESOLUTION" in h2.get_text():
                motion_header = h2
                break

        # Get all paragraphs after the motion header
        paragraphs = []
        next_element = motion_header.find_next_sibling()
        if next_element.name == 'div':
            next_element = next_element.find()
        while next_element:
            if next_element.name == 'p':
                paragraphs.append(next_element)
            if next_element.name == 'h2':
                break
            next_element = next_element.find_next_sibling()

        # Convert list of paragraph elements back to BeautifulSoup object
        if not paragraphs:
            return {'resolution': '', 'citations': [], 'recitals': [], 'paragraphs': [], 'explanatory_statement': None}
            
        # Create a new BeautifulSoup object with just the paragraphs
        soup_string = "".join(str(p) for p in paragraphs)
        paragraphs_soup = BeautifulSoup(soup_string, 'html.parser')
    
        resolution = self._extract_resolution_content(paragraphs_soup)

        explanatory_statement = self._extract_explanatory_statement(soup)

        return {'resolution': resolution['full_content'], 'citations': resolution['citations'], 'recitals': resolution['recitals'], 'paragraphs': resolution['paragraphs'], 'explanatory_statement': explanatory_statement}



# Example usage
if __name__ == "__main__":
    scraper = EuropeanParliamentScraper()
    
    # Example URL
    url = "https://www.europarl.europa.eu/doceo/document/B-7-2012-0349_EN.html"
    
    try:
        print("Scraping motion from European Parliament...")
        motion_data = scraper.scrape_motion(url)
        
        print(f"Number of authors: {len(motion_data['authors'])}")
        print(f"First author: {motion_data['authors'][0] if motion_data['authors'] else 'N/A'}")
        print(motion_data['full_content'])
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
    except Exception as e:
        print(f"Error processing the motion: {e}")