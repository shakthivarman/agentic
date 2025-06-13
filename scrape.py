import os
import asyncio
import requests
from urllib.parse import urlparse, unquote
from crawl4ai import AsyncWebCrawler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

# Create a session with retry strategy and headers
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
session.verify = False  # Disable SSL verification

# Suppress SSL verification warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Headers to mimic a real browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

# Directory to save downloaded PDFs and markdown files
pdf_dir = 'downloaded_pdfs'
markdown_dir = 'scraped_content'
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(markdown_dir, exist_ok=True)

# Define the URLs to scrape
urls = [
    'https://www.hcltech.com/ai-readiness-guide-genai-with-a-smart-cloud-strategy',
    'https://www.hcltech.com/ai-and-ml-services',
    'https://www.hcltech.com/engineering/ai-services',
    'https://www.hcltech.com/ai',
    'https://www.hcltech.com/hcl-annual-report-2024/artificial-intelligence',
    'https://www.hcltech.com/generative-ai-services',
    'https://www.hcltech.com/brochures/ai-and-generative-ai',
    'https://www.hcltech.com/life-sciences-healthcare/generative-ai-impact',
    'https://www.hcltech.com/ai-force',
    'https://www.hcltech.com/analyst-reports/hcltech-ai-force-scalable-modular-and-backed-proven-ai-expertise'
]

def can_scrape(url):
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        response = session.get(robots_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            lines = response.text.split('\n')
            for line in lines:
                if line.strip().lower().startswith('user-agent: *'):
                    if 'disallow: /' in line.lower():
                        return False
            return True
        return True  # If we can't access robots.txt, assume we can scrape
    except Exception as e:
        print(f"Error checking robots.txt for {url}: {str(e)}")
        return True  # If we can't access robots.txt, assume we can scrape

def download_pdf(pdf_url):
    try:
        response = session.get(pdf_url, headers=headers, timeout=30)
        if response.status_code == 200:
            # Clean the filename
            filename = pdf_url.split('/')[-1].replace('%20', '_')
            filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
            pdf_name = os.path.join(pdf_dir, filename)
            
            with open(pdf_name, 'wb') as pdf_file:
                pdf_file.write(response.content)
            print(f'Downloaded: {pdf_name}')
        else:
            print(f'Failed to download {pdf_url}: Status code {response.status_code}')
    except Exception as e:
        print(f'Error downloading {pdf_url}: {str(e)}')

def save_markdown(url, content):
    # Create a filename from the URL
    filename = unquote(url.split('/')[-1].replace('.html', '.md'))
    filepath = os.path.join(markdown_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Scraped content from {url}\n\n")
        f.write(content)
    print(f"Saved markdown content to: {filepath}")

async def process_url(url):
    if not can_scrape(url):
        print(f"Scraping not allowed for: {url}")
        return None
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            
            # Extract PDF links using a more comprehensive approach
            content = result.markdown
            pdf_links = []
            
            # Look for PDF links in markdown content
            for line in content.split('\n'):
                # Check for markdown links ending in .pdf
                if '.pdf' in line.lower():
                    # Extract URLs from markdown links [text](url)
                    if '(' in line and ')' in line:
                        start = line.find('(') + 1
                        end = line.find(')', start)
                        pdf_url = line[start:end]
                        if pdf_url.lower().endswith('.pdf'):
                            pdf_links.append(pdf_url)
                    # Extract direct URLs
                    words = line.split()
                    for word in words:
                        if word.lower().endswith('.pdf'):
                            pdf_links.append(word)
            
            # Remove duplicates and clean URLs
            pdf_links = list(set(pdf_links))
            print(f"Found {len(pdf_links)} PDF links")
            
            # Download PDFs
            for pdf_link in pdf_links:
                try:
                    # Ensure the URL is absolute
                    if not pdf_link.startswith(('http://', 'https://')):
                        base_url = '/'.join(url.split('/')[:-1])
                        pdf_link = f"{base_url}/{pdf_link.lstrip('/')}"
                    print(f"Downloading PDF: {pdf_link}")
                    download_pdf(pdf_link)
                except Exception as e:
                    print(f"Error downloading {pdf_link}: {str(e)}")
            
            # Save markdown content
            save_markdown(url, result.markdown)
            
            return result.markdown
            
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

async def main():
    tasks = [process_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    # Print summary for each URL
    for url, result in zip(urls, results):
        if result:
            print(f"\nProcessed {url}")
            print(f"Content length: {len(result)} characters")

if __name__ == "__main__":
    asyncio.run(main())
