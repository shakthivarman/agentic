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
base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(base_dir, 'downloaded_pdfs')
markdown_dir = os.path.join(base_dir, 'scraped_content')

# Create base directories
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(markdown_dir, exist_ok=True)

# Define the URLs to scrape
urls = [
    'https://www.astellas.com/in/',
    'https://www.astellas.com/in/about',
    'https://www.astellas.com/in/innovation',
    'https://www.astellas.com/in/innovation/clinical-trials',
    'https://www.astellas.com/in/innovation/rd',
    'https://www.astellas.com/in/innovation/rd/approach',
    'https://www.astellas.com/in/innovation/rd/ethics-principles',
    'https://www.astellas.com/in/innovation/rd/pipeline',
    'https://www.astellas.com/in/innovation/partnering',
    'https://www.astellas.com/in/innovation/grants-education-overview',
    'https://www.astellas.com/in/sustainability',
    'https://www.astellas.com/in/news',
    'https://www.astellas.com/in/products',
    'https://www.astellas.com/in/contact-us',
    'https://www.astellas.com/in/important-notice',
    'https://www.astellas.com/in/worldwide-old',
    'https://www.astellas.com/in/worldwide-india',
    'https://www.astellas.com/in/sitemap',
    'https://www.astellas.com/in/privacy-policy',
    'https://www.astellas.com/in/cookies-policy',
    'https://www.astellas.com/in/accessibility',
    'https://www.astellas.com/in/terms-of-use',
    'https://www.astellas.com/en/',
    'https://www.astellas.com/en/about',
    'https://www.astellas.com/en/about/corporate-strategic-plan-2021',
    'https://www.astellas.com/en/about/at-a-glance',
    'https://www.astellas.com/en/about/innovation',
    'https://www.astellas.com/en/about/sustainability-direction',
    'https://www.astellas.com/en/about/subsidiaries-locations',
    'https://www.astellas.com/en/about/our-business',
    'https://www.astellas.com/en/about/main-products',
    'https://www.astellas.com/en/about/history',
    'https://www.astellas.com/en/about/management',
    'https://www.astellas.com/en/rd',
    'https://www.astellas.com/en/rd/r-and-d-strategy',
    'https://www.astellas.com/en/rd/drug-research',
    'https://www.astellas.com/en/rd/astellas-institute-regenerative-medicine',
    'https://www.astellas.com/en/rd/universal-cells',
    'https://www.astellas.com/en/rd/xyphos-biosciences',
    'https://www.astellas.com/en/rd/immuno-oncology',
    'https://www.astellas.com/en/rd/astellas-gene-therapies',
    'https://www.astellas.com/en/rd/engineered-small-molecules',
    'https://www.astellas.com/en/rd/drug-discovery-platform',
    'https://www.astellas.com/en/rd/translational-research',
    'https://www.astellas.com/en/rd/regulatory-science',
    'https://www.astellas.com/en/rd/major-pipeline',
    'https://www.astellas.com/en/rd/clinical-trials',
    'https://www.astellas.com/en/rd/investigator-sponsored-research',
    'https://www.astellas.com/en/rd/grants-education-general-research',
    'https://www.astellas.com/en/rd/grants-hcos-support-hcp-attendance-congresses',
    'https://www.astellas.com/en/rd/primary-focus',
    'https://www.astellas.com/en/rd/primary-focus/genetic-regulation',
    'https://www.astellas.com/en/rd/primary-focus/blindness-regeneration',
    'https://www.astellas.com/en/rd/primary-focus/targeted-protein-degradation',
    'https://www.astellas.com/en/rd/primary-focus/cell-therapy',
    'https://www.astellas.com/en/rd/digital-transformation',
    'https://www.astellas.com/en/rd/digital-transformation/dx-initiatives-throughout-value-chain',
    'https://www.astellas.com/en/rd/digital-transformation/data-information-advanced-analytics',
    'https://www.astellas.com/en/rd/new-healthcare-solutions-beyond-medicine-rxplus',
    'https://www.astellas.com/en/rd/new-healthcare-solutions-beyond-medicine-rxplus/leading-programs',
    'https://www.astellas.com/en/rd/new-healthcare-solutions-beyond-medicine-rxplus/leading-programs/investigational-near-infrared-fluorescence-imaging-agent',
    'https://www.astellas.com/en/rd/new-healthcare-solutions-beyond-medicine-rxplus/leading-programs/clinically-relevant-holistic-solutions-mobile-healthcare-application',
    'https://www.astellas.com/en/rd/new-healthcare-solutions-beyond-medicine-rxplus/leading-programs/new-digital-solution-heart-failure-patients-horizon',
    'https://www.astellas.com/en/rd/open-innovation-drug-discovery-research',
    'https://www.astellas.com/en/rd/open-innovation-drug-discovery-research/our-philosophy-open-innovation',
    'https://www.astellas.com/en/rd/open-innovation-drug-discovery-research/our-labs',
    'https://www.astellas.com/en/rd/open-innovation-drug-discovery-research/our-labs/sakulab-tsukuba',
    'https://www.astellas.com/en/rd/open-innovation-drug-discovery-research/our-labs/tme-ilab',
    'https://www.astellas.com/en/rd/open-innovation-drug-discovery-research/our-labs/sakulab-cambridge-ma',
    'https://www.astellas.com/en/investors',
    'https://www.astellas.com/en/investors/business-results',
    'https://www.astellas.com/en/investors/overview-rd-pipeline',
    'https://www.astellas.com/en/investors/ir-meetings',
    'https://www.astellas.com/en/investors/consolidated-financial-statements-footnotes',
    'https://www.astellas.com/en/investors/corporate-governance-materials',
    'https://www.astellas.com/en/investors/shareholders-meeting',
    'https://www.astellas.com/en/investors/shareholder-return',
    'https://www.astellas.com/en/investors/current-share-status',
    'https://www.astellas.com/en/investors/chart',
    'https://www.astellas.com/en/investors/issuer-rating',
    'https://www.astellas.com/en/investors/analyst-coverage',
    'https://www.astellas.com/en/sustainability',
    'https://www.astellas.com/en/sustainability/hr-vision-talent-organization-management',
    'https://www.astellas.com/en/sustainability/engagement-diversity-equity-inclusion',
    'https://www.astellas.com/en/sustainability/a-living-wage',
    'https://www.astellas.com/en/sustainability/occupational-health-safety',
    'https://www.astellas.com/en/sustainability/promoting-health-management',
    'https://www.astellas.com/en/sustainability/major-activities-programs-japan',
    'https://www.astellas.com/en/sustainability/sustainable-procurement-initiatives',
    'https://www.astellas.com/en/sustainability/research-initiatives',
    'https://www.astellas.com/en/sustainability/clinical-development-initiatives',
    'https://www.astellas.com/en/sustainability/technological-development-production-initiatives',
    'https://www.astellas.com/en/sustainability/distribution-sales-initiatives',
    'https://www.astellas.com/en/sustainability/esg-data-environment',
    'https://www.astellas.com/en/sustainability/esg-data-social',
    'https://www.astellas.com/en/sustainability/esg-data-governance',
    'https://www.astellas.com/en/sustainability/our-people',
    'https://www.astellas.com/en/sustainability/sustainability-business-practice',
    'https://www.astellas.com/en/sustainability/respect-human-rights',
    'https://www.astellas.com/en/sustainability/community-engagement',
    'https://www.astellas.com/en/news',
    'https://www.astellas.com/en/news/all-news',
    'https://www.astellas.com/en/news/our-stories',
    'https://www.astellas.com/en/news/media-library',
    'https://www.astellas.com/en/partnering',
    'https://www.astellas.com/en/partnering/why-partner-astellas',
    'https://www.astellas.com/en/partnering/partnering-teams',
    'https://www.astellas.com/en/partnering/testimonies-our-partners',
    'https://www.astellas.com/en/partnering/registration-form',
    'https://www.astellas.com/en/careers',
    'https://www.astellas.com/en/suppliers',
    'https://www.astellas.com/en/contact-us',
    'https://www.astellas.com/en/worldwide',
    'https://www.astellas.com/en/sitemap',
    'https://www.astellas.com/en/accessibility',
    'https://www.astellas.com/en/disaster-information-employees',
    'https://www.astellas.com/us/',
    'https://www.astellas.com/us/about',
    'https://www.astellas.com/us/about/corporate-information',
    'https://www.astellas.com/us/about/corporate-information/company-facts',
    'https://www.astellas.com/us/about/corporate-information/astellas-fact-sheet',
    'https://www.astellas.com/us/about/corporate-information/integrated-report',
    'https://www.astellas.com/us/about/corporate-information/contact-us',
    'https://www.astellas.com/us/about/ethics-compliance',
    'https://www.astellas.com/us/about/ethics-compliance/governance',
    'https://www.astellas.com/us/about/ethics-compliance/group-code-of-conduct',
    'https://www.astellas.com/us/about/corporate-information/locations',
    'https://www.astellas.com/us/about/philosophy',
    'https://www.astellas.com/us/about/ethics-compliance/charter-corporate-conduct',
    'https://www.astellas.com/us/about/policies-position-statements',
    'https://www.astellas.com/us/about/political-contributions',
    'https://www.astellas.com/us/about/regulations',
    'https://www.astellas.com/us/about/regulations/state-regulations',
    'https://www.astellas.com/us/about/regulations/state-regulations/california-declaration-comprehensive-compliance-program',
    'https://www.astellas.com/us/about/regulations/state-regulations/vermonts-pharmaceutical-marketer-price-disclosure',
    'https://www.astellas.com/us/about/regulations/state-regulations/wholesale-acquisition-cost-information-colorado-prescribers',
    'https://www.astellas.com/us/about/regulations/state-regulations/wholesale-acquisition-cost-information-connecticut-prescribing-practitioners-pharmacists',
    'https://www.astellas.com/us/about/regulations/consumer-product-safety-commission-regulations',
    'https://www.astellas.com/us/about/us-leadership',
    'https://www.astellas.com/us/about/us-products',
    'https://www.astellas.com/us/medical-safety',
    'https://www.astellas.com/us/medical-safety/safety-data-sheets',
    'https://www.astellas.com/us/about/vision-strategy',
    'https://www.astellas.com/us/innovation',
    'https://www.astellas.com/us/innovation/clinical-trials',
    'https://www.astellas.com/us/innovation/areas-of-interest',
    'https://www.astellas.com/us/innovation/partnering',
    'https://www.astellas.com/us/innovation/rd',
    'https://www.astellas.com/us/innovation/rd/approach',
    'https://www.astellas.com/us/innovation/rd/ethics-principles',
    'https://www.astellas.com/us/innovation/rd/pipeline',
    'https://www.astellas.com/us/sustainability',
    'https://www.astellas.com/us/sustainability/changing-tomorrow-day',
    'https://www.astellas.com/us/sustainability/corporate-grants',
    'https://www.astellas.com/us/sustainability/covid-19-community-response',
    'https://www.astellas.com/us/sustainability/employee-impact',
    'https://www.astellas.com/us/news',
    'https://www.astellas.com/us/news/integrated-report',
    'https://www.astellas.com/us/news/articles',
    'https://www.astellas.com/us/news/media-inquiries',
    'https://www.astellas.com/us/news/news-releases',
    'https://www.astellas.com/us/news/stories',
    'https://www.astellas.com/us/news/stories/corporate',
    'https://www.astellas.com/us/news/stories/corporate-awards',
    'https://www.astellas.com/us/news/stories/corporate-social-responsibility',
    'https://www.astellas.com/us/news/stories/our-people',
    'https://www.astellas.com/us/news/stories/therapeutic-area-news',
    'https://www.astellas.com/us/news/statements',
    'https://www.astellas.com/us/news/video-gallery',
    'https://www.astellas.com/us/covid-19-response',
    'https://www.astellas.com/us/patient-focus',
    'https://www.astellas.com/us/patient-focus/support-for-patients',
    'https://www.astellas.com/us/patient-focus/astellas-pharma-support-solutions',
    'https://www.astellas.com/us/patient-focus/issues-that-matter-to-you',
    'https://www.astellas.com/us/patient-focus/patient-centricity',
    'https://www.astellas.com/us/patient-focus/patient-partnerships',
    'https://www.astellas.com/us/careers',
    'https://www.astellas.com/us/contact-us',
    'https://www.astellas.com/us/important-notice',
    'https://www.astellas.com/us/worldwide',
    'https://www.astellas.com/us/worldwide-us',
    'https://www.astellas.com/us/sitemap',
    'https://www.astellas.com/us/privacy-policy',
    'https://www.astellas.com/us/health-data-policy',
    'https://www.astellas.com/us/cookies-policy',
    'https://www.astellas.com/us/accessibility',
    'https://www.astellas.com/us/legal-disclaimer',
    'https://newsroom.astellas.us/2025-06-13-Mitsubishi-Research-Institute-and-Astellas-Announce-Collaboration-to-Support-Pharma-Startups-in-Japan',
    'https://newsroom.astellas.us/2025-05-29-Astellas-Enters-Exclusive-License-Agreement-with-Evopoint-Biosciences-for-XNW27011,-a-Novel-Clinical-stage-Antibody-Drug-Conjugate-Targeting-CLDN18-2',
    'https://newsroom.astellas.us/2025-05-28-Astellas-Announces-Second-Annual-Patient-Advocacy-Organization-PAO-Action-Week-TM-To-Empower-Patients-and-Caregivers',
    'https://newsroom.astellas.us/2025-05-28-Astellas-Announces-Second-Annual-Patient-Advocacy-Organization-PAO-Action-Week-TM-To-Empower-Patients-and-Caregivers#assets_31393_137629-136',
    'https://newsroom.astellas.us/2025-05-22-Astellas-and-Pfizers-XTANDI-TM-enzalutamide-Shows-Long-Term-Overall-Survival-in-Metastatic-Hormone-Sensitive-Prostate-Cancer',
    'https://newsroom.astellas.us/2025-05-21-Astellas-and-MBC-BioLabs-Announce-the-2025-Astellas-Future-Innovator-Prize-Awarded-to-DeepSeq-AI-Serna-Bio',
    'https://newsroom.astellas.us/2025-05-19-Astellas-Presents-New-Data-that-Explores-Potential-of-its-Cancer-Therapies-at-2025-ASCO-Annual-Meeting',
    'https://newsroom.astellas.us/2025-05-07-Notice-of-Nominees-for-Directors',
    'https://newsroom.astellas.us/2025-05-01-Astellas-to-Present-New-Data-in-Geographic-Atrophy-at-Upcoming-Ophthalmology-Annual-Congresses',
    'https://newsroom.astellas.us/2025-03-05-Astellas-and-YASKAWA-Agree-to-Establish-a-Joint-Venture-Focused-on-Cell-Therapy-Manufacturing',
    'https://newsroom.astellas.us/2025-02-27-Astellas-and-MBC-BioLabs-Announce-the-Sixth-Annual-Astellas-Future-Innovator-Prize',
    'https://www.astellasgenetherapies.com/'
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
            # Parse URL to get better filename
            parsed_url = urlparse(pdf_url)
            original_filename = parsed_url.path.split('/')[-1]
            
            # If no filename in URL, create one from the path
            if not original_filename or not original_filename.endswith('.pdf'):
                # Use path segments to create unique filename
                path_parts = [p for p in parsed_url.path.split('/') if p]
                if path_parts:
                    original_filename = '_'.join(path_parts[-2:]) + '.pdf'
                else:
                    original_filename = 'document.pdf'
            
            # Clean the filename but preserve more structure
            filename = original_filename.replace('%20', '_').replace(' ', '_')
            filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
            
            # Ensure it ends with .pdf
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
            
            # Add timestamp if file already exists to prevent overwriting
            pdf_name = os.path.join(pdf_dir, filename)
            counter = 1
            base_name = filename[:-4]  # Remove .pdf extension
            while os.path.exists(pdf_name):
                new_filename = f"{base_name}_{counter}.pdf"
                pdf_name = os.path.join(pdf_dir, new_filename)
                counter += 1
            
            # Download and save
            with open(pdf_name, 'wb') as pdf_file:
                pdf_file.write(response.content)
            print(f'✅ Downloaded PDF: {pdf_name} ({len(response.content)} bytes)')
        else:
            print(f'❌ Failed to download PDF {pdf_url}: Status code {response.status_code}')
    except Exception as e:
        print(f'❌ Error downloading PDF {pdf_url}: {str(e)}')

def save_markdown(url, content):
    try:
        # Check for error content before saving
        if is_error_content(content):
            print(f"Skipping {url} - contains error content (Page not found, etc.)")
            return False
            
        # Parse the URL to create a descriptive filename
        parsed_url = urlparse(url)
        path_parts = [p for p in parsed_url.path.split('/') if p and p != 'index.php']
        
        # Create a clean filename from the path
        if path_parts:
            # Join all path parts with hyphens to create a descriptive filename
            filename = '-'.join(path_parts) + '.md'
        else:
            filename = 'index.md'
            
        # Remove any problematic characters from filename
        filename = ''.join(c for c in filename if c.isalnum() or c in '.-')
        
        # Save directly to scraped_content directory
        filepath = os.path.join(markdown_dir, filename)
        
        # Write the content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Scraped content from {url}\n\n")
            f.write(content)
        
        print(f"Saved: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving markdown for {url}: {str(e)}")
        return False

def is_error_content(content):
    """Check if the content contains error messages that indicate failed page loading."""
    if not content or len(content.strip()) < 50:
        return True
    
    content_lower = content.lower()
    
    # Common error indicators
    error_patterns = [
        'page not found',
        'the requested page could not be found',
        'error 404',
        'error 403',
        'error 500',
        'access denied',
        'forbidden',
        'unauthorized',
        'service unavailable',
        'page is temporarily unavailable',
        'sorry, the page you are looking for',
        'this page does not exist',
        'page unavailable',
        'content not available'
    ]
    
    # Check for error patterns
    for pattern in error_patterns:
        if pattern in content_lower:
            return True
    
    # Check if content is too short and likely just navigation/error
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    substantial_lines = [line for line in lines if len(line) > 20]
    
    if len(substantial_lines) < 3:
        return True
    
    # Check if content is mostly navigation words
    words = content_lower.split()
    if len(words) < 50:  # Very short content
        return True
    
    navigation_words = {'menu', 'navigation', 'home', 'about', 'contact', 
                       'search', 'login', 'register', 'subscribe', 'follow',
                       'facebook', 'twitter', 'linkedin', 'instagram',
                       'privacy', 'terms', 'cookie', 'legal', 'disclaimer'}
    
    nav_word_count = sum(1 for word in words if word in navigation_words)
    
    # If more than 40% navigation words in short content, likely error/navigation page
    if len(words) < 100 and nav_word_count / len(words) > 0.4:
        return True
    
    return False

async def process_url_with_retry(url, max_retries=3):
    """Process a single URL with retry logic and timeout handling"""
    for attempt in range(max_retries):
        if not can_scrape(url):
            print(f"Scraping not allowed for: {url}")
            return None
        
        try:
            print(f"Processing {url} (attempt {attempt + 1}/{max_retries})")
            
            # Configure crawler with shorter timeout and better error handling
            async with AsyncWebCrawler(
                headless=True,
                browser_args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--memory-pressure-off'
                ],
                timeout=30000,  # 30 seconds timeout
                wait_for='domcontentloaded',
                delay_before_return_html=2.0
            ) as crawler:
                result = await crawler.arun(url=url)
                
                if result and result.markdown:
                    # Extract PDF links
                    content = result.markdown
                    pdf_links = []
                    
                    # Look for PDF links in markdown content
                    for line in content.split('\n'):
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
                    if pdf_links:
                        print(f"Found {len(pdf_links)} PDF links for {url}")
                        
                        # Download PDFs (limit to prevent overwhelming)
                        for pdf_link in pdf_links[:5]:  # Limit to 5 PDFs per page
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
                    if save_markdown(url, result.markdown):
                        print(f"✅ Successfully processed {url}")
                        return result.markdown
                    else:
                        print(f"⚠️ Skipped {url} - error content detected")
                        return None
                else:
                    print(f"⚠️ No content retrieved for {url}")
                    return None
                    
        except asyncio.TimeoutError:
            print(f"⏰ Timeout on {url} (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                print(f"❌ Failed to process {url} after {max_retries} attempts (timeout)")
                return None
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            print(f"❌ Error scraping {url} (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"❌ Failed to process {url} after {max_retries} attempts")
                return None
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return None

async def process_batch(batch_urls, batch_num, total_batches):
    """Process a batch of URLs with limited concurrency"""
    print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch_urls)} URLs)")
    
    # Use semaphore to limit concurrent connections
    semaphore = asyncio.Semaphore(3)  # Only 3 concurrent requests
    
    async def process_with_semaphore(url):
        async with semaphore:
            return await process_url_with_retry(url)
    
    # Process batch with limited concurrency
    tasks = [process_with_semaphore(url) for url in batch_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successful vs failed
    successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    failed = len(results) - successful
    
    print(f"✅ Batch {batch_num} complete: {successful} successful, {failed} failed")
    return results

async def main():
    print(f"🚀 Starting to scrape {len(urls)} URLs from Astellas")
    
    # Split URLs into smaller batches to prevent memory issues
    batch_size = 10  # Process 10 URLs at a time
    batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
    total_batches = len(batches)
    
    print(f"📦 Split into {total_batches} batches of {batch_size} URLs each")
    
    all_results = []
    successful_count = 0
    failed_count = 0
    
    for i, batch in enumerate(batches, 1):
        try:
            # Add delay between batches to be respectful to the server
            if i > 1:
                print(f"⏳ Waiting 5 seconds between batches...")
                await asyncio.sleep(5)
            
            batch_results = await process_batch(batch, i, total_batches)
            all_results.extend(batch_results)
            
            # Count results
            batch_successful = sum(1 for r in batch_results if r is not None and not isinstance(r, Exception))
            batch_failed = len(batch_results) - batch_successful
            successful_count += batch_successful
            failed_count += batch_failed
            
            print(f"📊 Progress: {successful_count} successful, {failed_count} failed so far")
            
        except Exception as e:
            print(f"❌ Error processing batch {i}: {str(e)}")
            failed_count += len(batch)
    
    # Final summary
    print(f"\n🎯 Final Results:")
    print(f"✅ Successfully processed: {successful_count} URLs")
    print(f"❌ Failed: {failed_count} URLs")
    print(f"📁 Content saved to: {markdown_dir}")
    
    return all_results

if __name__ == "__main__":
    asyncio.run(main())
